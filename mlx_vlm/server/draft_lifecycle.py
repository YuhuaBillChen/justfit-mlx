"""Request-phase lifecycle helpers for speculative drafters."""

from __future__ import annotations

import gc
import logging
import time
from typing import Any, Callable, Optional

import mlx.core as mx

logger = logging.getLogger("mlx_vlm.server")


class LazyDrafter:
    """Materialize drafter weights when speculative decode first needs them.

    The wrapper retains the already-resolved drafter metadata while allowing
    the loaded weights to be released during target-model prefill. It is
    intentionally single-owner; the server only enables it for a one-sequence
    generation worker.
    """

    def __init__(
        self,
        *,
        path: str,
        kind: str,
        config: Any,
        loader: Callable[[str, Optional[str]], tuple[Any, str]],
        validator: Callable[[Any, Any, str], None],
        target_model: Any,
    ) -> None:
        self.path = str(path)
        self.kind = str(kind)
        self.config = config
        self._loader = loader
        self._validator = validator
        self._target_model = target_model
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def materialize(self):
        if self._model is not None:
            return self._model
        started = time.perf_counter()
        model, resolved_kind = self._loader(self.path, self.kind)
        if resolved_kind != self.kind:
            raise ValueError(
                f"Lazy drafter resolved as {resolved_kind!r}, expected {self.kind!r}"
            )
        self._validator(self._target_model, model, resolved_kind)
        self._model = model
        logger.info(
            "Deferred drafter loaded for decode: kind=%s elapsed=%.3fs",
            resolved_kind,
            time.perf_counter() - started,
        )
        return model

    def unload(self) -> None:
        if self._model is None:
            return
        model, self._model = self._model, None
        del model
        gc.collect()
        mx.clear_cache()
        logger.info("Deferred drafter unloaded after decode.")

    def __getattr__(self, name: str):
        model = self.__dict__.get("_model")
        if model is None:
            if name in {"accept_lens", "draft_lens"}:
                return []
            if name in {
                "speculative_total_rounds",
                "speculative_total_accepted",
                "speculative_total_drafted",
            }:
                return 0
            raise AttributeError(name)
        return getattr(model, name)
