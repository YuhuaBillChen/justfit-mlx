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
    used by singleton speculative cohorts; the worker may also serve AR peers.
    Residency leases belong to the cohort, not to this reusable loader.
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
        # Host scalars only: unloading weights must not erase the final SSE
        # statistics or reset snapshot/diff accounting across re-promotion.
        self._retired_stats = {
            "speculative_total_rounds": 0,
            "speculative_total_accepted": 0.0,
            "speculative_total_drafted": 0,
        }

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def materialize(self):
        """Compatibility entry point for non-server speculative callers."""
        return self.load()

    def load(self):
        """ResidencyComponent adapter: load and validate before publishing."""
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
        self._retired_stats = {
            name: total + getattr(self._model, name, 0)
            for name, total in self._retired_stats.items()
        }
        model, self._model = self._model, None
        del model
        gc.collect()
        mx.clear_cache()
        logger.info("Deferred drafter unloaded after decode.")

    def __getattr__(self, name: str):
        model = self.__dict__.get("_model")
        retired_stats = self.__dict__.get("_retired_stats", {})
        if name in retired_stats:
            live = getattr(model, name, 0) if model is not None else 0
            return retired_stats[name] + live
        if model is None:
            if name in {"accept_lens", "draft_lens"}:
                return []
            raise AttributeError(name)
        return getattr(model, name)
