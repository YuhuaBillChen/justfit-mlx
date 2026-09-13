"""Request-phase lifecycle helpers for language-model output heads."""

from __future__ import annotations

import gc
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Set

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger("mlx_vlm.server")


class ComponentResidencyManager:
    """Coordinate phase-swapped components shared by one GPU scheduler.

    Components expose idempotent ``load()`` and ``unload()`` methods. Owners
    hold named leases while a component is required; an idle component may be
    swapped out only after its last lease is released. The generation worker
    is the sole caller, so transitions remain serialized with model forwards.
    """

    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}
        self._owners: Dict[str, Set[str]] = defaultdict(set)

    def register(self, name: str, component: Any) -> Any:
        existing = self._components.get(name)
        if existing is not None and existing is not component:
            raise ValueError(f"Phase component already registered: {name}")
        self._components[name] = component
        return component

    def contains(self, name: str) -> bool:
        return name in self._components

    def owners(self, name: str) -> frozenset[str]:
        return frozenset(self._owners.get(name, ()))

    def acquire(self, name: str, owner: str) -> Any:
        component = self._components[name]
        owners = self._owners[name]
        if owner in owners:
            return component
        component.load()
        owners.add(owner)
        logger.info(
            "Phase component acquired: component=%s owner=%s leases=%d",
            name,
            owner,
            len(owners),
        )
        return component

    def release(self, name: str, owner: str) -> bool:
        component = self._components.get(name)
        if component is None:
            return False
        owners = self._owners[name]
        if owner not in owners:
            return False
        owners.remove(owner)
        if owners:
            return False
        component.unload()
        logger.info("Phase component idle: component=%s", name)
        return True

    def unload_if_idle(self, name: str) -> bool:
        component = self._components.get(name)
        if component is None or self._owners.get(name):
            return False
        component.unload()
        return True


class LanguageHeadPhaseSwap:
    """Keep an untied LM head off-device during intermediate prompt chunks."""

    def __init__(self, language_model: Any, component_path: str) -> None:
        head = getattr(language_model, "lm_head", None)
        if head is None:
            raise ValueError("Language-head phase swap requires an untied lm_head")
        path = Path(component_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Language-head component not found: {path}")
        self.language_model = language_model
        self.component_path = path
        self.head_class = type(head)
        self.group_size = getattr(head, "group_size", None)
        self.bits = getattr(head, "bits", None)
        self.mode = getattr(head, "mode", None)
        self.has_bias = getattr(head, "bias", None) is not None
        self.biases_was_none = (
            hasattr(head, "biases") and getattr(head, "biases") is None
        )

    @property
    def loaded(self) -> bool:
        return getattr(self.language_model, "lm_head", None) is not None

    def load(self) -> Any:
        if self.loaded:
            return self.language_model.lm_head
        started = time.perf_counter()
        weights = mx.load(str(self.component_path))
        head = self.head_class.__new__(self.head_class)
        nn.Module.__init__(head)
        if self.group_size is not None:
            head.group_size = self.group_size
        if self.bits is not None:
            head.bits = self.bits
        if self.mode is not None:
            head.mode = self.mode
        for name, value in weights.items():
            setattr(head, name, value)
        if self.biases_was_none and not hasattr(head, "biases"):
            head.biases = None
        if not self.has_bias and not hasattr(head, "bias"):
            head.bias = None
        freeze = getattr(head, "freeze", None)
        if callable(freeze):
            freeze()
        mx.eval(head.parameters())
        self.language_model.lm_head = head
        del weights
        logger.info(
            "Language head restored for final prefill/decode: elapsed=%.3fs",
            time.perf_counter() - started,
        )
        return head

    def unload(self) -> None:
        head = getattr(self.language_model, "lm_head", None)
        if head is None:
            return
        active_before = mx.get_active_memory()
        self.language_model.lm_head = None
        del head
        gc.collect()
        mx.clear_cache()
        active_after = mx.get_active_memory()
        logger.info(
            "Language head released during intermediate prefill: "
            "active_before_mib=%.2f active_after_mib=%.2f released_mib=%.2f",
            active_before / (1024**2),
            active_after / (1024**2),
            max(0, active_before - active_after) / (1024**2),
        )


class LanguageEmbeddingPhaseSwap:
    """Release an input embedding table after warm-suffix embeddings spill."""

    def __init__(self, language_model: Any, component_path: str) -> None:
        owner = getattr(language_model, "model", None)
        embedding = getattr(owner, "embed_tokens", None)
        if embedding is None:
            raise ValueError("Embedding phase swap requires model.embed_tokens")
        path = Path(component_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Embedding component not found: {path}")
        self.owner = owner
        self.component_path = path
        self.embedding_class = type(embedding)
        self.group_size = getattr(embedding, "group_size", None)
        self.bits = getattr(embedding, "bits", None)
        self.mode = getattr(embedding, "mode", None)
        self.num_embeddings = getattr(embedding, "num_embeddings", None)
        self.dims = getattr(embedding, "dims", None)
        self.biases_was_none = (
            hasattr(embedding, "biases") and getattr(embedding, "biases") is None
        )

    @property
    def loaded(self) -> bool:
        return getattr(self.owner, "embed_tokens", None) is not None

    def load(self) -> Any:
        if self.loaded:
            return self.owner.embed_tokens
        started = time.perf_counter()
        weights = mx.load(str(self.component_path))
        embedding = self.embedding_class.__new__(self.embedding_class)
        nn.Module.__init__(embedding)
        for name in ("group_size", "bits", "mode", "num_embeddings", "dims"):
            value = getattr(self, name)
            if value is not None:
                setattr(embedding, name, value)
        for name, value in weights.items():
            setattr(embedding, name, value)
        if self.biases_was_none and not hasattr(embedding, "biases"):
            embedding.biases = None
        freeze = getattr(embedding, "freeze", None)
        if callable(freeze):
            freeze()
        mx.eval(embedding.parameters())
        self.owner.embed_tokens = embedding
        del weights
        logger.info(
            "Input embedding restored for decode: elapsed=%.3fs",
            time.perf_counter() - started,
        )
        return embedding

    def unload(self) -> None:
        embedding = getattr(self.owner, "embed_tokens", None)
        if embedding is None:
            return
        active_before = mx.get_active_memory()
        self.owner.embed_tokens = None
        del embedding
        gc.collect()
        mx.clear_cache()
        active_after = mx.get_active_memory()
        logger.info(
            "Input embedding released during warm prefill: "
            "active_before_mib=%.2f active_after_mib=%.2f released_mib=%.2f",
            active_before / (1024**2),
            active_after / (1024**2),
            max(0, active_before - active_after) / (1024**2),
        )
