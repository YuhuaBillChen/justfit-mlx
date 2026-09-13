"""Request-phase lifecycle helpers for vision towers."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger("mlx_vlm.server")


class VisionTowerPhaseSwap:
    """Load a standalone vision tower only while media embeddings are built."""

    def __init__(self, model: Any, component_path: str) -> None:
        tower = getattr(model, "vision_tower", None)
        if tower is None:
            raise ValueError("Vision phase swap requires model.vision_tower")
        path = Path(component_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Vision component not found: {path}")
        self.model = model
        self.component_path = path
        self.tower_class = type(tower)
        self.tower_config = tower.config

        model.vision_tower = None
        del tower
        gc.collect()
        mx.clear_cache()
        logger.info("Vision tower released after target initialization.")

    @property
    def loaded(self) -> bool:
        return getattr(self.model, "vision_tower", None) is not None

    def load(self) -> Any:
        if self.loaded:
            return self.model.vision_tower
        started = time.perf_counter()
        tower = self.tower_class(self.tower_config)
        weights = mx.load(str(self.component_path))
        tower.load_weights(list(weights.items()), strict=True)
        mx.eval(tower.parameters())
        tower.eval()
        self.model.vision_tower = tower
        del weights
        logger.info(
            "Vision tower loaded for media embedding: elapsed=%.3fs",
            time.perf_counter() - started,
        )
        return tower

    def unload(self) -> None:
        tower = getattr(self.model, "vision_tower", None)
        if tower is None:
            return
        self.model.vision_tower = None
        del tower
        gc.collect()
        mx.clear_cache()
        logger.info("Vision tower released after media embedding.")
