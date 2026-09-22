"""Backend-independent component leases for the serialized generation worker.

This module tracks logical ownership, not physical memory or GPU completion.
The execution scheduler must finish pending uses before releasing a lease.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Hashable, Protocol, TypeVar

logger = logging.getLogger("mlx_vlm.server")


class ResidencyComponent(Protocol):
    """An idempotent component adapter; no assumption about its backing store.

    load() must make the component usable before returning. unload() detaches
    its owned references; aliases or allocator caches can still retain storage.
    Neither operation implies memory has been returned to the operating system.
    """

    def load(self) -> object: ...

    def unload(self) -> None: ...


ComponentT = TypeVar("ComponentT", bound=ResidencyComponent)


class ComponentResidencyManager:
    """Coordinate named, idempotent leases shared by one GPU scheduler.

    An owner name identifies one logical lease, not a reference count: repeated
    acquire calls with that name do not nest. Callers must not release leases
    belonging to successor generators that reuse the same name.

    Calls are serialized by the generation worker. There are deliberately no
    locks, GPU synchronizations, memory sampling or eviction policy here.
    """

    def __init__(self) -> None:
        self._components: dict[str, ResidencyComponent] = {}
        self._owners: dict[str, set[Hashable]] = defaultdict(set)
        self._dependencies: dict[str, tuple[str, ...]] = {}
        self._dependency_owners: dict[str, object] = {}
        self._retained: set[str] = set()

    def register(
        self,
        name: str,
        component: ComponentT,
        *,
        dependencies: tuple[str, ...] = (),
        retain_on_release: bool = False,
    ) -> ComponentT:
        """Register dependencies first; immutable edges cannot form a cycle."""
        dependencies = tuple(dependencies)
        existing = self._components.get(name)
        if existing is not None and (
            existing is not component
            or self._dependencies[name] != dependencies
            or (name in self._retained) != retain_on_release
        ):
            raise ValueError(f"Phase component already registered: {name}")
        if existing is not None:
            return component
        if any(dep == name or dep not in self._components for dep in dependencies):
            raise ValueError(f"Register dependencies before component: {name}")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"Duplicate component dependencies: {name}")
        self._components[name] = component
        self._dependencies[name] = dependencies
        self._dependency_owners[name] = object()
        if retain_on_release:
            self._retained.add(name)
        return component

    def contains(self, name: str) -> bool:
        return name in self._components

    def owners(self, name: str) -> frozenset[Hashable]:
        return frozenset(self._owners.get(name, ()))

    def acquire(self, name: str, owner: Hashable) -> ResidencyComponent:
        component = self._components[name]
        owners = self._owners[name]
        if owner in owners:
            return component
        # Publish ownership only after the adapter successfully loads.
        self.ensure_loaded(name)
        owners.add(owner)
        logger.info(
            "Phase component acquired: component=%s owner=%s leases=%d",
            name,
            owner,
            len(owners),
        )
        return component

    def ensure_loaded(self, name: str) -> ResidencyComponent:
        """Restore without inventing a lease (e.g. retained embedding idle state)."""
        component = self._components[name]
        dependency_owner = self._dependency_owners[name]
        acquired = []
        try:
            for dependency in self._dependencies[name]:
                if dependency_owner not in self._owners[dependency]:
                    self.acquire(dependency, dependency_owner)
                    acquired.append(dependency)
            component.load()
        except BaseException:
            for dependency in reversed(acquired):
                try:
                    self.release(dependency, dependency_owner)
                except Exception:
                    logger.exception(
                        "Failed to roll back component dependency: %s", dependency
                    )
            raise
        return component

    def release(self, name: str, owner: Hashable) -> bool:
        """Consume a lease; return True only if unload completed.

        If unload raises, the lease remains consumed and the error propagates.
        Call unload_if_idle to retry reclamation: resurrecting an old owner
        would interfere with subsequent cohorts. No physical release is claimed.
        """
        component = self._components.get(name)
        if component is None:
            return False
        owners = self._owners[name]
        if owner not in owners:
            return False
        owners.remove(owner)
        if owners or name in self._retained:
            return False
        self._unload(name)
        logger.info("Phase component idle: component=%s", name)
        return True

    def unload_if_idle(self, name: str) -> bool:
        component = self._components.get(name)
        if component is None or self._owners.get(name):
            return False
        self._unload(name)
        return True

    def _unload(self, name: str) -> None:
        # A failed detach must keep its dependencies pinned until a safe retry.
        self._components[name].unload()
        for dependency in reversed(self._dependencies[name]):
            self.release(dependency, self._dependency_owners[name])
