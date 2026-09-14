"""Immutable execution policy for a paged TurboQuant cache or generator.

This module has no MLX dependency. Environment variables are a compatibility
input at construction time, never a source of per-token dispatch decisions.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class PagedTurboQuantConfig:
    """Execution choices shared by every request in one page pool.

    Explicit construction does not consult the environment. Defaults preserve
    the legacy behavior with all three environment overrides absent. Eager
    release evaluates prefill output before returning to bound temporary tensor
    lifetimes. Direct-inverse prefill retains precedence over MTP qtile dispatch.
    """

    prefill_impl: Literal["compatibility", "direct_inverse"] = "compatibility"
    prefill_eager_release: bool = False
    mtp_qtile: bool = False

    def __post_init__(self) -> None:
        if self.prefill_impl not in ("compatibility", "direct_inverse"):
            raise ValueError("prefill_impl must be 'compatibility' or 'direct_inverse'")
        for name in ("prefill_eager_release", "mtp_qtile"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PagedTurboQuantConfig:
        """Snapshot legacy overrides, rejecting typos before allocating a pool.

        Boolean overrides accept '0', '1', or an empty value (disabled).
        Passing a mapping permits isolated configuration without changing the
        process environment.
        """

        source = dict(os.environ if environ is None else environ)

        def flag(name: str) -> bool:
            value = source.get(name, "0")
            if value not in ("", "0", "1"):
                raise ValueError(f"{name} must be '0' or '1', got {value!r}")
            return value == "1"

        return cls(
            prefill_impl=source.get("MLX_VLM_PAGED_PREFILL_IMPL") or "compatibility",
            prefill_eager_release=flag("MLX_VLM_PAGED_PREFILL_EAGER_RELEASE"),
            mtp_qtile=flag("MLX_VLM_TQ_MTP_QTILE"),
        )

    def to_dict(self) -> dict[str, str | bool]:
        """Return effective settings suitable for experiment JSON or logging."""

        return asdict(self)
