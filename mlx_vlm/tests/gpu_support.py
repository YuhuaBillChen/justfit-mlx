"""Helpers for tests that dispatch wide Metal threadgroups.

Several paged TurboQuant kernels dispatch a 1024-thread threadgroup. The
maximum threadgroup size is a per-kernel limit that falls as the kernel's
register and threadgroup-memory footprint grows, so at head_dim 256 some Apple
GPUs refuse the launch even though the device itself allows 1024 threads. It is
not a property of the chip generation, and MLX reports it only when the graph is
evaluated, so tests react to the error rather than inspecting the device.
"""

import functools

import pytest


def skip_if_threadgroup_refused(test):
    """Skip the test when this GPU rejects a kernel's threadgroup size."""

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        try:
            return test(*args, **kwargs)
        except ValueError as exc:
            if "threads per threadgroup" not in str(exc):
                raise
            pytest.skip(f"this GPU rejects the kernel's threadgroup: {exc}")

    return wrapper
