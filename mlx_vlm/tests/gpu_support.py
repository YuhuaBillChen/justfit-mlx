"""Apple-GPU capability helpers shared by the Metal kernel tests.

The fused MTP qtile kernel dispatches a 1024-thread threadgroup
(``MLX_VLM_TQ_MTP_QTILE_SIMDGROUPS=32`` simd groups of 32 threads). A kernel's
maximum threadgroup size is limited by its register and threadgroup-memory
footprint, so at the production head dimension (256) pre-M3 Apple GPUs reject
the launch outright. These helpers let the affected tests skip instead of
failing on that hardware.
"""

import re

import mlx.core as mx

# ``architecture`` is reported as ``applegpu_gNN`` plus a variant suffix -- an
# M4 Pro reports ``applegpu_g16s``. The generation is 13=M1, 14=M2, 15=M3,
# 16=M4. MLX reads the same field with ``stoi``, which likewise stops at the
# suffix, so match a prefix rather than the whole string.
_MIN_QTILE_ARCHITECTURE_GEN = 15


def _metal_available():
    return hasattr(mx, "metal") and mx.metal.is_available()


def _gpu_architecture_gen():
    architecture = mx.device_info().get("architecture", "")
    match = re.match(r"applegpu_g(\d+)", architecture)
    return int(match.group(1)) if match else None


def qtile_threadgroup_supported():
    """True when the MTP qtile kernel's 1024-thread threadgroup can launch."""
    if not _metal_available():
        return False
    generation = _gpu_architecture_gen()
    # An unrecognized architecture string means the probe broke, not that the
    # GPU is old. Assume the kernel launches so that shows up as a failure
    # rather than as a silent skip on every machine.
    return generation is None or generation >= _MIN_QTILE_ARCHITECTURE_GEN
