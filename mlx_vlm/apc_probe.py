"""Opt-in host-boundary APC diagnostics; never synchronizes GPU work."""

import json
import os
import time
from contextlib import contextmanager
from functools import wraps


@contextmanager
def stage(name):
    if os.environ.get("APC_STAGE_PROBE") != "1":
        yield
        return
    import mlx.core as mx

    started = time.monotonic()

    def emit(event, **extra):
        print(
            "APC_STAGE "
            + json.dumps(
                dict(
                    stage=name,
                    event=event,
                    monotonic=time.monotonic(),
                    wall_time=time.time(),
                    pid=os.getpid(),
                    active_bytes=mx.get_active_memory(),
                    cache_bytes=mx.get_cache_memory(),
                    peak_bytes=mx.get_peak_memory(),
                    gpu_synchronized=False,
                    **extra,
                )
            ),
            flush=True,
        )

    emit("begin")
    try:
        yield
    except BaseException:
        emit("error", elapsed_seconds=time.monotonic() - started)
        raise
    else:
        emit("end", elapsed_seconds=time.monotonic() - started)


def traced(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)

        return wrapped

    return decorate
