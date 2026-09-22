import io

import mlx.core as mx
import pytest

from mlx_vlm.apc_single_pass import PayloadSink, save_single_pass


def test_single_pass_matches_mlx_for_noncontiguous_dtypes(tmp_path):
    arrays = {
        str(i): mx.arange(48).astype(dtype).reshape(4, 12)[:, ::3]
        for i, dtype in enumerate(
            [
                mx.float16,
                mx.bfloat16,
                mx.float32,
                mx.int32,
                mx.uint32,
                mx.uint8,
                mx.bool_,
            ]
        )
    }
    path = tmp_path / "single.safetensors"
    save_single_pass(path, arrays, metadata={"test": "yes"})
    restored = mx.load(str(path))
    assert set(restored) == set(arrays)
    for key, value in arrays.items():
        assert restored[key].dtype == value.dtype
        assert mx.array_equal(restored[key], value).item()
    assert list(tmp_path.iterdir()) == [path]


def test_failure_removes_only_partial_destination(tmp_path, monkeypatch):
    path = tmp_path / "partial.safetensors"
    original = mx.save_safetensors

    def fail(file, arrays, **kwargs):
        if isinstance(file, PayloadSink):
            raise OSError("injected writer failure")
        return original(file, arrays, **kwargs)

    monkeypatch.setattr(mx, "save_safetensors", fail)
    with pytest.raises(OSError, match="injected"):
        save_single_pass(path, {"a": mx.ones((8,))}, metadata={})
    assert not path.exists()


def test_sink_forwards_exactly_one_payload_and_rejects_truncation():
    array = mx.arange(16, dtype=mx.uint32)
    encoded = io.BytesIO()
    mx.save_safetensors(encoded, {"a": array})
    payload = encoded.getvalue()
    output = io.BytesIO()
    sink = PayloadSink(
        output, {"a": dict(dtype="U32", shape=[16], data_offsets=[0, 64])}
    )
    for start in range(0, len(payload), 3):
        sink.write(payload[start : start + 3])
    sink.finish()
    assert len(output.getvalue()) == 64
    incomplete = PayloadSink(io.BytesIO(), sink.expected)
    incomplete.write(payload[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        incomplete.finish()
