"""Single-destination safetensors assembly using MLX's payload encoder."""
import io
import json

import mlx.core as mx

from .apc_probe import stage


class PayloadSink:
    """Consume one MLX safetensors stream, forwarding only its tensor payload.

    Headers are small Python buffers. Tensor payload memoryviews are written
    directly; no tensor-sized Python copy or temporary disk shard is retained.
    """
    closed = False

    def __init__(self, destination, expected):
        self.destination = destination
        self.expected = expected
        self.header = bytearray()
        self.header_size = None
        self.position = 0
        self.payload_bytes = 0
        self.validated = False

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset
        if whence not in (0, 1) or target != self.position:
            raise OSError('MLX serializer unexpectedly requested non-sequential IO')
        return self.position

    def write(self, data):
        view = memoryview(data).cast('B')
        size = len(view)
        while len(view) and not self.validated:
            need = (8 if self.header_size is None else 8 + self.header_size) - len(self.header)
            count = min(need, len(view))
            self.header.extend(view[:count]); view = view[count:]
            if len(self.header) == 8 and self.header_size is None:
                self.header_size = int.from_bytes(self.header, 'little')
                if not 0 < self.header_size <= 1024 * 1024:
                    raise ValueError('invalid MLX tensor header length')
            if self.header_size is not None and len(self.header) == 8 + self.header_size:
                actual = json.loads(self.header[8:]); actual.pop('__metadata__', None)
                if actual != self.expected:
                    raise ValueError('MLX tensor header does not match final manifest')
                self.validated = True
        if len(view):
            expected_size = next(iter(self.expected.values()))['data_offsets'][1]
            if self.payload_bytes + len(view) > expected_size:
                raise ValueError('oversized MLX payload')
            written = self.destination.write(view)
            if written != len(view):
                raise OSError('short payload write')
            self.payload_bytes += written
        self.position += size
        return size

    def finish(self):
        expected_size = next(iter(self.expected.values()))['data_offsets'][1]
        if not self.validated or self.payload_bytes != expected_size:
            raise ValueError('incomplete MLX payload')


def save_single_pass(path, arrays, *, metadata):
    # Ask MLX for each dtype's canonical safetensors spelling; never reinterpret
    # BF16 or maintain a second dtype encoding table.
    dtypes = {}
    for value in arrays.values():
        key = str(value.dtype)
        if key not in dtypes:
            with io.BytesIO() as probe:
                mx.save_safetensors(probe, {'dtype': mx.zeros((1,), dtype=value.dtype)})
                probe.seek(0)
                n = int.from_bytes(probe.read(8), 'little')
                dtypes[key] = json.loads(probe.read(n))['dtype']['dtype']
    entries = {}; offset = 0
    for name, value in arrays.items():
        entries[name] = dict(dtype=dtypes[str(value.dtype)], shape=list(value.shape),
                             data_offsets=[offset, offset + value.nbytes])
        offset += value.nbytes
    header = dict(entries); header['__metadata__'] = metadata
    encoded = json.dumps(header, separators=(',', ':')).encode()
    size = (len(encoded) + 7) & ~7
    try:
        with open(path, 'wb') as destination:
            destination.write(size.to_bytes(8, 'little'))
            destination.write(encoded)
            destination.write(b' ' * (size - len(encoded)))
            for name, value in arrays.items():
                descriptor = dict(entries[name], data_offsets=[0, value.nbytes])
                sink = PayloadSink(destination, {name: descriptor})
                with stage('single_pass_tensor'):
                    mx.save_safetensors(sink, {name: value})
                    sink.finish()
                mx.clear_cache()
            destination.flush()
    except BaseException:
        path.unlink(missing_ok=True)
        raise
