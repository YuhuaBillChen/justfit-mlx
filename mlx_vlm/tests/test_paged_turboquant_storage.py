"""Tests for fixed-capacity page-backed TurboQuant MSE tensor storage."""

import mlx.core as mx
import pytest

from mlx_vlm.paged_turboquant import PageAllocator, PagedSequence
from mlx_vlm.paged_turboquant_storage import PagedTurboQuantMSEStorage
from mlx_vlm.turboquant import TurboQuantMSEState


def _state(values, *, heads=2, width=3, norm_dtype=mx.float16):
    tokens = len(values)
    norms = mx.array(values, dtype=norm_dtype).reshape(1, 1, tokens)
    norms = mx.broadcast_to(norms, (1, heads, tokens))
    indices = mx.array(values, dtype=mx.uint32).reshape(1, 1, tokens, 1)
    indices = mx.broadcast_to(indices, (1, heads, tokens, width))
    return TurboQuantMSEState(norms, indices)


def _assert_state_values(state, values, *, heads=2, width=3):
    expected = _state(values, heads=heads, width=width)
    assert mx.array_equal(state.norms, expected.norms).item()
    assert mx.array_equal(state.indices, expected.indices).item()


def test_pool_has_fixed_physical_page_shapes_and_configurable_page_size():
    allocator = PageAllocator(capacity_pages=7, page_size=4)
    storage = PagedTurboQuantMSEStorage(
        allocator,
        kv_heads=2,
        key_packed_width=3,
        value_packed_width=5,
    )

    assert storage.keys.norms.shape == (7, 2, 4)
    assert storage.keys.indices.shape == (7, 2, 4, 3)
    assert storage.values.norms.shape == (7, 2, 4)
    assert storage.values.indices.shape == (7, 2, 4, 5)
    assert storage.page_size == 4
    assert storage.capacity_pages == 7

    default_storage = PagedTurboQuantMSEStorage.create(
        2, kv_heads=1, key_packed_width=1
    )
    assert default_storage.page_size == 256


def test_direct_appends_cross_arbitrary_physical_page_boundaries():
    allocator = PageAllocator(capacity_pages=6, page_size=4)
    storage = PagedTurboQuantMSEStorage(allocator, kv_heads=2, key_packed_width=3)
    blockers = allocator.allocate(2)
    sequence = PagedSequence(allocator)

    first = sequence.append(3)
    storage.write_append(sequence, first, _state([1, 2, 3]), _state([11, 12, 13]))
    allocator.release(blockers[:1])
    second = sequence.append(7)
    storage.write_append(
        sequence,
        second,
        _state([4, 5, 6, 7, 8, 9, 10]),
        _state([14, 15, 16, 17, 18, 19, 20]),
    )

    # Logical pages deliberately map to physical pages 2, 0, and 3.
    assert sequence.page_ids == (2, 0, 3)
    keys, values = storage.materialize(sequence)
    _assert_state_values(keys, list(range(1, 11)))
    _assert_state_values(values, list(range(11, 21)))
    assert storage.keys.norms.shape == (6, 2, 4)


def test_all_payload_validation_happens_before_either_pool_is_modified():
    allocator = PageAllocator(capacity_pages=2, page_size=4)
    storage = PagedTurboQuantMSEStorage(allocator, kv_heads=2, key_packed_width=3)
    sequence = PagedSequence(allocator)
    append = sequence.append(2)
    before_keys = mx.array(storage.keys.norms)
    before_values = mx.array(storage.values.norms)
    mx.eval(before_keys, before_values)

    with pytest.raises(ValueError, match="values token count"):
        storage.write_append(
            sequence,
            append,
            _state([1, 2]),
            _state([11]),
        )

    assert mx.array_equal(storage.keys.norms, before_keys).item()
    assert mx.array_equal(storage.values.norms, before_values).item()


def test_recycled_poisoned_page_is_overwritten_and_stale_tail_is_inaccessible():
    allocator = PageAllocator(capacity_pages=1, page_size=4)
    storage = PagedTurboQuantMSEStorage(allocator, kv_heads=2, key_packed_width=3)
    old = PagedSequence(allocator)
    old_append = old.append(4)
    storage.write_append(
        old,
        old_append,
        _state([101, 102, 103, 104]),
        _state([201, 202, 203, 204]),
    )
    old.release()

    # The page is intentionally not cleared on release. A shorter new owner
    # must overwrite every live slot and must not expose the stale tail.
    new = PagedSequence(allocator)
    new_append = new.append(2)
    storage.write_append(new, new_append, _state([1, 2]), _state([11, 12]))
    keys, values = storage.materialize(new)

    assert new.page_ids == (0,)
    _assert_state_values(keys, [1, 2])
    _assert_state_values(values, [11, 12])
    assert storage.keys.norms[0, 0].tolist() == [1.0, 2.0, 103.0, 104.0]
    assert storage.values.norms[0, 0].tolist() == [11.0, 12.0, 203.0, 204.0]


def test_full_recycled_page_overwrites_poison_in_every_payload_tensor():
    allocator = PageAllocator(capacity_pages=1, page_size=4)
    storage = PagedTurboQuantMSEStorage(allocator, kv_heads=2, key_packed_width=3)
    storage.keys.norms[:] = 60000
    storage.values.norms[:] = 60000
    poison = mx.array(0xDEADBEEF, dtype=mx.uint32)
    storage.keys.indices[:] = poison
    storage.values.indices[:] = poison
    mx.eval(storage.keys.norms, storage.values.norms)

    sequence = PagedSequence(allocator)
    append = sequence.append(4)
    storage.write_append(
        sequence,
        append,
        _state([1, 2, 3, 4]),
        _state([11, 12, 13, 14]),
    )
    keys, values = storage.materialize(sequence)

    _assert_state_values(keys, [1, 2, 3, 4])
    _assert_state_values(values, [11, 12, 13, 14])


def test_empty_materialization_and_foreign_or_shared_writes_are_safe():
    allocator = PageAllocator(capacity_pages=2, page_size=4)
    storage = PagedTurboQuantMSEStorage(allocator, kv_heads=2, key_packed_width=3)
    empty = PagedSequence(allocator)
    keys, values = storage.materialize(empty)
    assert keys.norms.shape == values.norms.shape == (1, 2, 0)
    assert keys.indices.shape == values.indices.shape == (1, 2, 0, 3)

    foreign = PagedSequence(PageAllocator(capacity_pages=2, page_size=4))
    foreign_append = foreign.append(1)
    with pytest.raises(ValueError, match="same allocator"):
        storage.write_append(foreign, foreign_append, _state([1]), _state([2]))

    owner = PagedSequence(allocator)
    owner_append = owner.append(2)
    shared = PagedSequence(
        allocator, page_ids=owner.page_ids, length=owner.length, retain_pages=True
    )
    with pytest.raises(RuntimeError, match="copy-on-write"):
        storage.write_append(owner, owner_append, _state([1, 2]), _state([3, 4]))
    shared.release()
