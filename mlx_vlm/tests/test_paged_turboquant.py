import random

import pytest

from mlx_vlm.paged_turboquant import (
    CrossLayerPagedSequence,
    PageAllocationError,
    PageAllocator,
    PagedBatchRows,
    PagedPoolSet,
    PagedSequence,
    PagePoolSpec,
)


def test_allocator_exhaustion_is_atomic_and_reuses_released_pages():
    allocator = PageAllocator(capacity_pages=3, page_size=256)

    first = allocator.allocate(2)
    before = allocator.stats()
    with pytest.raises(PageAllocationError):
        allocator.allocate(2)

    assert allocator.stats() == before
    allocator.release(first[:1])
    assert allocator.allocate(1) == first[:1]
    assert allocator.stats().high_water_pages == 2


def test_sequence_append_truncate_and_release_across_page_boundaries():
    allocator = PageAllocator(capacity_pages=8, page_size=256)
    sequence = PagedSequence(allocator)

    append = sequence.append(257)
    assert append.start == 0
    assert append.stop == 257
    assert len(append.new_page_ids) == 2
    assert sequence.length == 257
    assert sequence.capacity_tokens == 512

    sequence.truncate(256)
    assert sequence.length == 256
    assert len(sequence.page_ids) == 1
    assert allocator.stats().used_pages == 1

    sequence.release()
    assert sequence.released
    assert allocator.stats().used_pages == 0
    with pytest.raises(RuntimeError, match="released"):
        sequence.append(1)


def test_sequence_physical_reservation_is_invisible_until_commit():
    allocator = PageAllocator(capacity_pages=4, page_size=256)
    sequence = PagedSequence(allocator)

    append = sequence.reserve_append(257)

    assert sequence.length == 0
    assert sequence.page_ids == ()
    assert allocator.stats().used_pages == 2

    sequence.commit_append(append)
    assert sequence.length == 257
    assert sequence.page_ids == append.new_page_ids


def test_sequence_cancel_releases_uncommitted_physical_reservation():
    allocator = PageAllocator(capacity_pages=2, page_size=256)
    sequence = PagedSequence(allocator)
    append = sequence.reserve_append(1)

    sequence.cancel_append(append)

    assert sequence.length == 0
    assert sequence.page_ids == ()
    assert allocator.stats().used_pages == 0


def test_failed_payload_write_can_roll_back_metadata_allocation():
    allocator = PageAllocator(capacity_pages=4, page_size=4)
    sequence = PagedSequence(allocator)
    sequence.append(3)

    append = sequence.append(3)
    assert sequence.length == 6
    assert len(sequence.page_ids) == 2

    sequence.rollback(append)
    assert sequence.length == 3
    assert len(sequence.page_ids) == 1
    assert allocator.stats().used_pages == 1


def test_append_rollback_handle_cannot_mutate_another_sequence():
    allocator = PageAllocator(capacity_pages=4, page_size=4)
    first = PagedSequence(allocator)
    second = PagedSequence(allocator)
    append = first.append(3)
    second.append(3)

    with pytest.raises(ValueError, match="another sequence"):
        second.rollback(append)

    assert first.length == second.length == 3
    assert allocator.stats().used_pages == 2


def test_batch_extend_is_metadata_only_and_requires_same_allocator():
    allocator = PageAllocator(capacity_pages=8, page_size=4)
    left_row = PagedSequence(allocator)
    right_row = PagedSequence(allocator)
    left_row.append(5)
    right_row.append(2)
    left = PagedBatchRows([left_row])
    right = PagedBatchRows([right_row])
    page_ids_before = (left_row.page_ids, right_row.page_ids)

    left.extend(right)

    assert [row.length for row in left.rows] == [5, 2]
    assert (left.rows[0].page_ids, left.rows[1].page_ids) == page_ids_before
    assert right.rows == ()

    foreign = PagedBatchRows(
        [PagedSequence(PageAllocator(capacity_pages=2, page_size=4))]
    )
    with pytest.raises(ValueError, match="same page allocator"):
        left.extend(foreign)


def test_filter_reorders_rows_and_releases_removed_pages():
    allocator = PageAllocator(capacity_pages=12, page_size=4)
    rows = [PagedSequence(allocator) for _ in range(3)]
    for row, length in zip(rows, [5, 9, 2]):
        row.append(length)
    batch = PagedBatchRows(rows)

    batch.filter([2, 0])

    assert [row.length for row in batch.rows] == [2, 5]
    assert rows[1].released
    assert allocator.stats().used_pages == 3
    with pytest.raises(ValueError, match="unique"):
        batch.filter([0, 0])


def test_compact_schedule_contains_only_live_logical_pages():
    allocator = PageAllocator(capacity_pages=12, page_size=4)
    rows = [PagedSequence(allocator) for _ in range(3)]
    for row, length in zip(rows, [9, 0, 5]):
        row.append(length)
    batch = PagedBatchRows(rows)

    schedule = batch.compact_schedule()

    assert schedule.physical_page_ids == rows[0].page_ids + rows[2].page_ids
    assert schedule.page_owner == (0, 0, 0, 2, 2)
    assert schedule.row_page_offsets == (0, 3, 3, 5)
    assert batch.block_tables() == (
        (*rows[0].page_ids,),
        (-1, -1, -1),
        (*rows[2].page_ids, -1),
    )
    assert batch.sequence_lengths == (9, 0, 5)


def test_randomized_lifecycle_returns_every_page_without_duplicates():
    rng = random.Random(20260903)
    allocator = PageAllocator(capacity_pages=64, page_size=8)
    sequences = [PagedSequence(allocator) for _ in range(8)]

    for _ in range(10_000):
        live = [sequence for sequence in sequences if not sequence.released]
        if not live:
            break
        sequence = rng.choice(live)
        operation = rng.choice(("append", "truncate", "release"))

        if operation == "append":
            count = rng.randrange(0, 17)
            try:
                sequence.append(count)
            except PageAllocationError:
                pass
        elif operation == "truncate":
            sequence.truncate(rng.randrange(sequence.length + 1))
        else:
            sequence.release()

        all_pages = [
            page_id
            for candidate in sequences
            if not candidate.released
            for page_id in candidate.page_ids
        ]
        assert len(all_pages) == len(set(all_pages))
        assert allocator.stats().used_pages == len(all_pages)

    for sequence in sequences:
        sequence.release()
    assert allocator.stats().used_pages == 0
    assert allocator.stats().free_pages == allocator.capacity_pages


def test_pool_set_reuses_layer_allocators_across_independent_prompt_sequences():
    pools = PagedPoolSet.uniform(("layer.0", "layer.1"), capacity_pages=8, page_size=4)

    first = pools.new_sequence()
    second = pools.new_sequence()
    first.append(5)
    second.append(2)

    for layer in pools.layer_keys:
        first_row = first.sequence_for(layer)
        second_row = second.sequence_for(layer)
        assert first_row.allocator is pools.allocator_for(layer)
        assert second_row.allocator is pools.allocator_for(layer)

        active = PagedBatchRows([first_row])
        pending = PagedBatchRows([second_row])
        active.extend(pending)
        assert [row.length for row in active.rows] == [5, 2]
        assert pending.rows == ()


def test_cross_layer_append_rolls_back_every_successful_layer_on_exhaustion():
    pools = PagedPoolSet.uniform((0, 1), capacity_pages=2, page_size=4)
    blocker = pools.allocator_for(1).allocate(1)
    sequence = pools.new_sequence()

    before = {layer: pools.allocator_for(layer).stats() for layer in pools.layer_keys}
    with pytest.raises(PageAllocationError):
        sequence.append(8)

    assert sequence.length == 0
    assert all(
        sequence.sequence_for(layer).page_ids == () for layer in pools.layer_keys
    )
    for layer in pools.layer_keys:
        after = pools.allocator_for(layer).stats()
        assert after.used_pages == before[layer].used_pages
        assert after.free_pages == before[layer].free_pages

    pools.allocator_for(1).release(blocker)


def test_cross_layer_append_handle_supports_atomic_payload_failure_rollback():
    pools = PagedPoolSet.uniform(("k0", "k1"), capacity_pages=4, page_size=4)
    sequence = pools.new_sequence()
    sequence.append(3)

    append = sequence.append(3)
    assert append.start == 3
    assert append.stop == 6
    assert sequence.length == 6
    assert all(
        len(sequence.sequence_for(layer).page_ids) == 2 for layer in pools.layer_keys
    )

    sequence.rollback(append)

    assert sequence.length == 3
    assert all(
        len(sequence.sequence_for(layer).page_ids) == 1 for layer in pools.layer_keys
    )
    assert all(
        pools.allocator_for(layer).stats().used_pages == 1 for layer in pools.layer_keys
    )


def test_cross_layer_truncate_and_release_keep_layer_lengths_in_lockstep():
    pools = PagedPoolSet(
        {
            "full.0": PagePoolSpec(capacity_pages=6, page_size=4),
            "full.1": PagePoolSpec(capacity_pages=6, page_size=4),
        }
    )
    sequence = pools.new_sequence(length=9)

    assert isinstance(sequence, CrossLayerPagedSequence)
    assert sequence.length == 9
    assert all(sequence.sequence_for(layer).length == 9 for layer in pools.layer_keys)

    sequence.truncate(4)
    assert sequence.length == 4
    assert all(sequence.sequence_for(layer).length == 4 for layer in pools.layer_keys)

    sequence.release()
    assert sequence.released
    assert all(
        pools.allocator_for(layer).stats().used_pages == 0 for layer in pools.layer_keys
    )
    sequence.release()


def test_pool_set_validates_specs_and_layer_lookup():
    with pytest.raises(ValueError, match="at least one layer"):
        PagedPoolSet({})
    with pytest.raises(ValueError, match="same page_size"):
        PagedPoolSet(
            {
                0: PagePoolSpec(capacity_pages=2, page_size=4),
                1: PagePoolSpec(capacity_pages=2, page_size=8),
            }
        )

    pools = PagedPoolSet.uniform((0, 1), capacity_pages=2, page_size=4)
    with pytest.raises(KeyError, match="unknown paged layer"):
        pools.allocator_for(2)
