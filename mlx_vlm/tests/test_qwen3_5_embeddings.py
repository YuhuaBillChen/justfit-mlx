import mlx.core as mx

from mlx_vlm.models.qwen3_5.qwen3_5 import ChunkedInputEmbeddingProvider


def test_chunked_provider_matches_full_visual_embedding_merge():
    image_token = 99
    video_token = 100

    def embed_tokens(input_ids):
        return mx.repeat(input_ids[..., None].astype(mx.float32), 3, axis=-1)

    input_ids = mx.array([[1, image_token, 2, video_token, 3]])
    image_features = mx.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    provider = ChunkedInputEmbeddingProvider(
        embed_tokens,
        image_features,
        (1, 3),
        image_token,
        video_token,
    )

    chunked = mx.concatenate(
        [
            provider(input_ids[:, :2], start=0),
            provider(input_ids[:, 2:4], start=2),
            provider(input_ids[:, 4:], start=4),
        ],
        axis=1,
    )

    expected = mx.array(
        [
            [
                [1.0, 1.0, 1.0],
                [10.0, 11.0, 12.0],
                [2.0, 2.0, 2.0],
                [20.0, 21.0, 22.0],
                [3.0, 3.0, 3.0],
            ]
        ]
    )
    assert mx.array_equal(chunked, expected).item()


def test_chunked_provider_suffix_drops_consumed_visual_features():
    image_token = 99
    video_token = 100

    def embed_tokens(input_ids):
        return mx.repeat(input_ids[..., None].astype(mx.float32), 3, axis=-1)

    image_features = mx.array([[10.0, 11.0, 12.0]])
    provider = ChunkedInputEmbeddingProvider(
        embed_tokens,
        image_features,
        (1,),
        image_token,
        video_token,
    )

    suffix = provider.slice_from(4)

    assert suffix.image_features is None
    assert suffix.visual_positions == ()
    assert mx.array_equal(
        suffix(mx.array([[5, 6]]), start=0),
        embed_tokens(mx.array([[5, 6]])),
    ).item()


def test_text_suffix_embedding_spill_round_trips_bfloat16_bits(tmp_path):
    def embed_tokens(input_ids):
        values = mx.repeat(input_ids[..., None].astype(mx.float32), 5, axis=-1)
        return (values / 7).astype(mx.bfloat16)

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    provider = ChunkedInputEmbeddingProvider(embed_tokens, None, (), 99, 100)
    spilled = provider.spill_to_disk(input_ids, str(tmp_path), chunk_size=2)

    expected = embed_tokens(input_ids)
    actual = mx.concatenate(
        [
            spilled(input_ids[:, :3], start=0),
            spilled(input_ids[:, 3:], start=3),
        ],
        axis=1,
    )
    assert mx.array_equal(actual, expected).item()
    path = spilled.path
    spilled.cleanup()
    assert not __import__("os").path.exists(path)
