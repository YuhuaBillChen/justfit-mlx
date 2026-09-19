import mlx.core as mx

from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5.qwen3_5 import Model, encode_vision_microbatches

PATCH_EMBED_KEY = "model.visual.patch_embed.proj.weight"
SANITIZED_KEY = "vision_tower.patch_embed.proj.weight"


def _tiny_model(in_channels=3, temporal_patch_size=2, patch_size=4, hidden_size=8):
    text_config = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        full_attention_interval=2,
        head_dim=16,
    )
    vision_config = VisionConfig(
        model_type="qwen3_5",
        depth=1,
        hidden_size=hidden_size,
        intermediate_size=16,
        out_hidden_size=32,
        num_heads=1,
        in_channels=in_channels,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=1,
        num_position_embeddings=4,
    )
    config = ModelConfig(
        text_config=text_config, vision_config=vision_config, model_type="qwen3_5"
    )
    return Model(config), vision_config


def test_patch_embed_is_transposed_from_ncdhw_to_ndhwc():
    """Qwen3.8 stores the Conv3d patch embed as NCDHW; MLX expects NDHWC."""
    model, vision_config = _tiny_model()
    expected = model.vision_tower.patch_embed.proj.weight.shape

    ncdhw = mx.zeros(
        (
            vision_config.hidden_size,
            vision_config.in_channels,
            vision_config.temporal_patch_size,
            vision_config.patch_size,
            vision_config.patch_size,
        ),
        dtype=mx.bfloat16,
    )
    sanitized = model.sanitize({PATCH_EMBED_KEY: ncdhw})

    assert sanitized[SANITIZED_KEY].shape == expected


def test_patch_embed_transpose_is_idempotent():
    """Weights already in NDHWC order must pass through untouched."""
    model, _ = _tiny_model()
    ndhwc = mx.zeros(
        model.vision_tower.patch_embed.proj.weight.shape, dtype=mx.bfloat16
    )

    once = model.sanitize({PATCH_EMBED_KEY: ndhwc})[SANITIZED_KEY]
    twice = model.sanitize({SANITIZED_KEY: once})[SANITIZED_KEY]

    assert once.shape == ndhwc.shape
    assert twice.shape == ndhwc.shape


def test_sanitized_patch_embed_loads_strictly():
    """The whole point: a strict load must accept the sanitized weight."""
    model, vision_config = _tiny_model()
    ncdhw = mx.zeros(
        (
            vision_config.hidden_size,
            vision_config.in_channels,
            vision_config.temporal_patch_size,
            vision_config.patch_size,
            vision_config.patch_size,
        ),
        dtype=mx.bfloat16,
    )
    sanitized = model.sanitize({PATCH_EMBED_KEY: ncdhw})[SANITIZED_KEY]

    bias = model.vision_tower.patch_embed.proj.bias
    model.vision_tower.patch_embed.load_weights(
        [("proj.weight", sanitized), ("proj.bias", bias)], strict=True
    )
    assert model.vision_tower.patch_embed.proj.weight.shape == sanitized.shape


def test_vision_microbatches_preserve_image_order_and_features():
    class RecordingTower:
        def __init__(self):
            self.calls = []

        def __call__(self, pixels, grid):
            self.calls.append((pixels.shape[0], grid.shape[0]))
            return pixels * 2, None

    tower = RecordingTower()
    # Three images with 2, 3, and 1 patch respectively.
    grid = mx.array([[1, 1, 2], [1, 1, 3], [1, 1, 1]])
    pixels = mx.arange(12).reshape(6, 2)

    result = encode_vision_microbatches(tower, pixels, grid, image_batch_size=1)
    mx.eval(result)

    assert tower.calls == [(2, 1), (3, 1), (1, 1)]
    assert result.tolist() == (pixels * 2).tolist()


def test_vision_microbatch_rejects_inconsistent_grid():
    class Tower:
        def __call__(self, pixels, grid):
            return pixels, None

    grid = mx.array([[1, 1, 2], [1, 1, 3]])
    pixels = mx.zeros((4, 2))

    try:
        encode_vision_microbatches(Tower(), pixels, grid, image_batch_size=1)
    except ValueError as exc:
        assert "grid and pixel patches do not match" in str(exc)
    else:
        raise AssertionError("inconsistent vision grid must fail closed")


def test_vision_microbatch_matches_full_tiny_qwen_vision_tower():
    model, config = _tiny_model()
    grid = mx.array([[1, 2, 2], [1, 2, 2]])
    pixel_width = (
        config.in_channels
        * config.temporal_patch_size
        * config.patch_size
        * config.patch_size
    )
    pixels = mx.random.normal((8, pixel_width))

    full, _ = model.vision_tower(pixels, grid)
    microbatched = encode_vision_microbatches(
        model.vision_tower, pixels, grid, image_batch_size=1
    )
    mx.eval(full, microbatched)

    assert full.shape == microbatched.shape
    assert mx.array_equal(full, microbatched).item()
