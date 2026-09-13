import gc
import logging
import os
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable, Generator, List, Optional, Tuple

import mlx.core as mx
from fastapi import HTTPException

from .. import apc as _apc
from .._stream_cleanup import clear_mlx_streams
from ..generate import (  # noqa: F401 - compatibility re-exported by server.__init__
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_TOP_P,
    BatchGenerator,
    _make_cache,
)
from ..generate.common import (
    DEFAULT_COMPLETION_BATCH_SIZE,
    DEFAULT_PREFILL_BATCH_SIZE,
)
from ..generate.diffusion import (
    is_diffusion_model,
    stream_diffusion_generate_from_kwargs,
)
from ..models import cache as model_cache
from ..paged_turboquant_kernel import PAGED_TURBOQUANT_PAGE_SIZE
from ..paged_turboquant_pool import (
    PagedTurboQuantLayerSpec,
    PagedTurboQuantPoolRegistry,
)
from ..sample_utils import (
    apply_top_k,
    make_logits_processors,
    make_sampler,
    top_p_sampling,
)
from ..speculative.utils import speculative_stats_since, speculative_stats_snapshot
from ..structured import ThinkingAwareLogitsProcessor
from ..tokenizer_utils import _ServerTokenStreamer, make_streaming_detokenizer
from ..utils import ThinkingBudgetCriteria, load, prepare_inputs, resolve_eos_token_ids
from .draft_lifecycle import LazyDrafter
from .language_lifecycle import (
    ComponentResidencyManager,
    LanguageEmbeddingPhaseSwap,
    LanguageHeadPhaseSwap,
)
from .runtime import runtime
from .vision_lifecycle import VisionTowerPhaseSwap

logger = logging.getLogger("mlx_vlm.server")

DEFAULT_SPECULATIVE_BATCH_COALESCE_MS = 5.0
DEFAULT_LOG_PROGRESS_INTERVAL = 10
DEFAULT_ENABLE_THINKING = False
METRICS_HISTORY_LIMIT = 100
METRICS_RECENT_LIMIT = 32

# Internal queue marker: a successful GPU step means the producer is alive even
# when a queued/chunk-prefilling request cannot emit a token yet.  Consumers
# swallow this marker and restart their inactivity timeout.
_TOKEN_QUEUE_ACTIVITY = object()


class PromptTooLongError(ValueError):
    """Raised when a request exceeds the configured server context budget."""


def _get_draft_block_size_from_env():
    draft_block_size_str = os.environ.get("MLX_VLM_DRAFT_BLOCK_SIZE")
    return int(draft_block_size_str) if draft_block_size_str else None


def _notify_queues(queues, *items):
    for queue in queues:
        for item in items:
            try:
                queue.put(item)
            except Exception:
                break


def get_prefill_step_size():
    return int(os.environ.get("PREFILL_STEP_SIZE", DEFAULT_PREFILL_STEP_SIZE))


def get_max_num_seqs():
    """Max sequences allowed in the running batch at once (None = unbounded)."""
    raw = os.environ.get("MLX_VLM_MAX_NUM_SEQS", "")
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def get_lm_head_mixed_prefill_max_tokens() -> int:
    """Largest prompt allowed to join a phase-swapped active cohort."""

    raw = os.environ.get("MLX_VLM_LM_HEAD_MIXED_PREFILL_MAX_TOKENS", "0")
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_LM_HEAD_MIXED_PREFILL_MAX_TOKENS=%r; using 0.",
            raw,
        )
        return 0
    return max(0, value)


def get_batch_kv_slot_budget():
    """Maximum rectangular KV token slots across an active batch.

    A value of ``N`` permits a batch only while
    ``batch_size * max(per_request_context_budget) <= N``.  The first row is
    always admitted so this guard reduces concurrency instead of reducing the
    server's single-request context limit.
    """
    raw = os.environ.get("MLX_VLM_BATCH_KV_SLOT_BUDGET", "")
    if not raw:
        return None
    try:
        budget = int(raw)
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_BATCH_KV_SLOT_BUDGET=%r; disabling KV-budget admission.",
            raw,
        )
        return None
    return budget if budget > 0 else None


def paged_turboquant_enabled() -> bool:
    return os.environ.get("MLX_VLM_PAGED_TQ", "0").lower() in ("1", "true", "yes")


def paged_scheduler_enabled() -> bool:
    return paged_turboquant_enabled() and os.environ.get(
        "MLX_VLM_PAGED_SCHEDULER", "0"
    ).lower() in ("1", "true", "yes")


def get_paged_scheduler_scan_limit() -> int:
    raw = os.environ.get("MLX_VLM_PAGED_SCHEDULER_SCAN_LIMIT", "32")
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_PAGED_SCHEDULER_SCAN_LIMIT=%r; using 32.", raw
        )
        return 32
    return max(1, value)


def get_paged_scheduler_max_bypass() -> int:
    raw = os.environ.get("MLX_VLM_PAGED_SCHEDULER_MAX_BYPASS", "8")
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_PAGED_SCHEDULER_MAX_BYPASS=%r; using 8.", raw
        )
        return 8
    return max(0, value)


def get_paged_kv_capacity_tokens() -> Optional[int]:
    raw = os.environ.get("MLX_VLM_PAGED_KV_CAPACITY_TOKENS", "")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "MLX_VLM_PAGED_KV_CAPACITY_TOKENS must be a positive integer"
        ) from exc
    if value <= 0:
        raise ValueError("MLX_VLM_PAGED_KV_CAPACITY_TOKENS must be positive")
    return value


def _get_nonnegative_env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d.", name, raw, default)
        return default
    if value < 0:
        logger.warning("Invalid %s=%r; using %d.", name, raw, default)
        return default
    return value


def get_paged_output_guarantee_tokens() -> Optional[int]:
    """Output tokens reserved at admission for each paged request.

    When unset, admission keeps the legacy behavior and reserves the request's
    complete output ceiling.  A configured value permits output beyond the
    guarantee to consume shared capacity elastically as it is generated.
    """

    raw = os.environ.get("MLX_VLM_PAGED_OUTPUT_GUARANTEE_TOKENS", "")
    if not raw:
        return None
    return _get_nonnegative_env_int("MLX_VLM_PAGED_OUTPUT_GUARANTEE_TOKENS")


def get_paged_kv_safety_tokens() -> int:
    """Global token headroom withheld from paged admission promises."""

    return _get_nonnegative_env_int("MLX_VLM_PAGED_KV_SAFETY_TOKENS")


def make_paged_turboquant_registry(language_model):
    """Build the fixed generator-lifetime page pools for supported KV leaves."""

    capacity_tokens = get_paged_kv_capacity_tokens()
    if capacity_tokens is None:
        raise ValueError(
            "MLX_VLM_PAGED_KV_CAPACITY_TOKENS is required with MLX_VLM_PAGED_TQ=1"
        )
    max_num_seqs = get_max_num_seqs()
    if max_num_seqs is None:
        raise ValueError("MLX_VLM_PAGED_TQ=1 requires MLX_VLM_MAX_NUM_SEQS")
    if not hasattr(language_model, "make_cache"):
        raise ValueError("paged TurboQuant requires a model-defined cache layout")
    model_args = getattr(language_model, "args", None)
    # Qwen3.5 exposes a legacy ``head_dim`` property derived as
    # hidden_size/num_heads, which is not the configured Q/K/V head width for
    # models such as 5120/24 with explicit head_dim=256. Prefer model config.
    head_dim = int(
        getattr(model_args, "head_dim", 0)
        or getattr(language_model, "head_dim", 0)
    )
    kv_heads = int(
        getattr(
            language_model,
            "n_kv_heads",
            getattr(model_args, "num_key_value_heads", 0),
        )
    )
    layout = language_model.make_cache()
    specs = {
        index: PagedTurboQuantLayerSpec(
            capacity_pages=(
                capacity_tokens + PAGED_TURBOQUANT_PAGE_SIZE - 1
            )
            // PAGED_TURBOQUANT_PAGE_SIZE,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        for index, entry in enumerate(layout)
        if model_cache.should_quantize_kv_layer(index, len(layout))
        and isinstance(
            entry,
            (
                model_cache.KVCache,
                model_cache.ChunkedKVCache,
                model_cache.SimpleKVCache,
            ),
        )
    }
    if not specs:
        raise ValueError("model has no supported Q4 paged KV leaves")
    return PagedTurboQuantPoolRegistry(specs)


def get_server_max_tokens():
    return int(os.environ.get("MLX_VLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))


def get_token_queue_timeout():
    return runtime.config.token_queue_timeout


def get_speculative_batch_coalesce_s():
    raw = os.environ.get(
        "MLX_VLM_SPEC_BATCH_COALESCE_MS", str(DEFAULT_SPECULATIVE_BATCH_COALESCE_MS)
    )
    try:
        return max(0.0, float(raw)) / 1000.0
    except ValueError:
        return DEFAULT_SPECULATIVE_BATCH_COALESCE_MS / 1000.0


def speculative_singleton_only() -> bool:
    return os.environ.get("MLX_VLM_SPECULATIVE_SINGLETON_ONLY", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def mtp_repromotion_enabled() -> bool:
    return os.environ.get("MLX_VLM_MTP_REPROMOTE", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def get_log_progress_interval():
    """Number of decoded tokens between INFO progress messages (0 disables)."""
    raw = os.environ.get(
        "MLX_VLM_LOG_PROGRESS_INTERVAL", str(DEFAULT_LOG_PROGRESS_INTERVAL)
    )
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_LOG_PROGRESS_INTERVAL=%r; falling back to %d.",
            raw,
            DEFAULT_LOG_PROGRESS_INTERVAL,
        )
        return DEFAULT_LOG_PROGRESS_INTERVAL


def _position_seed(seed: int, row_id: int, position: int) -> int:
    x = (int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF
    x = (x + (int(row_id) + 1) * 0x85EBCA6B) & 0xFFFFFFFF
    x = (x ^ ((int(position) + 1) * 0xC2B2AE35)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    return int(x & 0xFFFFFFFF)


def _position_keys(seed: int, row_ids: List[int], positions: List[int]) -> mx.array:
    return mx.stack(
        [
            mx.random.key(_position_seed(seed, row, pos))
            for row, pos in zip(row_ids, positions)
        ]
    )


class _PositionedTargetSampler:
    """Server sampler with stateless target draws for ragged verification."""

    def __init__(
        self,
        *,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        seed: Optional[int],
    ):
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = DEFAULT_SEED if seed is None else int(seed)

    def _apply_top_k(self, logprobs: mx.array) -> mx.array:
        return apply_top_k(logprobs, self.top_k) if self.top_k > 0 else logprobs

    def __call__(self, logprobs: mx.array) -> mx.array:
        logprobs = self._apply_top_k(logprobs)
        if self.top_p > 0 and self.top_p < 1.0:
            return top_p_sampling(logprobs, self.top_p, self.temperature)
        return mx.random.categorical(logprobs * (1 / self.temperature))

    def sample_target(
        self,
        logprobs: mx.array,
        *,
        row_ids: List[int],
        positions: List[int],
    ) -> mx.array:
        if logprobs.shape[0] != len(row_ids) or len(row_ids) != len(positions):
            raise ValueError("row_ids and positions must match logprobs batch size.")
        logprobs = self._apply_top_k(logprobs)
        keys = _position_keys(self.seed, row_ids, positions)
        if self.top_p > 0 and self.top_p < 1.0:
            return mx.vmap(self._sample_top_p_one, in_axes=(0, 0))(logprobs, keys)
        return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)

    def sample_proposal(
        self,
        logprobs: mx.array,
        *,
        row_ids: List[int],
        positions: List[int],
    ) -> mx.array:
        keys = _position_keys(self.seed ^ 0x0DFA5202, row_ids, positions)
        return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)

    def _sample_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        return mx.random.categorical(logprobs * (1 / self.temperature), key=key)

    def _sample_top_p_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        if logprobs.dtype == mx.bfloat16:
            logprobs = logprobs.astype(mx.float32)
        probs = mx.softmax(logprobs / self.temperature, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        top_probs = mx.where(
            cumulative_probs > 1 - self.top_p,
            sorted_probs,
            mx.zeros_like(sorted_probs),
        )
        sampled_pos = mx.random.categorical(mx.log(top_probs), key=key)
        return mx.take_along_axis(sorted_indices, sampled_pos[..., None], axis=-1)[0]


def get_server_enable_thinking():
    raw = os.environ.get("MLX_VLM_ENABLE_THINKING")
    if raw is None:
        return DEFAULT_ENABLE_THINKING
    return raw.lower() in ("1", "true", "yes", "on")


def capacity_ignore_eos_enabled():
    """Return whether exact-length capacity runs should ignore EOS tokens."""
    return os.environ.get("MLX_VLM_CAPACITY_IGNORE_EOS", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def get_server_thinking_budget():
    raw = os.environ.get("MLX_VLM_THINKING_BUDGET")
    return None if raw is None else int(raw)


def get_server_thinking_start_token():
    return os.environ.get("MLX_VLM_THINKING_START_TOKEN")


def get_server_thinking_end_token():
    return os.environ.get("MLX_VLM_THINKING_END_TOKEN")


def get_expert_cache_gb():
    raw = os.environ.get("EXPERT_CACHE_GB", "")
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid EXPERT_CACHE_GB=%r; ignoring.", raw)
        return None


def get_quantized_kv_bits():
    kv_bits = float(os.environ.get("KV_BITS", 0))
    return kv_bits or None


def get_quantized_kv_split_bits():
    def _read(name):
        raw = os.environ.get(name)
        return float(raw) if raw else None

    return _read("KV_KEY_BITS"), _read("KV_VALUE_BITS")


def get_kv_split_schemes():
    return os.environ.get("KV_KEY_SCHEME"), os.environ.get("KV_VALUE_SCHEME")


def get_kv_group_size():
    return int(os.environ.get("KV_GROUP_SIZE", DEFAULT_KV_GROUP_SIZE))


def get_kv_quant_scheme():
    return os.environ.get("KV_QUANT_SCHEME", DEFAULT_KV_QUANT_SCHEME)


def get_max_kv_size(model: str):
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    if max_kv_tokens == 0:
        return None
    if get_quantized_kv_bits() is not None:
        logger.warning("Model %s uses QuantizedKVCache; MAX_KV_SIZE is ignored.", model)
        return None
    return max_kv_tokens


def get_configured_context_limit():
    if runtime.config.max_kv_size is not None:
        return runtime.config.max_kv_size or None
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    return max_kv_tokens or None


def _count_prompt_tokens(raw_inputs: dict) -> int:
    input_ids = raw_inputs["input_ids"]
    return input_ids.size if hasattr(input_ids, "size") else len(input_ids)


def _check_configured_context_budget(prompt_tokens: int, max_tokens: int):
    context_limit = get_configured_context_limit()
    requested_tokens = prompt_tokens + max(0, int(max_tokens or 0))
    if context_limit is not None and requested_tokens > context_limit:
        raise PromptTooLongError(
            "Request needs "
            f"{requested_tokens} context tokens "
            f"({prompt_tokens} prompt + {max_tokens} max generation), "
            f"but MAX_KV_SIZE is {context_limit}."
        )
    if paged_turboquant_enabled():
        pool_capacity = get_paged_kv_capacity_tokens()
        if pool_capacity is not None and requested_tokens > pool_capacity:
            raise PromptTooLongError(
                "Request needs "
                f"{requested_tokens} context tokens "
                f"({prompt_tokens} prompt + {max_tokens} max generation), "
                f"but paged KV pool capacity is {pool_capacity}."
            )


def get_quantized_kv_start():
    return int(os.environ.get("QUANTIZED_KV_START", DEFAULT_QUANTIZED_KV_START))


def get_top_logprobs_k():
    """Max per-token top_logprobs honored by the server (0 = disabled).

    Set via TOP_LOGPROBS_K env var. OpenAI caps this at 20. When 0, requests
    with top_logprobs>0 still succeed but the top_logprobs list stays empty.
    """
    k = int(os.environ.get("TOP_LOGPROBS_K", 0))
    return max(0, min(k, 20))


def _token_window_rate(token_times: List[float], window: int) -> Optional[float]:
    if len(token_times) < 2:
        return None
    subset = token_times[-window:]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


def _token_window_rate_first(token_times: List[float], window: int) -> Optional[float]:
    if len(token_times) < 2:
        return None
    subset = token_times[:window]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


class ServerMetricsStore:
    """Rolling request metrics and lifetime counters for the server."""

    def __init__(self, history_limit: int = METRICS_HISTORY_LIMIT):
        self.history_limit = history_limit
        self.reset()

    def reset(self):
        self.started_at = time.time()
        self._lock = Lock()
        self._latest: Optional[dict] = None
        self._recent = deque(maxlen=self.history_limit)
        self._requests_started = 0
        self._requests_completed = 0
        self._requests_failed = 0
        self._streaming_requests = 0
        self._in_flight = 0
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._generated_tokens_total = 0
        self._request_time_total_s = 0.0
        self._decode_time_total_s = 0.0
        self._last_request_at: Optional[float] = None
        self._last_error: Optional[dict] = None

    def begin_request(self, *, endpoint: str, model: str, stream: bool):
        with self._lock:
            self._requests_started += 1
            self._in_flight += 1
            if stream:
                self._streaming_requests += 1
            in_flight = self._in_flight
        logger.info(
            "Request started: endpoint=%s model=%s stream=%s in_flight=%d",
            endpoint,
            model,
            stream,
            in_flight,
        )

    def record_success(self, envelope: dict):
        payload = dict(envelope)
        with self._lock:
            self._requests_completed += 1
            self._in_flight = max(0, self._in_flight - 1)
            self._latest = payload
            self._recent.append(payload)
            self._last_request_at = payload.get("timestamp_unix")
            self._prompt_tokens_total += int(payload.get("prompt_tokens") or 0)
            self._completion_tokens_total += int(payload.get("completion_tokens") or 0)
            self._generated_tokens_total += int(payload.get("generated_tokens") or 0)
            self._request_time_total_s += float(payload.get("request_elapsed_s") or 0.0)
            self._decode_time_total_s += float(payload.get("decode_elapsed_s") or 0.0)
            in_flight = self._in_flight
        logger.info(
            "Request completed: endpoint=%s model=%s stream=%s backend=%s "
            "prompt_tokens=%d generated_tokens=%d elapsed=%.3fs "
            "prefill=%.1f tok/s decode=%.1f tok/s finish_reason=%s in_flight=%d",
            payload.get("endpoint"),
            payload.get("model"),
            payload.get("stream"),
            payload.get("backend"),
            int(payload.get("prompt_tokens") or 0),
            int(payload.get("generated_tokens") or 0),
            float(payload.get("request_elapsed_s") or 0.0),
            float(payload.get("prefill_tok_s") or 0.0),
            float(payload.get("decode_tok_s") or 0.0),
            payload.get("finish_reason"),
            in_flight,
        )

    def record_failure(self, *, endpoint: str, model: str, stream: bool, error: str):
        with self._lock:
            self._requests_failed += 1
            self._in_flight = max(0, self._in_flight - 1)
            self._last_request_at = time.time()
            self._last_error = {
                "timestamp_unix": self._last_request_at,
                "endpoint": endpoint,
                "model": model,
                "stream": bool(stream),
                "error": error,
            }
            in_flight = self._in_flight
        logger.warning(
            "Request failed: endpoint=%s model=%s stream=%s error=%s in_flight=%d",
            endpoint,
            model,
            stream,
            error,
            in_flight,
        )

    def snapshot(self) -> dict:
        with self._lock:
            latest = dict(self._latest) if self._latest is not None else None
            recent = [dict(item) for item in list(self._recent)[-METRICS_RECENT_LIMIT:]]
            requests_completed = self._requests_completed
            avg_request_time = (
                self._request_time_total_s / requests_completed
                if requests_completed > 0
                else 0.0
            )
            avg_request_tok_s = (
                self._completion_tokens_total / self._request_time_total_s
                if self._request_time_total_s > 0
                else 0.0
            )
            avg_decode_tok_s = (
                self._generated_tokens_total / self._decode_time_total_s
                if self._decode_time_total_s > 0
                else 0.0
            )
            last_error = (
                dict(self._last_error) if self._last_error is not None else None
            )
            last_request_at = self._last_request_at
            return {
                "latest": latest,
                "recent": recent,
                "summary": {
                    "uptime_s": max(0.0, time.time() - self.started_at),
                    "requests_started": self._requests_started,
                    "requests_completed": requests_completed,
                    "requests_failed": self._requests_failed,
                    "streaming_requests": self._streaming_requests,
                    "in_flight": self._in_flight,
                    "prompt_tokens_total": self._prompt_tokens_total,
                    "completion_tokens_total": self._completion_tokens_total,
                    "generated_tokens_total": self._generated_tokens_total,
                    "avg_request_time_s": avg_request_time,
                    "avg_request_tok_s": avg_request_tok_s,
                    "avg_decode_tok_s": avg_decode_tok_s,
                    "last_request_at": last_request_at,
                    "last_error": last_error,
                },
            }


def _prompt_eval_time_from_tps(
    prompt_tokens: int, prompt_tps: Optional[float]
) -> Optional[float]:
    if prompt_tokens <= 0 or prompt_tps is None or prompt_tps <= 0:
        return None
    return prompt_tokens / prompt_tps


def _decode_elapsed_from_metrics(
    token_times: List[float],
    generated_tokens: int,
    generation_tps: Optional[float],
) -> Optional[float]:
    if len(token_times) >= 2:
        elapsed = token_times[-1] - token_times[0]
        if elapsed > 0:
            return elapsed
    if generated_tokens > 0 and generation_tps is not None and generation_tps > 0:
        return generated_tokens / generation_tps
    return None


def _build_metrics_envelope(
    *,
    endpoint: str,
    model: str,
    stream: bool,
    backend: str,
    prompt_tokens: int,
    completion_tokens: int,
    generated_tokens: int,
    request_elapsed_s: float,
    request_started_s: float,
    token_times: Optional[List[float]] = None,
    prompt_tps: Optional[float] = None,
    generation_tps: Optional[float] = None,
    peak_memory_gb: Optional[float] = None,
    finish_reason: Optional[str] = None,
    image_count: int = 0,
    audio_count: int = 0,
    structured_output: bool = False,
    thinking_enabled: bool = False,
    tool_parser: Optional[str] = None,
    tool_calls: bool = False,
) -> dict:
    token_times = token_times or []
    ttft_s = max(0.0, token_times[0] - request_started_s) if token_times else None
    decode_elapsed_s = _decode_elapsed_from_metrics(
        token_times, generated_tokens, generation_tps
    )
    decode_tok_s = None
    if generation_tps is not None and generation_tps > 0:
        decode_tok_s = generation_tps
    elif decode_elapsed_s is not None and decode_elapsed_s > 0 and generated_tokens > 0:
        decode_tok_s = generated_tokens / decode_elapsed_s
    prompt_eval_time_s = _prompt_eval_time_from_tps(prompt_tokens, prompt_tps)
    request_tok_s = (
        completion_tokens / request_elapsed_s if request_elapsed_s > 0 else 0.0
    )
    return {
        "timestamp_unix": time.time(),
        "endpoint": endpoint,
        "model": model,
        "stream": bool(stream),
        "backend": backend,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "generated_tokens": int(generated_tokens),
        "reasoning_tokens": max(0, int(generated_tokens) - int(completion_tokens)),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "prompt_eval_time_s": prompt_eval_time_s,
        "prefill_tok_s": prompt_tps,
        "ttft_s": ttft_s,
        "decode_elapsed_s": decode_elapsed_s,
        "request_elapsed_s": request_elapsed_s,
        "request_tok_s": request_tok_s,
        "decode_tok_s": decode_tok_s,
        "sliding_decode_tok_s_first_32": _token_window_rate_first(token_times, 32),
        "sliding_decode_tok_s_first_64": _token_window_rate_first(token_times, 64),
        "sliding_decode_tok_s_last_32": _token_window_rate(token_times, 32),
        "sliding_decode_tok_s_last_64": _token_window_rate(token_times, 64),
        "peak_memory_gb": peak_memory_gb,
        "finish_reason": finish_reason,
        "image_count": int(image_count),
        "audio_count": int(audio_count),
        "structured_output": bool(structured_output),
        "thinking_enabled": bool(thinking_enabled),
        "tool_parser": tool_parser,
        "tool_calls": bool(tool_calls),
        "apc_enabled": runtime.apc_manager is not None,
    }


def load_model_resources(model_path: str, adapter_path: Optional[str]):
    """
    Loads model, processor, and config based on paths.
    Handles potential loading errors.
    """
    try:
        logger.info("Loading model: %s", model_path)
        if adapter_path:
            logger.info("Loading adapter: %s", adapter_path)
        # Use the load function from utils.py which handles path resolution and loading
        trust_remote_code = (
            os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
        )
        model, processor = load(
            model_path,
            adapter_path,
            trust_remote_code=trust_remote_code,
            expert_cache_gb=get_expert_cache_gb(),
            max_kv_size=get_configured_context_limit(),
        )
        config = model.config
        logger.info("Model and processor loaded successfully.")
        return model, processor, config
    except Exception as e:
        logger.exception("Error loading model %s: %s", model_path, e)
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")


# =============================================================================
# ResponseGenerator - Concurrent Request Handling with Threaded Batching
# =============================================================================


@dataclass
class GenerationArguments:
    """Arguments for a generation request."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = 0
    min_p: float = 0.0
    top_n_sigma: float = 0.0
    p_less: bool = False
    typical_p: float = 1.0
    seed: Optional[int] = None
    logprobs: bool = False
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE
    presence_penalty: Optional[float] = None
    presence_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE
    frequency_penalty: Optional[float] = None
    frequency_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE
    max_denoising_steps: Optional[int] = None
    block_length: Optional[int] = None
    num_to_transfer: Optional[int] = None
    max_transfer_per_step: Optional[int] = None
    editing_threshold: Optional[float] = None
    max_post_steps: Optional[int] = None
    stability_steps: Optional[int] = None
    diffusion_full_canvas: Optional[bool] = None
    diffusion_min_canvas_length: Optional[int] = None
    diffusion_max_canvas_length: Optional[int] = None
    diffusion_sampler: Optional[str] = None
    threshold: Optional[float] = None
    min_threshold: Optional[float] = None
    logit_bias: Optional[dict] = None
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    reasoning: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    thinking_budget: Optional[int] = None
    thinking_start_token: Optional[str] = None
    thinking_end_token: Optional[str] = None
    skip_special_tokens: bool = True
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None
    # Per-tenant salt for APC. When set, it's mixed into ``extra_hash`` so
    # cached blocks from one tenant can't be reused (or detected via timing)
    # by another. None = no salt = single-tenant behaviour.
    tenant_id: Optional[str] = None

    def diffusion_kwargs(self) -> dict:
        """Diffusion-only generation kwargs explicitly supplied by a request."""
        kw = {}
        for key in (
            "max_denoising_steps",
            "block_length",
            "num_to_transfer",
            "max_transfer_per_step",
            "editing_threshold",
            "max_post_steps",
            "stability_steps",
            "diffusion_full_canvas",
            "diffusion_min_canvas_length",
            "diffusion_max_canvas_length",
            "diffusion_sampler",
            "threshold",
            "min_threshold",
        ):
            value = getattr(self, key)
            if value is not None:
                kw[key] = value
        return kw

    def to_generate_kwargs(self) -> dict:
        """Convert to kwargs dict for generate()/stream_generate()."""
        kw = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "top_n_sigma": self.top_n_sigma,
            "p_less": self.p_less,
            "typical_p": self.typical_p,
            "enable_thinking": self.enable_thinking,
        }
        if self.seed is not None:
            kw["seed"] = self.seed
        if self.repetition_penalty is not None:
            kw["repetition_penalty"] = self.repetition_penalty
        if self.repetition_context_size is not None:
            kw["repetition_context_size"] = self.repetition_context_size
        if self.presence_penalty is not None:
            kw["presence_penalty"] = self.presence_penalty
        if self.presence_context_size is not None:
            kw["presence_context_size"] = self.presence_context_size
        if self.frequency_penalty is not None:
            kw["frequency_penalty"] = self.frequency_penalty
        if self.frequency_context_size is not None:
            kw["frequency_context_size"] = self.frequency_context_size
        if self.logit_bias is not None:
            kw["logit_bias"] = self.logit_bias
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        if self.thinking_end_token is not None:
            kw["thinking_end_token"] = self.thinking_end_token
        if self.logits_processors is not None:
            kw["logits_processors"] = self.logits_processors
        if self.tenant_id is not None:
            kw["apc_tenant"] = self.tenant_id
        kw.update(self.diffusion_kwargs())
        return kw

    def to_template_kwargs(self) -> dict:
        """Convert to kwargs for apply_chat_template()."""
        kw = {"enable_thinking": self.enable_thinking}
        if self.reasoning is not None:
            kw["reasoning"] = self.reasoning
        if self.reasoning_effort is not None:
            kw["reasoning_effort"] = self.reasoning_effort
            # Muse Glimmer's chat template reads the reasoning_strength alias.
            kw["reasoning_strength"] = self.reasoning_effort
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        if self.thinking_end_token is not None:
            kw["thinking_end_token"] = self.thinking_end_token
        return kw


@dataclass
class GenerationContext:
    """Context returned when a request is queued."""

    uid: int
    prompt_tokens: int


@dataclass
class QueuedGenerationRequest:
    """Preprocessed generation request waiting for the GPU worker."""

    rqueue: Queue
    raw_inputs: dict
    prompt_tokens: int
    args: GenerationArguments
    thinking_budget_criteria: Optional[ThinkingBudgetCriteria] = None
    images: Optional[List] = None
    videos: Optional[List] = None
    audio: Optional[List] = None
    apc_semantic_hash: Optional[int] = None
    request_id: Optional[str] = None
    cancel_event: Optional[Event] = None
    queued_at: float = field(default_factory=time.perf_counter)
    kv_bypass_count: int = 0
    apc_prefix_tokens_hint: Optional[int] = None
    apc_prefix_probe_done: bool = False


@dataclass
class GenerationMetrics:
    """Runtime metrics collected while consuming generation output."""

    token_times: List[float] = field(default_factory=list)
    peak_memory: float = 0.0
    cached_tokens: int = 0
    prompt_tps: Optional[float] = None
    generation_tps: Optional[float] = None
    last_chunk_at: Optional[float] = None
    last_chunk_rate: Optional[float] = None
    generated_tokens: int = 0
    first_chunk_tokens: int = 0
    draft_kind: Optional[str] = None
    draft_rounds: Optional[int] = None
    draft_n_accepted: Optional[int] = None
    draft_n: Optional[int] = None
    capacity_exhausted: bool = False
    guaranteed_output_tokens: Optional[int] = None
    elastic_output_tokens: int = 0
    capacity_wait_ms: float = 0.0

    def record_chunk(self, chunk) -> Optional[float]:
        now = getattr(chunk, "emitted_at", None) or time.perf_counter()
        token_count = getattr(chunk, "token_count", None)
        if token_count is None:
            generation_tokens = getattr(chunk, "generation_tokens", None)
            token_count = (
                max(0, int(generation_tokens) - self.generated_tokens)
                if generation_tokens is not None
                else 1
            )
        emitted_tokens = max(0, int(token_count or 0))
        self.last_chunk_rate = None
        if emitted_tokens > 0:
            if not self.token_times:
                self.first_chunk_tokens = emitted_tokens
            self.generated_tokens += emitted_tokens
            if self.last_chunk_at is not None and now > self.last_chunk_at:
                self.last_chunk_rate = emitted_tokens / (now - self.last_chunk_at)
            self.last_chunk_at = now
            self.token_times.append(now)
        self.record_result(chunk)
        return self.last_chunk_rate

    @property
    def rate(self) -> Optional[float]:
        if len(self.token_times) < 2:
            return None
        elapsed = self.token_times[-1] - self.token_times[0]
        measured_tokens = self.generated_tokens - self.first_chunk_tokens
        return measured_tokens / elapsed if elapsed > 0 else None

    def record_result(self, result) -> None:
        self.peak_memory = max(
            self.peak_memory, float(getattr(result, "peak_memory", 0.0) or 0.0)
        )
        prompt_tps = getattr(result, "prompt_tps", None)
        if prompt_tps is not None:
            self.prompt_tps = prompt_tps
        generation_tps = getattr(result, "generation_tps", None)
        if generation_tps is not None:
            self.generation_tps = generation_tps
        cached_tokens = getattr(result, "cached_tokens", None)
        if cached_tokens is not None:
            self.cached_tokens = max(self.cached_tokens, int(cached_tokens))
        draft_kind = getattr(result, "draft_kind", None)
        if draft_kind is not None:
            self.draft_kind = draft_kind
        draft_rounds = getattr(result, "draft_rounds", None)
        if draft_rounds is not None:
            self.draft_rounds = int(draft_rounds)
        draft_n_accepted = getattr(result, "draft_n_accepted", None)
        if draft_n_accepted is not None:
            self.draft_n_accepted = int(draft_n_accepted)
        draft_n = getattr(result, "draft_n", None)
        if draft_n is not None:
            self.draft_n = int(draft_n)
        self.capacity_exhausted = bool(
            self.capacity_exhausted
            or getattr(result, "capacity_exhausted", False)
        )
        guaranteed = getattr(result, "guaranteed_output_tokens", None)
        if guaranteed is not None:
            self.guaranteed_output_tokens = int(guaranteed)
        elastic = getattr(result, "elastic_output_tokens", None)
        if elastic is not None:
            self.elastic_output_tokens = max(
                self.elastic_output_tokens, int(elastic)
            )
        wait_ms = getattr(result, "capacity_wait_ms", None)
        if wait_ms is not None:
            self.capacity_wait_ms = max(self.capacity_wait_ms, float(wait_ms))


@dataclass
class StreamingToken:
    """A single token response during streaming generation.

    Diffusion models stream block-by-block: one StreamingToken per denoised
    block, with ``token_count`` carrying the number of tokens in the block.
    """

    text: str
    token: int
    logprobs: float
    finish_reason: Optional[str]
    peak_memory: float = 0.0
    prompt_tps: Optional[float] = None
    generation_tps: Optional[float] = None
    draft_kind: Optional[str] = None
    draft_rounds: Optional[int] = None
    draft_n_accepted: Optional[int] = None
    draft_n: Optional[int] = None
    top_logprobs: Optional[List[Tuple[int, float]]] = None
    cached_tokens: int = 0
    token_count: int = 1
    emitted_at: Optional[float] = None
    capacity_exhausted: bool = False
    guaranteed_output_tokens: Optional[int] = None
    elastic_output_tokens: int = 0
    capacity_wait_ms: float = 0.0


class _DiffusionBlockEmitter:
    """Incrementally group diffusion results into block streaming tokens."""

    def __init__(self):
        self.block_text: List[str] = []
        self.last_token = 0
        self.emitted_tokens = 0

    def feed(self, result) -> "Generator[StreamingToken, None, None]":
        if result.is_draft:
            return
        if result.text:
            self.block_text.append(result.text)
        if result.token is not None:
            self.last_token = int(result.token)
        if not result.diffusion_block_complete and not result.finish_reason:
            return
        if result.finish_reason or self.block_text:
            token_count = max(result.generation_tokens - self.emitted_tokens, 0)
            self.emitted_tokens = result.generation_tokens
            yield StreamingToken(
                text="".join(self.block_text),
                token=self.last_token,
                logprobs=None,
                finish_reason=result.finish_reason,
                peak_memory=result.peak_memory,
                prompt_tps=result.prompt_tps,
                generation_tps=result.generation_tps,
                cached_tokens=result.cached_tokens,
                token_count=token_count,
            )
            self.block_text = []


def _diffusion_block_chunks(results) -> "Generator[StreamingToken, None, None]":
    """Group diffusion engine results into block-by-block streaming tokens.

    The diffusion engine emits a canvas's tokens right after that canvas
    finishes denoising, followed by a block-boundary marker. Each completed
    block becomes one StreamingToken; the final token carries the finish
    reason (plus any text flushed by detokenizer finalization).
    """
    emitter = _DiffusionBlockEmitter()
    for result in results:
        for chunk in emitter.feed(result):
            yield chunk
            if chunk.finish_reason:
                return


class _TokenIterator:
    """Closeable iterator over queued tokens for one generation request.

    close() cancels unfinished requests and is safe while another thread is
    blocked in __next__ waiting for the next token.
    """

    def __init__(self, rqueue, uid, cancel_fn, queue_timeout):
        self._rqueue = rqueue
        self._uid = uid
        self._cancel_fn = cancel_fn
        self._queue_timeout = queue_timeout
        self._ended = False
        self._closed = False
        self._lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        if self._ended:
            raise StopIteration
        while True:
            try:
                item = self._rqueue.get(timeout=self._queue_timeout)
            except QueueEmpty as exc:
                # Consumer is stalled or upstream is wedged — treat as cancel.
                self.close()
                label = (
                    "without a timeout"
                    if self._queue_timeout is None
                    else f"for {self._queue_timeout:g}s"
                )
                raise RuntimeError(
                    "Timed out waiting "
                    f"{label} for the next generated token or GPU activity. "
                    "Increase MLX_VLM_TOKEN_QUEUE_TIMEOUT for unusually slow "
                    "steps."
                ) from exc
            if item is _TOKEN_QUEUE_ACTIVITY:
                continue
            break
        if item is None:
            self._ended = True
            raise StopIteration
        if isinstance(item, Exception):
            self._ended = True
            raise item
        if getattr(item, "finish_reason", None):
            self._ended = True
        return item

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if not self._ended:
            self._cancel_fn(self._uid)

    def __del__(self):
        # Mirror generator semantics: implicit close on GC.
        try:
            self.close()
        except Exception:
            pass


class ResponseGenerator:
    """
    Continuous batching for concurrent requests via a single GPU thread.

    A dedicated thread owns all GPU work (BatchGenerator). FastAPI async
    handlers submit requests to a queue and read tokens back from
    per-request queues. Multiple requests are batched together for
    higher throughput — same pattern as mlx-lm's server.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        vision_cache=None,
        kv_bits=None,
        kv_key_bits=None,
        kv_value_bits=None,
        kv_key_scheme=None,
        kv_value_scheme=None,
        kv_group_size=DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme=DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start=DEFAULT_QUANTIZED_KV_START,
        top_logprobs_k=0,
        apc_manager: Optional["_apc.APCManager"] = None,
        draft_model_path: Optional[str] = None,
        draft_kind: Optional[str] = None,
        defer_draft_model: bool = False,
        vision_phase_swap_path: Optional[str] = None,
        chunk_local_input_embeddings: bool = False,
        prefill_step_size: Optional[int] = None,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.config = None
        self.stop_tokens = set()
        self.vision_cache = vision_cache
        self.draft_model_path = draft_model_path
        self.draft_kind_override = draft_kind
        self.defer_draft_model = bool(defer_draft_model) or os.environ.get(
            "MLX_VLM_DEFER_DRAFT_MODEL", "0"
        ).lower() in ("1", "true", "yes")
        self.draft_model = None
        self.vision_phase_swap_path = vision_phase_swap_path
        self.vision_phase_swap = None
        self.chunk_local_input_embeddings = bool(
            chunk_local_input_embeddings
        ) or os.environ.get("MLX_VLM_CHUNK_LOCAL_INPUT_EMBEDS", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.kv_bits = kv_bits
        self.kv_key_bits = kv_key_bits
        self.kv_value_bits = kv_value_bits
        self.kv_key_scheme = kv_key_scheme
        self.kv_value_scheme = kv_value_scheme
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.quantized_kv_start = quantized_kv_start
        self.top_logprobs_k = top_logprobs_k
        self.apc_manager = apc_manager
        self.prefill_step_size = (
            get_prefill_step_size()
            if prefill_step_size is None
            else int(prefill_step_size)
        )
        self.apc_mode = None
        self.tokenizer = None
        self.requests: Queue = Queue()
        self._stop = False
        self._ready = Event()
        self._load_error: Optional[Exception] = None
        # Paged storage is a server resource. Request cohorts borrow facades
        # from it and return their pages, while the backing arrays stay alive
        # across idle gaps and sequential requests.
        self._paged_registry = None
        self._cancelled: set = set()
        self._cancel_lock = Lock()
        self._tokenizer_lock = Lock()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _effective_prefill_step_size(self) -> int:
        prefill_step_size = getattr(self, "prefill_step_size", None)
        return (
            get_prefill_step_size()
            if prefill_step_size is None
            else int(prefill_step_size)
        )

    def _unload_deferred_drafter(self) -> None:
        unload = getattr(getattr(self, "draft_model", None), "unload", None)
        if callable(unload):
            unload()

    def stop_and_join(self, timeout: float = 10.0):
        self._stop = True
        self.requests.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.info(
                "Generation thread still draining in-flight requests; "
                "letting it finish in the background."
            )

    def wait_until_ready(self, timeout: Optional[float] = None):
        if not self._ready.wait(timeout):
            raise RuntimeError("Timed out waiting for generation thread to load model.")
        if self._load_error is not None:
            raise self._load_error
        return self.model, self.processor, self.config

    def _cancel(self, uid):
        with self._cancel_lock:
            self._cancelled.add(uid)

    def _drain_cancellations(self) -> set:
        with self._cancel_lock:
            pending, self._cancelled = self._cancelled, set()
            return pending

    def _initialize_model(self):
        model, processor, config = load_model_resources(
            self.model_path, self.adapter_path
        )

        stop_tokens = set(
            resolve_eos_token_ids(
                getattr(config, "eos_token_id", None),
                getattr(processor, "tokenizer", processor),
            )
        )
        stop_tokens.update(getattr(processor, "additional_eos_token_ids", ()))
        if capacity_ignore_eos_enabled():
            stop_tokens.clear()
            tokenizer = getattr(processor, "tokenizer", processor)
            stopping_criteria = getattr(tokenizer, "stopping_criteria", None)
            if stopping_criteria is not None:
                # StoppingCriteria.reset([]) deliberately re-adds the
                # tokenizer's own EOS token. Capacity qualification needs an
                # actually empty predicate so generation reaches max_tokens.
                stopping_criteria.eos_token_ids = []
            logger.warning(
                "Capacity benchmark mode enabled: EOS stopping is disabled."
            )

        draft_model = None
        draft_kind = self.draft_kind_override or os.environ.get("MLX_VLM_DRAFT_KIND")
        draft_model_path = self.draft_model_path or os.environ.get(
            "MLX_VLM_DRAFT_MODEL"
        )
        if draft_model_path:
            from ..speculative.drafters import (
                load_drafter,
                validate_drafter_compatibility,
            )

            logger.info(
                "Loading speculative drafter (%s): %s",
                draft_kind or "auto",
                draft_model_path,
            )
            draft_model, resolved_kind = load_drafter(draft_model_path, kind=draft_kind)
            if draft_kind is None:
                logger.info("Auto-detected speculative draft kind: %s", resolved_kind)
            elif resolved_kind != draft_kind:
                logger.warning(
                    "Drafter requires draft kind %s; using it instead of %s.",
                    resolved_kind,
                    draft_kind,
                )
            draft_kind = resolved_kind
            try:
                validate_drafter_compatibility(model, draft_model, draft_kind)
            except ValueError as e:
                logger.warning(
                    "Speculative drafter is incompatible with the target model; "
                    "falling back to autoregressive generation: %s",
                    e,
                )
                draft_model = None
                draft_kind = None
            else:
                logger.info("Drafter ready; speculative decoding enabled.")

                if getattr(self, "defer_draft_model", False):
                    if draft_kind != "mtp":
                        raise ValueError(
                            "Deferred drafter loading currently supports MTP only."
                        )
                    draft_model = LazyDrafter(
                        path=draft_model_path,
                        kind=draft_kind,
                        config=getattr(draft_model, "config", None),
                        loader=load_drafter,
                        validator=validate_drafter_compatibility,
                        target_model=model,
                    )
                    gc.collect()
                    mx.clear_cache()
                    logger.info(
                        "MTP drafter weights released until speculative decode."
                    )

        self.model = model
        self.processor = processor
        self.config = config
        if (
            self.chunk_local_input_embeddings
            and getattr(config, "model_type", None) != "qwen3_5"
        ):
            raise ValueError(
                "Chunk-local input embeddings currently support Qwen3.5 only."
            )
        self.stop_tokens = stop_tokens
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.component_residency = ComponentResidencyManager()
        vision_component = self.vision_phase_swap_path or os.environ.get(
            "MLX_VLM_VISION_PHASE_SWAP_PATH"
        )
        if vision_component:
            self.vision_phase_swap = VisionTowerPhaseSwap(model, vision_component)
            self.component_residency.register(
                "vision_tower", self.vision_phase_swap
            )
        language_head_component = os.environ.get(
            "MLX_VLM_LANGUAGE_HEAD_PHASE_SWAP_PATH"
        )
        if language_head_component:
            language_model = getattr(model, "language_model", model)
            language_model.prefill_head_phase_swap = LanguageHeadPhaseSwap(
                language_model, language_head_component
            )
            self.component_residency.register(
                "lm_head", language_model.prefill_head_phase_swap
            )
            language_model.phase_residency_manager = self.component_residency
        embedding_component = os.environ.get("MLX_VLM_INPUT_EMBEDDING_PHASE_SWAP_PATH")
        if embedding_component:
            if get_max_num_seqs() != 1:
                raise ValueError("Embedding phase swap requires --max-num-seqs 1.")
            language_model = getattr(model, "language_model", model)
            language_model.prefill_embedding_phase_swap = LanguageEmbeddingPhaseSwap(
                language_model, embedding_component
            )
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.apc_mode = None
        if self.apc_manager is not None:
            language_model = getattr(model, "language_model", model)
            self.apc_mode = _apc.model_apc_mode(language_model)

    def generate(
        self,
        prompt: str,
        images: Optional[List] = None,
        audio: Optional[List] = None,
        args: Optional[GenerationArguments] = None,
        videos: Optional[List] = None,
        cancel_event: Optional[Event] = None,
    ) -> Tuple[GenerationContext, "_TokenIterator"]:
        self.wait_until_ready()
        args = args or GenerationArguments(max_tokens=get_server_max_tokens())
        if self.draft_model is not None and args.logits_processors is not None:
            raise ValueError(
                "Structured response_format is not supported with speculative decoding."
            )
        if (
            self.draft_model is not None
            and self.draft_kind != "mtp"
            and args.thinking_budget is not None
        ):
            raise ValueError(
                "thinking_budget is only supported with MTP speculative decoding "
                "in the server."
            )
        rqueue: Queue = Queue()
        request_started_at = time.perf_counter()

        # CPU preprocessing and thinking-token resolution share tokenizer state.
        # Keep both on the caller side and serialize them so the GPU worker never
        # races request threads through the mutable fast-tokenizer backend.
        tokenizer_lock = getattr(self, "_tokenizer_lock", None)
        with tokenizer_lock if tokenizer_lock is not None else nullcontext():
            raw_inputs = self._preprocess_request(prompt, images, audio, videos)
            thinking_budget_criteria = self._make_thinking_budget_criteria(
                args, raw_inputs.get("input_ids")
            )
        prompt_tokens = _count_prompt_tokens(raw_inputs)
        _check_configured_context_budget(prompt_tokens, args.max_tokens)

        apc_semantic_hash = None
        if getattr(self, "apc_mode", None) is not None:
            pixel_values = raw_inputs.get("pixel_values")
            image_hash = 0
            if pixel_values is not None:
                image_hash = _apc.hash_image_payload(pixel_values=pixel_values)
            elif images is not None:
                image_hash = _apc.hash_image_payload(image_ref=images)
            apc_semantic_hash = _apc.semantic_extra_hash(
                tenant=getattr(args, "tenant_id", None),
                image_hash=image_hash,
                media={
                    "audio": raw_inputs.get("input_features"),
                    "video": raw_inputs.get("pixel_values_videos"),
                },
                model=getattr(self.model, "language_model", self.model),
                processor=self.processor,
            )

        request_id = f"{id(rqueue):x}"
        queued_request = QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs=raw_inputs,
            prompt_tokens=prompt_tokens,
            args=args,
            thinking_budget_criteria=thinking_budget_criteria,
            images=images,
            videos=videos,
            audio=audio,
            apc_semantic_hash=apc_semantic_hash,
            request_id=request_id,
            cancel_event=cancel_event,
            queued_at=request_started_at,
        )
        logger.info(
            "Generation queued: request=%s prompt_tokens=%d max_tokens=%d "
            "images=%d audio=%d videos=%d",
            request_id,
            prompt_tokens,
            args.max_tokens,
            len(images or []),
            len(audio or []),
            len(videos or []),
        )
        self.requests.put(queued_request)

        # Block until the GPU thread sends back the context
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Generation cancelled before GPU admission.")
            try:
                ctx = rqueue.get(timeout=0.1 if cancel_event is not None else None)
                break
            except QueueEmpty:
                continue
        if isinstance(ctx, Exception):
            raise ctx

        return ctx, _TokenIterator(
            rqueue, ctx.uid, self._cancel, get_token_queue_timeout()
        )

    def _cpu_preprocess(self, prompt, images=None, audio=None, videos=None) -> dict:
        """CPU-only: tokenize text, load/resize images. Thread-safe."""
        add_special_tokens = (
            getattr(self.processor, "chat_template", None) is None
            if self.model.config.model_type
            in ["gemma3", "gemma3n", "gemma4", "gemma4_unified"]
            else True
        )
        image_token_index = getattr(self.model.config, "image_token_index", None)
        return prepare_inputs(
            self.processor,
            images=images,
            audio=audio,
            videos=videos,
            prompts=prompt,
            image_token_index=image_token_index,
            add_special_tokens=add_special_tokens,
        )

    def _preprocess_request(self, prompt, images=None, audio=None, videos=None) -> dict:
        if videos is None:
            return self._cpu_preprocess(prompt, images, audio)
        return self._cpu_preprocess(prompt, images, audio, videos)

    # -- internals --

    @staticmethod
    def _request_log_id(request: QueuedGenerationRequest) -> str:
        if request.request_id is None:
            request.request_id = f"{id(request.rqueue):x}"
        return request.request_id

    def _log_prefill_started(
        self, request: QueuedGenerationRequest, *, backend: str
    ) -> dict:
        request_id = self._request_log_id(request)
        now = time.perf_counter()
        logger.info(
            "Prefill started: request=%s backend=%s prompt_tokens=%d "
            "images=%d audio=%d videos=%d",
            request_id,
            backend,
            request.prompt_tokens,
            len(request.images or []),
            len(request.audio or []),
            len(request.videos or []),
        )
        return {
            "request_id": request_id,
            "queued_at": request.queued_at,
            "prefill_started_at": now,
            "prefill_processed": -1,
            "generated_tokens": 0,
            "decode_started_at": None,
            "last_token_at": None,
        }

    def _log_prefill_progress(self, batch_gen, active: dict) -> None:
        """Report each chunked-prefill step without changing generation output."""
        prompt_batch = getattr(batch_gen, "_prompt_batch", None)
        if prompt_batch is None:
            return
        processed_columns = int(
            getattr(prompt_batch, "_processed_prompt_columns", 0) or 0
        )
        remaining = getattr(prompt_batch, "_inputs_embeds", None)
        remaining_columns = (
            int(remaining.shape[1]) if getattr(remaining, "ndim", 0) >= 2 else 0
        )
        total_columns = processed_columns + remaining_columns
        uids = list(getattr(prompt_batch, "uids", []))
        suffix_lengths = list(getattr(prompt_batch, "_suffix_lens", []))
        cached_tokens = list(getattr(prompt_batch, "_cached_tokens_per_row", []))
        left_padding = list(getattr(prompt_batch, "_left_padding_per_row", []))
        right_padding = getattr(prompt_batch, "_right_pad_per_row", None)

        for index, uid in enumerate(uids):
            info = active.get(uid)
            if info is None:
                continue
            suffix_len = (
                int(suffix_lengths[index])
                if index < len(suffix_lengths)
                else total_columns
            )
            cached = int(cached_tokens[index]) if index < len(cached_tokens) else 0
            if right_padding is not None:
                suffix_processed = min(suffix_len, processed_columns)
            else:
                pad = int(left_padding[index]) if index < len(left_padding) else 0
                suffix_processed = min(suffix_len, max(0, processed_columns - pad))
            processed = cached + suffix_processed
            total = cached + suffix_len
            if processed <= int(info.get("prefill_processed", -1)):
                continue
            info["prefill_processed"] = processed
            percent = 100.0 * processed / total if total > 0 else 100.0
            logger.info(
                "Prefill progress: request=%s tokens=%d/%d (%.1f%%)",
                info.get("request_id", uid),
                processed,
                total,
                percent,
            )

    @staticmethod
    def _log_prefill_completed(uid, info: dict, prompt_response) -> None:
        prompt_tokens = int(getattr(prompt_response, "prompt_tokens", 0) or 0)
        cached_tokens = int(getattr(prompt_response, "cached_tokens", 0) or 0)
        prompt_tps = float(getattr(prompt_response, "prompt_tps", 0.0) or 0.0)
        prompt_time = float(getattr(prompt_response, "prompt_time", 0.0) or 0.0)
        if prompt_time <= 0 and prompt_tps > 0:
            prompt_time = prompt_tokens / prompt_tps
        info["prefill_processed"] = prompt_tokens
        logger.info(
            "Prefill completed: request=%s prompt_tokens=%d cached_tokens=%d "
            "elapsed=%.3fs rate=%.1f tok/s",
            info.get("request_id", uid),
            prompt_tokens,
            cached_tokens,
            prompt_time,
            prompt_tps,
        )

    @staticmethod
    def _log_decode_progress(
        uid,
        info: dict,
        *,
        token: int,
        text: str,
        finish_reason: Optional[str],
        token_count: int = 1,
    ) -> float:
        now = time.perf_counter()
        previous_tokens = int(info.get("generated_tokens", 0) or 0)
        emitted_tokens = max(0, int(token_count or 0))
        generated_tokens = previous_tokens + emitted_tokens
        info["generated_tokens"] = generated_tokens
        request_id = info.get("request_id", uid)

        previous_token_at = info.get("last_token_at")
        token_rate = None
        if emitted_tokens > 0:
            if previous_token_at is not None and now > previous_token_at:
                token_rate = emitted_tokens / (now - previous_token_at)
            info["last_token_at"] = now
        progress_rate_text = "n/a" if token_rate is None else f"{token_rate:.1f} tok/s"

        decode_started_at = info.get("decode_started_at")
        if decode_started_at is None:
            decode_started_at = now
            info["decode_started_at"] = now
            queued_at = float(info.get("queued_at", now) or now)
            logger.info(
                "Decode started: request=%s time_to_first_token=%.3fs",
                request_id,
                max(0.0, now - queued_at),
            )

        elapsed = max(0.0, now - decode_started_at)
        first_chunk_tokens = int(info.get("decode_first_chunk_tokens", 0) or 0)
        if first_chunk_tokens == 0:
            first_chunk_tokens = emitted_tokens
            info["decode_first_chunk_tokens"] = emitted_tokens
        measured_tokens = max(0, generated_tokens - first_chunk_tokens)
        rate = measured_tokens / elapsed if elapsed > 0 else 0.0
        interval = get_log_progress_interval()
        crossed_interval = interval > 0 and (
            generated_tokens // interval > previous_tokens // interval
        )
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        if debug_enabled:
            logger.debug(
                "Decode progress: request=%s generated_tokens=%d elapsed=%.3fs "
                "rate=%s token_number=%d token_id=%s text=%r",
                request_id,
                generated_tokens,
                elapsed,
                progress_rate_text,
                generated_tokens,
                token,
                text,
            )

        if finish_reason is not None:
            logger.info(
                "Decode completed: request=%s generated_tokens=%d elapsed=%.3fs "
                "rate=%.1f tok/s finish_reason=%s",
                request_id,
                generated_tokens,
                elapsed,
                rate,
                finish_reason,
            )
        elif crossed_interval and not debug_enabled:
            logger.info(
                "Decode progress: request=%s generated_tokens=%d elapsed=%.3fs "
                "rate=%s",
                request_id,
                generated_tokens,
                elapsed,
                progress_rate_text,
            )
        return now

    def _make_sampler(self, args: GenerationArguments) -> Optional[Callable]:
        if args.temperature == 0:
            return None
        if args.top_n_sigma > 0:
            return make_sampler(
                temp=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                top_n_sigma=args.top_n_sigma,
            )
        if args.p_less:
            return make_sampler(
                temp=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                p_less=True,
            )
        if args.typical_p < 1.0:
            return make_sampler(
                temp=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                typical_p=args.typical_p,
            )
        return _PositionedTargetSampler(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )

    def _make_logits_processors(
        self, args: GenerationArguments, input_ids: Optional[mx.array] = None
    ) -> List[Callable[[mx.array, mx.array], mx.array]]:
        processors = make_logits_processors(
            args.logit_bias,
            args.repetition_penalty,
            args.repetition_context_size,
            args.presence_penalty,
            args.presence_context_size,
            args.frequency_penalty,
            args.frequency_context_size,
        )
        if args.logits_processors is not None:
            request_processors = args.logits_processors
            already_closed = (
                input_ids is not None
                and self._prompt_thinking_already_closed(args, input_ids)
            )
            if args.enable_thinking and not already_closed:
                request_processors = self._wrap_processors_until_thinking_done(
                    args, request_processors
                )
            processors.extend(request_processors)
        return processors

    def _thinking_token_ids(self, args: GenerationArguments) -> Tuple[int, int]:
        tokenizer = self.tokenizer
        thinking_start_token = args.thinking_start_token or DEFAULT_THINKING_START_TOKEN
        thinking_end_token = args.thinking_end_token or DEFAULT_THINKING_END_TOKEN
        thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        thinking_end_token_id = tokenizer.encode(
            thinking_end_token, add_special_tokens=False
        )[-1]
        return thinking_start_token_id, thinking_end_token_id

    def _prompt_has_open_thinking(
        self, args: GenerationArguments, input_ids: mx.array
    ) -> bool:
        if not args.enable_thinking:
            return False
        thinking_start_token_id, thinking_end_token_id = self._thinking_token_ids(args)
        tokens = input_ids.flatten().tolist()
        try:
            last_start = len(tokens) - 1 - tokens[::-1].index(thinking_start_token_id)
        except ValueError:
            return False
        try:
            last_end = len(tokens) - 1 - tokens[::-1].index(thinking_end_token_id)
        except ValueError:
            last_end = -1
        return last_start > last_end

    def _prompt_thinking_already_closed(
        self, args: GenerationArguments, input_ids: mx.array
    ) -> bool:
        if not args.enable_thinking:
            return False
        thinking_start_token_id, thinking_end_token_id = self._thinking_token_ids(args)
        tokens = input_ids.flatten().tolist()
        try:
            last_start = len(tokens) - 1 - tokens[::-1].index(thinking_start_token_id)
        except ValueError:
            return False
        try:
            last_end = len(tokens) - 1 - tokens[::-1].index(thinking_end_token_id)
        except ValueError:
            return False
        return last_end > last_start

    def _wrap_processors_until_thinking_done(
        self,
        args: GenerationArguments,
        processors: List[Callable[[mx.array, mx.array], mx.array]],
    ) -> List[Callable[[mx.array, mx.array], mx.array]]:
        thinking_start_token = args.thinking_start_token or DEFAULT_THINKING_START_TOKEN
        thinking_end_token = args.thinking_end_token or DEFAULT_THINKING_END_TOKEN
        return [
            ThinkingAwareLogitsProcessor(
                processor=processor,
                tokenizer=self.tokenizer,
                thinking_start_token=thinking_start_token,
                thinking_end_token=thinking_end_token,
                enable_thinking=True,
            )
            for processor in processors
        ]

    def _make_thinking_budget_criteria(
        self, args: GenerationArguments, input_ids: mx.array
    ) -> Optional[ThinkingBudgetCriteria]:
        if args.thinking_budget is None:
            return None
        tokenizer = self.tokenizer
        thinking_start_token = args.thinking_start_token or DEFAULT_THINKING_START_TOKEN
        thinking_end_token = args.thinking_end_token or DEFAULT_THINKING_END_TOKEN
        prompt_preopens_thinking = self._prompt_has_open_thinking(args, input_ids)
        return ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=args.thinking_budget,
            thinking_end_token=thinking_end_token,
            thinking_start_token=thinking_start_token,
            enable_thinking=args.enable_thinking,
            prompt_preopens_thinking=prompt_preopens_thinking,
        )

    def _gpu_embed(
        self,
        raw_inputs: dict,
        images=None,
        apc_semantic_hash: Optional[int] = None,
    ) -> Tuple[mx.array, dict]:
        """GPU-only: run vision encoder if needed. Must run on GPU thread."""
        input_ids = raw_inputs.get("input_ids")
        pixel_values = raw_inputs.get("pixel_values")
        mask = raw_inputs.get("attention_mask")
        data_kwargs = {
            k: v
            for k, v in raw_inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        media_uses_vision = pixel_values is not None or any(
            data_kwargs.get(name) is not None
            for name in ("pixel_values_videos", "image_grid_thw", "video_grid_thw")
        )
        phase_swap = (
            getattr(self, "vision_phase_swap", None) if media_uses_vision else None
        )
        if phase_swap is not None:
            residency = getattr(self, "component_residency", None)
            if residency is not None and residency.contains("vision_tower"):
                residency.acquire("vision_tower", "media_embedding")
            else:
                phase_swap.load()
        # Pass vision cache for image feature caching
        if (
            pixel_values is not None
            and self.vision_cache is not None
            and images is not None
        ):
            data_kwargs["vision_cache"] = self.vision_cache
            data_kwargs["_image_key"] = images

        # Always call get_input_embeddings — BatchGenerator requires inputs_embeds
        try:
            embedding_kwargs = dict(data_kwargs)
            if getattr(self, "chunk_local_input_embeddings", False):
                embedding_kwargs["chunked"] = True
            embed = self.model.get_input_embeddings(
                input_ids,
                pixel_values,
                mask=mask,
                **embedding_kwargs,
            )
            if phase_swap is not None:
                arrays = [
                    value
                    for value in embed.to_dict().values()
                    if value is not None and hasattr(value, "shape")
                ]
                provider = getattr(embed, "input_embedding_provider", None)
                image_features = getattr(provider, "image_features", None)
                if image_features is not None:
                    arrays.append(image_features)
                if arrays:
                    mx.eval(arrays)
        finally:
            if phase_swap is not None:
                residency = getattr(self, "component_residency", None)
                if residency is not None and residency.contains("vision_tower"):
                    residency.release("vision_tower", "media_embedding")
                else:
                    phase_swap.unload()
        # Remove cache kwargs before passing to BatchGenerator
        data_kwargs.pop("vision_cache", None)
        data_kwargs.pop("_image_key", None)
        gen_kwargs = {
            **data_kwargs,
            **{k: v for k, v in embed.to_dict().items() if v is not None},
        }
        provider = getattr(embed, "input_embedding_provider", None)
        if provider is not None:
            gen_kwargs["input_embedding_provider"] = provider
        if apc_semantic_hash is not None:
            gen_kwargs["_apc_semantic_hash"] = apc_semantic_hash
        return input_ids, gen_kwargs

    def _collect_pending_requests(
        self,
        *,
        active: bool,
        capacity: Optional[int] = None,
        idle_timeout: float = 0.1,
        coalesce_s: float = 0.0,
    ):
        """Collect the first queued request, then drain immediately available peers.

        When ``capacity`` is set, admit at most ``capacity`` new requests and leave
        the rest queued (backpressure), so the running batch never exceeds
        ``--max-num-seqs`` concurrent sequences.
        """
        pending = []
        should_stop = False

        def append_item(item):
            nonlocal should_stop
            if item is None:
                if self._stop and not pending:
                    should_stop = True
                return
            pending.append(item)

        def _has_room():
            return capacity is None or len(pending) < capacity

        try:
            if active:
                if _has_room():
                    append_item(self.requests.get_nowait())
            else:
                append_item(self.requests.get(timeout=idle_timeout))
        except QueueEmpty:
            pass

        if pending and coalesce_s > 0:
            time.sleep(coalesce_s)

        while not should_stop and _has_room():
            try:
                append_item(self.requests.get_nowait())
            except QueueEmpty:
                break

        return pending, should_stop

    def _uses_phase_cohort_admission(self) -> bool:
        """Keep phase-swapped prefills out of an active decode cohort."""
        return bool(
            getattr(self, "chunk_local_input_embeddings", False)
            or getattr(self, "vision_phase_swap", None) is not None
            or isinstance(getattr(self, "draft_model", None), LazyDrafter)
        )

    @staticmethod
    def _is_text_only_request(request: QueuedGenerationRequest) -> bool:
        return not (request.images or request.audio or request.videos)

    @staticmethod
    def _request_cancelled_before_admission(
        request: QueuedGenerationRequest,
    ) -> bool:
        event = getattr(request, "cancel_event", None)
        return event is not None and event.is_set()

    def _collect_active_phase_text_requests(self, capacity: Optional[int]):
        """Admit queued text rows while leaving media at the cohort boundary."""
        admitted = []
        deferred = []
        should_stop = False
        scan = self.requests.qsize()

        for _ in range(scan):
            try:
                item = self.requests.get_nowait()
            except QueueEmpty:
                break
            if item is None:
                deferred.append(item)
                continue
            if self._request_cancelled_before_admission(item):
                logger.info(
                    "Dropped cancelled queued request before phase admission: "
                    "request=%s",
                    self._request_log_id(item),
                )
                continue
            if (
                capacity is None or len(admitted) < capacity
            ) and self._is_text_only_request(item):
                admitted.append(item)
            else:
                deferred.append(item)

        for item in deferred:
            self.requests.put(item)
        if self._stop and not admitted and not deferred:
            should_stop = True
        return admitted, should_stop

    def _partition_lm_head_phase_admission(self, pending, active):
        """Keep unqualified mixed prefills at a safe cohort boundary.

        Paged prefill is B1. Requests collected together therefore become
        mixed-prefill work after the first row reaches decode. Admit one cold
        request plus only peers within the explicitly qualified mixed limit.
        """

        residency = getattr(self, "component_residency", None)
        if (
            not pending
            or residency is None
            or not residency.contains("lm_head")
        ):
            return list(pending), []

        limit = get_lm_head_mixed_prefill_max_tokens()
        admitted = []
        deferred = []
        cold_slot_available = not active
        fairness_blocked = False
        admitted_indexes = []
        for index, request in enumerate(pending):
            if fairness_blocked:
                deferred.append(request)
                continue
            if cold_slot_available:
                admitted.append(request)
                admitted_indexes.append(index)
                cold_slot_available = False
            elif self._request_mixed_prefill_tokens(request) <= limit:
                admitted.append(request)
                admitted_indexes.append(index)
            else:
                deferred.append(request)
                if int(getattr(request, "kv_bypass_count", 0)) >= (
                    get_paged_scheduler_max_bypass()
                ):
                    fairness_blocked = True

        if admitted_indexes:
            last_admitted = max(admitted_indexes)
            admitted_ids = {id(request) for request in admitted}
            for index, request in enumerate(pending):
                if index >= last_admitted:
                    break
                if id(request) not in admitted_ids:
                    request.kv_bypass_count = (
                        int(getattr(request, "kv_bypass_count", 0)) + 1
                    )
        return admitted, deferred

    def _request_mixed_prefill_tokens(self, request) -> int:
        """Estimate model-prefill work after an exact APC prefix match."""
        prompt_tokens = max(0, int(getattr(request, "prompt_tokens", 0) or 0))
        if getattr(request, "apc_prefix_probe_done", False):
            prefix_tokens = int(
                getattr(request, "apc_prefix_tokens_hint", 0) or 0
            )
            return max(0, prompt_tokens - prefix_tokens)

        request.apc_prefix_probe_done = True
        request.apc_prefix_tokens_hint = 0
        manager = getattr(self, "apc_manager", None)
        if (
            manager is None
            or getattr(self, "apc_mode", None) != "exact"
            or not self._is_text_only_request(request)
        ):
            return prompt_tokens

        raw_inputs = getattr(request, "raw_inputs", None) or {}
        input_ids = raw_inputs.get("input_ids")
        if input_ids is None:
            return prompt_tokens
        try:
            if hasattr(input_ids, "reshape"):
                input_ids = input_ids.reshape(-1)
            token_ids = (
                input_ids.tolist()
                if hasattr(input_ids, "tolist")
                else list(input_ids)
            )
            if token_ids and isinstance(token_ids[0], (list, tuple)):
                token_ids = [token for row in token_ids for token in row]
            prefix_tokens = manager.peek_exact_prefix_length(
                token_ids,
                extra_hash=int(getattr(request, "apc_semantic_hash", 0) or 0),
            )
        except Exception:
            logger.exception(
                "APC prefix admission probe failed: request=%s",
                self._request_log_id(request),
            )
            return prompt_tokens

        prefix_tokens = min(prompt_tokens, max(0, int(prefix_tokens)))
        request.apc_prefix_tokens_hint = prefix_tokens
        if prefix_tokens:
            logger.info(
                "APC-aware phase admission: request=%s prompt_tokens=%d "
                "prefix_tokens=%d suffix_tokens=%d",
                self._request_log_id(request),
                prompt_tokens,
                prefix_tokens,
                prompt_tokens - prefix_tokens,
            )
        return max(0, prompt_tokens - prefix_tokens)

    def _admission_capacity(
        self, *, active_count: int, max_num_seqs: Optional[int]
    ) -> Optional[int]:
        if (
            active_count
            and self._uses_phase_cohort_admission()
            and getattr(self, "draft_model", None) is not None
            and getattr(self, "draft_kind", None) != "mtp"
        ):
            return 0
        if max_num_seqs is None:
            return None
        return max(0, max_num_seqs - active_count)

    @staticmethod
    def _request_context_budget(request: QueuedGenerationRequest) -> int:
        requested_output = max(0, int(request.args.max_tokens or 0))
        output_guarantee = get_paged_output_guarantee_tokens()
        if paged_turboquant_enabled() and output_guarantee is not None:
            requested_output = min(requested_output, output_guarantee)
        return max(0, int(request.prompt_tokens)) + requested_output

    @staticmethod
    def _active_context_budget(info: dict) -> Optional[int]:
        """Return an active row's current reservation plus elastic usage."""

        prompt_tokens = info.get("prompt_tokens")
        output_guarantee = info.get("guaranteed_output_tokens")
        if prompt_tokens is not None and output_guarantee is not None:
            generated_tokens = max(0, int(info.get("generated_tokens", 0) or 0))
            return max(0, int(prompt_tokens)) + max(
                max(0, int(output_guarantee)), generated_tokens
            )
        value = info.get("context_budget_tokens")
        return None if value is None else max(0, int(value))

    @staticmethod
    def _page_round_tokens(tokens: int) -> int:
        return (
            (max(0, int(tokens)) + PAGED_TURBOQUANT_PAGE_SIZE - 1)
            // PAGED_TURBOQUANT_PAGE_SIZE
            * PAGED_TURBOQUANT_PAGE_SIZE
        )

    def _elastic_capacity_victims(self, active: dict) -> list[int]:
        """Choose rows to finish before their next token can exceed the pool.

        Admission guarantees remain protected. Only rows that have already
        generated their guarantee are eligible, and the row using the most
        elastic surplus yields first. Ties prefer the newest request.
        """

        if not active or not paged_turboquant_enabled():
            return []
        if get_paged_output_guarantee_tokens() is None:
            return []
        capacity = get_paged_kv_capacity_tokens()
        if capacity is None:
            return []
        usable = max(
            0,
            int(capacity)
            - self._page_round_tokens(get_paged_kv_safety_tokens()),
        )
        budgets = {
            uid: self._page_round_tokens(
                int(info.get("prompt_tokens", 0) or 0)
                + max(
                    int(info.get("guaranteed_output_tokens", 0) or 0),
                    int(info.get("generated_tokens", 0) or 0) + 1,
                )
            )
            for uid, info in active.items()
        }
        victims = []
        while sum(budgets.values()) > usable:
            eligible = [
                uid
                for uid in budgets
                if int(active[uid].get("generated_tokens", 0) or 0)
                >= int(active[uid].get("guaranteed_output_tokens", 0) or 0)
            ]
            if not eligible:
                raise RuntimeError(
                    "Elastic KV guarantees exceed usable paged capacity."
                )
            victim = max(
                eligible,
                key=lambda uid: (
                    int(active[uid].get("generated_tokens", 0) or 0)
                    - int(
                        active[uid].get("guaranteed_output_tokens", 0) or 0
                    ),
                    float(active[uid].get("queued_at", 0.0) or 0.0),
                ),
            )
            victims.append(victim)
            del budgets[victim]
        return victims

    def _finish_elastic_capacity_victims(
        self, batch_gen, active: dict
    ) -> list[int]:
        victims = self._elastic_capacity_victims(active)
        for uid in victims:
            info = active.get(uid)
            if info is None or not batch_gen.remove(uid):
                continue
            generated = int(info.get("generated_tokens", 0) or 0)
            guaranteed = int(info.get("guaranteed_output_tokens", 0) or 0)
            text = info["streamer"].finalize()
            emitted_at = self._log_decode_progress(
                uid,
                info,
                token=0,
                text=text,
                finish_reason="length",
                token_count=0,
            )
            info["rqueue"].put(
                StreamingToken(
                    text=text,
                    token=0,
                    logprobs=0.0,
                    finish_reason="length",
                    peak_memory=mx.get_peak_memory() / 1e9,
                    prompt_tps=info.get("prompt_tps"),
                    cached_tokens=info.get("cached_tokens", 0),
                    token_count=0,
                    emitted_at=emitted_at,
                    capacity_exhausted=True,
                    guaranteed_output_tokens=guaranteed,
                    elastic_output_tokens=max(0, generated - guaranteed),
                    capacity_wait_ms=float(
                        info.get("capacity_wait_ms", 0.0) or 0.0
                    ),
                )
            )
            info["rqueue"].put(None)
            del active[uid]
            logger.info(
                "Elastic KV capacity exhausted: request=%s "
                "generated_tokens=%d guaranteed_output_tokens=%d",
                info.get("request_id", uid),
                generated,
                guaranteed,
            )
        return victims

    def _partition_kv_budget_admission(
        self, pending, active, admission_capacity: Optional[int] = None
    ):
        """Admit requests that fit the active KV storage and lane budgets.

        Dense/segmented compatibility mode retains the conservative rectangular
        projection. Paged mode reserves only real per-request token budgets,
        including requested output, because rows do not inherit one another's
        maximum sequence capacity. The optional paged scheduler may bypass a
        temporarily blocked request to fill a free lane, with a bounded bypass
        count that eventually restores strict FIFO ordering.
        """
        paged = paged_turboquant_enabled()
        slot_budget = (
            get_paged_kv_capacity_tokens() if paged else get_batch_kv_slot_budget()
        )
        if slot_budget is None or not pending:
            return list(pending), []
        if paged:
            safety_tokens = get_paged_kv_safety_tokens()
            safety_tokens = (
                (safety_tokens + PAGED_TURBOQUANT_PAGE_SIZE - 1)
                // PAGED_TURBOQUANT_PAGE_SIZE
                * PAGED_TURBOQUANT_PAGE_SIZE
            )
            slot_budget = max(0, slot_budget - safety_tokens)

        context_budgets = []
        for info in active.values():
            value = self._active_context_budget(info)
            if value is None:
                logger.warning(
                    "Deferring KV-budget admission because an active request "
                    "has no context budget metadata."
                )
                return [], list(pending)
            context_budgets.append(max(0, int(value)))

        admitted = []
        deferred = []
        work_conserving = paged and paged_scheduler_enabled()
        max_bypass = get_paged_scheduler_max_bypass()
        fairness_blocked = False
        admitted_indexes = []
        for index, request in enumerate(pending):
            if admission_capacity is not None and len(admitted) >= admission_capacity:
                deferred.extend(pending[index:])
                break
            if fairness_blocked:
                deferred.append(request)
                continue
            request_budget = self._request_context_budget(request)
            if not context_budgets:
                # Let the first request reach cache allocation even if it is
                # oversized, where it receives an explicit allocation error
                # instead of waiting forever in the queue.
                admitted.append(request)
                context_budgets.append(request_budget)
                admitted_indexes.append(index)
                continue

            projected = context_budgets + [request_budget]
            projected_slots = (
                sum(
                    (
                        value + PAGED_TURBOQUANT_PAGE_SIZE - 1
                    )
                    // PAGED_TURBOQUANT_PAGE_SIZE
                    * PAGED_TURBOQUANT_PAGE_SIZE
                    for value in projected
                )
                if paged
                else len(projected) * max(projected, default=0)
            )
            if projected_slots > slot_budget:
                deferred.append(request)
                if not work_conserving:
                    deferred.extend(pending[index + 1 :])
                    break
                if int(getattr(request, "kv_bypass_count", 0)) >= max_bypass:
                    fairness_blocked = True
                continue
            admitted.append(request)
            context_budgets.append(request_budget)
            admitted_indexes.append(index)

        if work_conserving and admitted_indexes:
            last_admitted = max(admitted_indexes)
            admitted_ids = {id(request) for request in admitted}
            for index, request in enumerate(pending):
                if index >= last_admitted:
                    break
                if id(request) not in admitted_ids:
                    request.kv_bypass_count = (
                        int(getattr(request, "kv_bypass_count", 0)) + 1
                    )

        return admitted, deferred

    def _can_repromote_singleton_mtp(self, active, batch_gen, admitted) -> bool:
        """Return whether the surviving AR lane can resume singleton MTP.

        Deferred requests may remain in ``self.requests`` solely because their
        reserved KV budgets do not fit beside the active lane.  Their presence
        must not suppress MTP until the active lane finishes; only a request
        admitted in this iteration or a prompt already pending in the batch
        needs to keep the cohort in AR mode.
        """
        return (
            len(active) == 1
            and mtp_repromotion_enabled()
            and speculative_singleton_only()
            and not batch_gen.has_pending_prompts
            and not admitted
        )

    def _run(self):
        try:
            self._run_impl()
        finally:
            try:
                # The worker can outlive stop_and_join's timeout. Retire its
                # disk writer only after its final request/store has finished.
                manager = getattr(self, "apc_manager", None)
                if manager is not None:
                    manager.close()
            except Exception:
                logger.exception("Error closing APC after generation worker shutdown")
            paged_registry = getattr(self, "_paged_registry", None)
            release_registry = getattr(paged_registry, "release", None)
            if callable(release_registry):
                final_stats = release_registry()
                logger.info(
                    "Persistent Paged TurboQuant pool closed at server shutdown: "
                    "used_pages=%s high_water_pages=%s capacity_pages=%s "
                    "pool_bytes=%s",
                    getattr(final_stats, "used_layer_pages", None),
                    getattr(final_stats, "high_water_layer_pages", None),
                    getattr(final_stats, "capacity_layer_pages", None),
                    getattr(final_stats, "pool_nbytes", None),
                )
                self._paged_registry = None
            clear_mlx_streams()

    def _run_impl(self):
        """Single GPU thread: owns BatchGenerator, runs tight next() loop."""
        try:
            self._initialize_model()
        except Exception as e:
            self._load_error = e
            self._ready.set()
            logger.exception("Error loading model in generation thread: %s", e)
            return

        self._ready.set()

        # Diffusion models cannot run through the AR batch generator.
        if is_diffusion_model(self.model):
            self._run_diffusion()
            return

        generation_stream = mx.default_stream(mx.default_device())

        batch_gen = None
        # uid -> {rqueue, tokens, gen_kwargs}
        active: dict = {}
        last_paged_active_count = None
        last_phase_deferral_signature = None
        last_kv_deferral_signature = None
        max_num_seqs = get_max_num_seqs()
        phase_cohorts = self._uses_phase_cohort_admission()

        while not (self._stop and not active and self.requests.empty()):
            new_items = []
            try:
                # Poll the request queue — non-blocking when generating, short
                # blocking wait when idle so we don't spin.
                active_batch = bool(active)
                coalesce_s = (
                    get_speculative_batch_coalesce_s()
                    if not active_batch
                    and (self.draft_model is not None or phase_cohorts)
                    else 0.0
                )
                capacity = self._admission_capacity(
                    active_count=len(active), max_num_seqs=max_num_seqs
                )
                collection_capacity = capacity
                if paged_scheduler_enabled() and (
                    capacity is None or capacity > 0
                ):
                    collection_capacity = get_paged_scheduler_scan_limit()
                if (
                    active_batch
                    and phase_cohorts
                    and (capacity is None or capacity > 0)
                ):
                    new_items, should_stop = self._collect_active_phase_text_requests(
                        collection_capacity
                    )
                else:
                    new_items, should_stop = self._collect_pending_requests(
                        active=active_batch,
                        capacity=collection_capacity,
                        coalesce_s=coalesce_s,
                    )
                live_items = []
                for item in new_items:
                    if self._request_cancelled_before_admission(item):
                        logger.info(
                            "Dropped cancelled queued request before GPU "
                            "admission: request=%s",
                            self._request_log_id(item),
                        )
                    else:
                        live_items.append(item)
                new_items = live_items
                collected_items = list(new_items)
                new_items, phase_deferred_items = (
                    self._partition_lm_head_phase_admission(new_items, active)
                )
                if phase_deferred_items:
                    phase_deferral_signature = (
                        len(active),
                        tuple(
                            self._request_log_id(item)
                            for item in phase_deferred_items
                        ),
                    )
                    if phase_deferral_signature != last_phase_deferral_signature:
                        logger.info(
                            "LM-head phase admission deferred: active=%d "
                            "admitted=%d deferred=%d mixed_prefill_limit_tokens=%d",
                            len(active),
                            len(new_items),
                            len(phase_deferred_items),
                            get_lm_head_mixed_prefill_max_tokens(),
                        )
                        last_phase_deferral_signature = phase_deferral_signature
                else:
                    last_phase_deferral_signature = None
                new_items, deferred_items = self._partition_kv_budget_admission(
                    new_items, active, admission_capacity=capacity
                )
                if deferred_items:
                    deferral_signature = (
                        len(active),
                        tuple(self._request_log_id(item) for item in deferred_items),
                    )
                    if deferral_signature != last_kv_deferral_signature:
                        logger.info(
                            "KV admission deferred: active=%d admitted=%d "
                            "deferred=%d capacity_tokens=%s",
                            len(active),
                            len(new_items),
                            len(deferred_items),
                            get_paged_kv_capacity_tokens()
                            if paged_turboquant_enabled()
                            else get_batch_kv_slot_budget(),
                        )
                        last_kv_deferral_signature = deferral_signature
                else:
                    last_kv_deferral_signature = None
                deferred_ids = {
                    id(item) for item in phase_deferred_items + deferred_items
                }
                for item in collected_items:
                    if id(item) in deferred_ids:
                        self.requests.put(item)
                if should_stop and not active:
                    break

                if (
                    new_items
                    and active
                    and batch_gen is not None
                    and speculative_singleton_only()
                    and batch_gen.demote_mtp_to_ar()
                ):
                    gc.collect()
                    mx.clear_cache()
                    logger.info(
                        "Demoted active MTP cohort to AR before batch admission."
                    )

                # Drop abandoned requests before doing more work.
                cancelled = self._drain_cancellations()
                cancelled.update(
                    uid
                    for uid, info in active.items()
                    if info.get("cancel_event") is not None
                    and info["cancel_event"].is_set()
                )
                if cancelled and batch_gen is not None:
                    for uid in cancelled:
                        if uid in active:
                            batch_gen.remove(uid)
                            info = active.pop(uid)
                            logger.info(
                                "Generation cancelled: request=%s generated_tokens=%d",
                                info.get("request_id", uid),
                                int(info.get("generated_tokens", 0) or 0),
                            )
                            try:
                                info["rqueue"].put(None)
                            except Exception:
                                pass
                    if not active and not batch_gen.has_work:
                        batch_gen.close()
                        batch_gen = None
                        self._unload_deferred_drafter()

                if new_items and batch_gen is not None and not active:
                    if not batch_gen.has_work:
                        batch_gen.close()
                        batch_gen = None
                        self._unload_deferred_drafter()

                for request in new_items:
                    if self._request_cancelled_before_admission(request):
                        logger.info(
                            "Dropped cancelled queued request before embedding: "
                            "request=%s",
                            self._request_log_id(request),
                        )
                        continue
                    rqueue = request.rqueue
                    raw_inputs = request.raw_inputs
                    prompt_tokens = request.prompt_tokens
                    args = request.args
                    images = request.images
                    log_state = self._log_prefill_started(
                        request, backend="continuous_batching"
                    )
                    if batch_gen is None:
                        paged_registry = getattr(self, "_paged_registry", None)
                        if paged_turboquant_enabled():
                            if paged_registry is None:
                                paged_registry = make_paged_turboquant_registry(
                                    self.model.language_model
                                )
                                self._paged_registry = paged_registry
                                logger.info(
                                    "Paged TurboQuant enabled: capacity_tokens=%d "
                                    "pageable_layers=%d max_num_seqs=%d; APC=%s; "
                                    "singleton-only MTP=%s",
                                    get_paged_kv_capacity_tokens(),
                                    len(paged_registry.leaf_keys),
                                    get_max_num_seqs(),
                                    self.apc_manager is not None,
                                    bool(
                                        self.draft_model is not None
                                        and self.draft_kind == "mtp"
                                        and speculative_singleton_only()
                                    ),
                                )
                            else:
                                pool_stats = paged_registry.stats()
                                if pool_stats.used_layer_pages:
                                    raise RuntimeError(
                                        "Cannot reuse a paged TurboQuant pool with "
                                        f"{pool_stats.used_layer_pages} live pages."
                                    )
                                logger.info(
                                    "Reusing persistent Paged TurboQuant pool: "
                                    "capacity_tokens=%d high_water_pages=%d",
                                    get_paged_kv_capacity_tokens(),
                                    pool_stats.high_water_layer_pages,
                                )
                        batch_gen = BatchGenerator(
                            self.model.language_model,
                            self.processor,
                            stop_tokens=self.stop_tokens,
                            sampler=self._make_sampler(args),
                            kv_bits=self.kv_bits,
                            kv_key_bits=getattr(self, "kv_key_bits", None),
                            kv_value_bits=getattr(self, "kv_value_bits", None),
                            kv_key_scheme=getattr(self, "kv_key_scheme", None),
                            kv_value_scheme=getattr(self, "kv_value_scheme", None),
                            kv_group_size=self.kv_group_size,
                            kv_quant_scheme=self.kv_quant_scheme,
                            quantized_kv_start=self.quantized_kv_start,
                            compute_logprobs=bool(args.logprobs),
                            top_logprobs_k=self.top_logprobs_k if args.logprobs else 0,
                            stream=generation_stream,
                            apc_manager=self.apc_manager,
                            draft_model=self.draft_model,
                            draft_kind=self.draft_kind,
                            draft_block_size=_get_draft_block_size_from_env(),
                            greedy_sampling=args.temperature == 0,
                            prefill_step_size=self._effective_prefill_step_size(),
                            prefill_batch_size=(
                                1
                                if paged_registry is not None
                                else DEFAULT_PREFILL_BATCH_SIZE
                            ),
                            completion_batch_size=(
                                get_max_num_seqs()
                                if paged_registry is not None
                                else DEFAULT_COMPLETION_BATCH_SIZE
                            ),
                            paged_cache_factory=paged_registry,
                            owns_paged_cache_factory=False,
                        )

                    # Vision encoder runs on the GPU thread; text tokenization
                    # already happened on the caller thread.
                    if self.apc_manager is not None:
                        self.apc_manager.prepare_prefill(prompt_tokens)
                    input_ids, gen_kwargs = self._gpu_embed(
                        raw_inputs,
                        images,
                        apc_semantic_hash=request.apc_semantic_hash,
                    )
                    has_embeds = bool(
                        gen_kwargs.get("inputs_embeds") is not None
                        or gen_kwargs.get("input_embedding_provider") is not None
                    )
                    # Preserve tenant isolation for manually queued requests
                    # that predate server-side semantic hash computation.
                    if request.apc_semantic_hash is None and getattr(
                        args, "tenant_id", None
                    ):
                        gen_kwargs["_apc_tenant"] = args.tenant_id

                    # Drain pending text-only prompts before inserting an
                    # embed-bearing request — multi-row PromptProcessingBatch
                    # admission expects all rows to carry inputs_embeds (the
                    # mixed APC path concatenates them per-row).
                    if has_embeds and any(
                        not (
                            s[3]
                            and (
                                s[3].get("inputs_embeds") is not None
                                or s[3].get("input_embedding_provider") is not None
                            )
                        )
                        for s in batch_gen.unprocessed_prompts
                    ):
                        self._flush(batch_gen, active)

                    try:
                        thinking_budget_criteria = request.thinking_budget_criteria
                        (uid,) = batch_gen.insert(
                            [input_ids.squeeze(0).tolist()],
                            max_tokens=args.max_tokens,
                            prompt_kwargs=[gen_kwargs],
                            logits_processors=[
                                self._make_logits_processors(args, input_ids)
                            ],
                            thinking_budget_criteria=[thinking_budget_criteria],
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    active[uid] = {
                        "rqueue": rqueue,
                        "streamer": _ServerTokenStreamer(
                            self.tokenizer,
                            make_streaming_detokenizer(self.processor),
                        ),
                        "gen_kwargs": gen_kwargs if has_embeds else None,
                        "prompt_tps": None,
                        "cached_tokens": 0,
                        "cancel_event": request.cancel_event,
                        "queued_at": request.queued_at,
                        "capacity_wait_ms": max(
                            0.0,
                            (time.perf_counter() - request.queued_at) * 1000.0,
                        ),
                        "prompt_tokens": max(0, int(prompt_tokens)),
                        "requested_output_tokens": max(
                            0, int(args.max_tokens or 0)
                        ),
                        "guaranteed_output_tokens": min(
                            max(0, int(args.max_tokens or 0)),
                            get_paged_output_guarantee_tokens()
                            if paged_turboquant_enabled()
                            and get_paged_output_guarantee_tokens() is not None
                            else max(0, int(args.max_tokens or 0)),
                        ),
                        "context_budget_tokens": self._request_context_budget(
                            request
                        ),
                        "spec_snapshot": (
                            speculative_stats_snapshot(self.draft_model)
                            if self.draft_model is not None
                            else None
                        ),
                        **log_state,
                    }

                if not active or batch_gen is None:
                    continue

                self._finish_elastic_capacity_victims(batch_gen, active)
                if not active:
                    continue
                self._step(batch_gen, active)
                previous_paged_active_count = last_paged_active_count
                if len(active) != previous_paged_active_count:
                    pool_stats_fn = getattr(batch_gen, "paged_pool_stats", None)
                    pool_stats = pool_stats_fn() if callable(pool_stats_fn) else None
                    if pool_stats is not None:
                        logger.info(
                            "Paged TurboQuant pool state: active=%d used_pages=%d "
                            "free_pages=%d high_water_pages=%d capacity_pages=%d",
                            len(active),
                            pool_stats.used_layer_pages,
                            pool_stats.free_layer_pages,
                            pool_stats.high_water_layer_pages,
                            pool_stats.capacity_layer_pages,
                        )
                    last_paged_active_count = len(active)
                if self._can_repromote_singleton_mtp(active, batch_gen, new_items):
                    # A completed peer can leave B>1 attention temporaries in
                    # MLX's allocator cache. Drop those before reloading the
                    # singleton MTP drafter or their footprints overlap.
                    if (
                        previous_paged_active_count is not None
                        and previous_paged_active_count > 1
                    ):
                        gc.collect()
                        mx.clear_cache()
                    if batch_gen.promote_ar_singleton_to_mtp():
                        logger.info(
                            "Re-promoted surviving AR singleton to MTP without "
                            "KV migration."
                        )
                if (
                    not active
                    and batch_gen is not None
                    and not getattr(batch_gen, "has_work", True)
                ):
                    batch_gen.close()
                    batch_gen = None
                    self._unload_deferred_drafter()

            except Exception as e:
                logger.exception("Error in generation thread")
                error_queues = {
                    id(info["rqueue"]): info["rqueue"] for info in active.values()
                }
                error_queues.update(
                    {id(request.rqueue): request.rqueue for request in new_items}
                )
                _notify_queues(error_queues.values(), e, None)
                active.clear()
                close_batch = getattr(batch_gen, "close", None)
                if callable(close_batch):
                    close_batch()
                batch_gen = None
                self._unload_deferred_drafter()
                mx.clear_cache()
                gc.collect()

        if batch_gen is not None and callable(getattr(batch_gen, "close", None)):
            batch_gen.close()
        self._unload_deferred_drafter()

    def _run_diffusion(self):
        """GPU thread loop for diffusion models.

        Diffusion generation runs one request at a time (batch size 1).
        Output streams back block-by-block: one queue item per denoised
        block, plus a final item carrying the finish reason. Non-streaming
        endpoints aggregate the same items into a single response.
        """
        uid_counter = 0
        cancelled: set = set()
        while not self._stop:
            try:
                new_items, should_stop = self._collect_pending_requests(active=False)
                if should_stop:
                    break
                cancelled |= self._drain_cancellations()
                for request in new_items:
                    rqueue = request.rqueue
                    raw_inputs = request.raw_inputs
                    prompt_tokens = request.prompt_tokens
                    args = request.args
                    log_state = self._log_prefill_started(request, backend="diffusion")
                    uid_counter += 1
                    uid = uid_counter
                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    try:
                        self._generate_diffusion(
                            uid,
                            rqueue,
                            raw_inputs,
                            args,
                            cancelled,
                            log_state,
                            apc_semantic_hash=request.apc_semantic_hash,
                        )
                        rqueue.put(None)
                    except Exception as e:
                        logger.exception("Error in diffusion generation")
                        try:
                            rqueue.put(e)
                            rqueue.put(None)
                        except Exception:
                            pass
                    mx.clear_cache()
            except Exception:
                logger.exception("Error in diffusion generation thread")
                mx.clear_cache()
                gc.collect()

    def _generate_diffusion(
        self,
        uid,
        rqueue,
        raw_inputs,
        args,
        cancelled,
        log_state=None,
        apc_semantic_hash=None,
    ):
        log_state = log_state or {"request_id": uid, "generated_tokens": 0}
        input_ids = raw_inputs.get("input_ids")
        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids[None]
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "stopping_criteria"):
            tokenizer.stopping_criteria.reset(self.config.eos_token_id)
        skip_special_token_ids = (
            set(getattr(tokenizer, "all_special_ids", None) or [])
            if args.skip_special_tokens
            else set()
        )

        stream_kwargs = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "mm_token_type_ids": raw_inputs.get("mm_token_type_ids"),
        }
        prefill_step_size = self._effective_prefill_step_size()
        if prefill_step_size > 0:
            stream_kwargs["prefill_step_size"] = prefill_step_size
        if args.seed is not None:
            stream_kwargs["seed"] = args.seed
        if args.logits_processors is not None:
            stream_kwargs["logits_processors"] = args.logits_processors
        stream_kwargs.update(args.diffusion_kwargs())
        if self.apc_manager is not None and self.apc_mode is not None:
            stream_kwargs["_apc_manager"] = self.apc_manager
            stream_kwargs["_apc_semantic_hash"] = int(apc_semantic_hash or 0)

        emitter = _DiffusionBlockEmitter()
        prefill_logged = False

        def on_result(result):
            nonlocal prefill_logged
            if not prefill_logged and getattr(result, "prompt_tps", None) is not None:
                prompt_tps = float(result.prompt_tps or 0.0)
                prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
                logger.info(
                    "Prefill completed: request=%s prompt_tokens=%d cached_tokens=%d "
                    "elapsed=%.3fs rate=%.1f tok/s",
                    log_state.get("request_id", uid),
                    prompt_tokens,
                    int(getattr(result, "cached_tokens", 0) or 0),
                    prompt_tokens / prompt_tps if prompt_tps > 0 else 0.0,
                    prompt_tps,
                )
                prefill_logged = True
            for chunk in emitter.feed(result):
                chunk.emitted_at = self._log_decode_progress(
                    uid,
                    log_state,
                    token=chunk.token,
                    text=chunk.text,
                    finish_reason=chunk.finish_reason,
                    token_count=chunk.token_count,
                )
                rqueue.put(chunk)
                if chunk.finish_reason:
                    return False
            cancelled.update(self._drain_cancellations())
            if uid in cancelled:
                cancelled.discard(uid)
                return False
            return True

        results = stream_diffusion_generate_from_kwargs(
            self.model,
            self.processor,
            tokenizer,
            input_ids,
            raw_inputs.get("pixel_values"),
            raw_inputs.get("attention_mask"),
            skip_special_token_ids,
            stream_kwargs,
            skip_special_tokens=args.skip_special_tokens,
            on_result=on_result,
        )
        try:
            for _ in results:
                pass
        finally:
            results.close()

    def _step(self, batch_gen, active, gen_kwargs=None):
        """One batch generation step: prefill + decode."""
        kwargs = gen_kwargs or {}
        prompt_responses, responses = batch_gen.next(**kwargs)
        # A request can wait behind another row's long chunked prefill after it
        # has already received its GenerationContext. Keep only those silent
        # consumers alive; a real response already resets its inactivity
        # timeout and must remain the next observable queue item.
        responding_uids = {response.uid for response in responses}
        for uid, info in active.items():
            rqueue = info.get("rqueue")
            if uid not in responding_uids and rqueue is not None:
                rqueue.put(_TOKEN_QUEUE_ACTIVITY)
        self._log_prefill_progress(batch_gen, active)
        released_prefill_inputs = False
        for prompt_response in prompt_responses:
            if prompt_response.uid in active:
                info = active[prompt_response.uid]
                info["prompt_tps"] = prompt_response.prompt_tps
                info["cached_tokens"] = getattr(prompt_response, "cached_tokens", 0)
                if (
                    getattr(self, "defer_draft_model", False)
                    and info.get("gen_kwargs") is not None
                ):
                    info["gen_kwargs"] = None
                    released_prefill_inputs = True
                self._log_prefill_completed(prompt_response.uid, info, prompt_response)
        if released_prefill_inputs:
            mx.clear_cache()
        if not responses:
            return

        for r in responses:
            if r.uid not in active:
                continue

            info = active[r.uid]
            rqueue = info["rqueue"]

            tok = r.token
            token_count = 0 if tok is None else 1
            if tok is None:
                text = info["streamer"].finalize()
                tok = 0
            elif hasattr(tok, "item"):
                tok = tok.item()
                text = self._stream_text(info, tok, r.finish_reason)
            else:
                text = self._stream_text(info, tok, r.finish_reason)

            lp = r.token_logprob

            emitted_at = self._log_decode_progress(
                r.uid,
                info,
                token=tok,
                text=text,
                finish_reason=r.finish_reason,
                token_count=token_count,
            )

            draft_rounds = draft_accepted = draft_total = None
            request_draft_kind = None
            if r.finish_reason is not None and info.get("spec_snapshot") is not None:
                draft_rounds, draft_accepted, draft_total = speculative_stats_since(
                    self.draft_model, info["spec_snapshot"]
                )
                if draft_rounds is not None:
                    request_draft_kind = self.draft_kind

            rqueue.put(
                StreamingToken(
                    text=text,
                    token=tok,
                    logprobs=lp,
                    finish_reason=r.finish_reason,
                    peak_memory=mx.get_peak_memory() / 1e9 if r.finish_reason else 0,
                    prompt_tps=info.get("prompt_tps"),
                    draft_kind=request_draft_kind,
                    draft_rounds=draft_rounds,
                    draft_n_accepted=draft_accepted,
                    draft_n=draft_total,
                    top_logprobs=getattr(r, "top_logprobs", None),
                    cached_tokens=info.get("cached_tokens", 0),
                    token_count=token_count,
                    emitted_at=emitted_at,
                    capacity_exhausted=False,
                    guaranteed_output_tokens=info.get(
                        "guaranteed_output_tokens"
                    ),
                    elastic_output_tokens=max(
                        0,
                        int(info.get("generated_tokens", 0) or 0)
                        - int(
                            info.get("guaranteed_output_tokens", 0) or 0
                        ),
                    ),
                    capacity_wait_ms=float(
                        info.get("capacity_wait_ms", 0.0) or 0.0
                    ),
                )
            )

            if r.finish_reason is not None:
                rqueue.put(None)
                del active[r.uid]

    def _stream_text(self, info: dict, token: int, finish_reason: Optional[str]) -> str:
        """Convert one generated token into a streaming text segment."""
        return info["streamer"].advance(token, finish_reason)

    def _flush(self, batch_gen, active):
        """Drain all pending text-only prompts before inserting an image request."""
        while batch_gen.has_pending_prompts:
            self._step(batch_gen, active)

    def validate_context_budget(
        self,
        prompt: str,
        images: Optional[List] = None,
        audio: Optional[List] = None,
        args: Optional[GenerationArguments] = None,
        videos: Optional[List] = None,
    ):
        """Validate request size before opening a streaming response."""
        if get_configured_context_limit() is None and not paged_turboquant_enabled():
            return
        self.wait_until_ready()
        args = args or GenerationArguments(max_tokens=get_server_max_tokens())
        raw_inputs = self._preprocess_request(prompt, images, audio, videos)
        _check_configured_context_budget(
            _count_prompt_tokens(raw_inputs), args.max_tokens
        )
