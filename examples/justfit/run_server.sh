#!/usr/bin/env bash
set -euo pipefail

required=(MODEL_PATH MTP_PATH LM_HEAD_PATH VISION_PATH)
for name in "${required[@]}"; do
  value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

for path in "$MODEL_PATH/config.json" "$MTP_PATH/config.json" "$LM_HEAD_PATH" "$VISION_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "required path does not exist: $path" >&2
    exit 2
  fi
done

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
LANES="${LANES:-1}"
KV_CAPACITY="${KV_CAPACITY:-32768}"
MAX_TOKENS="${MAX_TOKENS:-512}"
OUTPUT_GUARANTEE="${OUTPUT_GUARANTEE:-8192}"
CAPACITY_MODE="${CAPACITY_MODE:-0}"

for pair in "LANES:$LANES" "KV_CAPACITY:$KV_CAPACITY" "MAX_TOKENS:$MAX_TOKENS" "OUTPUT_GUARANTEE:$OUTPUT_GUARANTEE"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "$name must be a positive integer: $value" >&2
    exit 2
  fi
done
if (( KV_CAPACITY % 256 != 0 )); then
  echo "KV_CAPACITY must be a multiple of the 256-token page size" >&2
  exit 2
fi

export MLX_VLM_PAGED_TQ=1
export MLX_VLM_PAGED_KV_CAPACITY_TOKENS="$KV_CAPACITY"
export MLX_VLM_PAGED_SCHEDULER=1
export MLX_VLM_PAGED_SCHEDULER_SCAN_LIMIT="${SCHEDULER_SCAN_LIMIT:-32}"
export MLX_VLM_PAGED_SCHEDULER_MAX_BYPASS="${SCHEDULER_MAX_BYPASS:-8}"
export MLX_VLM_PAGED_OUTPUT_GUARANTEE_TOKENS="$OUTPUT_GUARANTEE"
export MLX_VLM_PAGED_KV_SAFETY_TOKENS="${KV_SAFETY_TOKENS:-0}"
export MLX_VLM_PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-4}"
export MLX_VLM_MIXED_PREFILL_STEP_SIZE="${MIXED_PREFILL_STEP_SIZE:-64}"
export MLX_VLM_PAGED_PREFILL_EAGER_RELEASE=1
export MLX_VLM_PAGED_PREFILL_IMPL=direct_inverse
export MLX_VLM_LM_HEAD_MIXED_PREFILL_MAX_TOKENS="${LM_HEAD_MIXED_PREFILL_MAX_TOKENS:-8192}"
export MLX_VLM_QUANTIZE_LAST_KV_LAYER=1
export MLX_VLM_SPECULATIVE_SINGLETON_ONLY=1
export MLX_VLM_MTP_REPROMOTE=1
export MLX_VLM_TQ_MTP_QTILE=1
export MLX_VLM_TQ_RESERVE_DECODE=1
export MLX_VLM_TQ_LAZY_VERIFY_APPEND=1
export MLX_VLM_TQ_FUSED_DEQUANT=1
export MLX_VLM_CHUNK_LOCAL_INPUT_EMBEDS=1
export MLX_VLM_LANGUAGE_HEAD_PHASE_SWAP_PATH="$LM_HEAD_PATH"
export MLX_VLM_CAPACITY_IGNORE_EOS="$CAPACITY_MODE"
export MLX_VLM_TOKEN_QUEUE_TIMEOUT="${TOKEN_QUEUE_TIMEOUT:-0}"

exec python -m mlx_vlm.server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --draft-model "$MTP_PATH" \
  --draft-kind mtp \
  --draft-block-size "${DRAFT_BLOCK_SIZE:-3}" \
  --defer-draft-model \
  --vision-phase-swap-path "$VISION_PATH" \
  --chunk-local-input-embeddings \
  --max-num-seqs "$LANES" \
  --max-tokens "$MAX_TOKENS" \
  --max-kv-size "$KV_CAPACITY" \
  --kv-bits 4 \
  --kv-quant-scheme turboquant \
  --quantized-kv-start 0 \
  --prefill-step-size "${PREFILL_STEP_SIZE:-256}" \
  --log-progress-interval "${LOG_PROGRESS_INTERVAL:-256}"
