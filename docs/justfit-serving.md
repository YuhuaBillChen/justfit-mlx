# Serving JustFit

> New to local model serving? Follow the complete
> [beginner Quick Start](../examples/justfit/quickstart.md) first. This page is
> the shorter serving reference.

This guide turns the JustFit reproduction kit into a local OpenAI-compatible
server for three common clients:

- Open WebUI for browser chat;
- the included terminal chat client;
- Hermes Agent for coding and tool use.

The recommended Hermes profile reserves **65,536 input positions plus 8,192
generated positions**, for a total KV capacity of **73,728 positions**. This is
a single-lane service profile, not the 192K+16K capacity experiment and not a
four-lane aggregate budget.

## 1. Install and download

Follow the pinned source and artifact steps in the
[reproduction guide](../examples/justfit/README.md), then use a short, stable
local model directory name. The directory name is exposed as the model ID by
the OpenAI-compatible endpoint.

```bash
git clone https://github.com/YuhuaBillChen/mlx-vlm.git
cd mlx-vlm
git checkout production/qwen-paged-continuous-batching

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -U huggingface_hub

hf download mlx-community/Qwen3.8-27B-mxfp4 \
  --revision 97ab0819817ab1c61d7d39f9169fc71999915641 \
  --local-dir ./justfit-qwen38
hf download billchen42/JustFit-Qwen3.8-27B-components \
  --revision 13e0462fa911a1bcbdccbaa759120500c0f90582 \
  --local-dir ./justfit-components

python examples/justfit/prepare_components.py \
  --model ./justfit-qwen38 \
  --output ./justfit-extracted \
  --head-only
```

## 2. Start the 64K-input + 8K-output profile

Generate a local secret and keep it out of shell history, screenshots and
committed configuration files:

```bash
export JUSTFIT_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

MODEL_PATH=justfit-qwen38 \
MTP_PATH="$PWD/justfit-components/mtp" \
VISION_PATH="$PWD/justfit-components/vision-bf16.safetensors" \
LM_HEAD_PATH="$PWD/justfit-extracted/language-head.safetensors" \
HOST=127.0.0.1 PORT=8080 API_KEY="$JUSTFIT_API_KEY" \
LANES=1 KV_CAPACITY=73728 MAX_TOKENS=8192 OUTPUT_GUARANTEE=8192 \
TOKEN_QUEUE_TIMEOUT=1800 \
examples/justfit/run_server.sh
```

The server URL is `http://127.0.0.1:8080/v1`, and its model ID is
`justfit-qwen38`. Keep `CAPACITY_MODE=0` for real use so EOS remains enabled.
Long cold prefills can legitimately take several minutes; configure clients
and reverse proxies with at least a 1,800-second request timeout.

The launcher refuses an unauthenticated non-loopback bind. Do not expose this
research server directly to the public Internet; place TLS and normal access
controls in front of it if remote access is required.

## 3. Terminal chat

In another terminal, activate the same environment and reuse the key:

```bash
export JUSTFIT_API_KEY='the same value used by the server'
python examples/justfit/chat_cli.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model justfit-qwen38 \
  --max-tokens 8192
```

The client streams Chat Completions, retains conversation history, and accepts
`/reset` and `/exit`.

## 4. Open WebUI

When Open WebUI runs in Docker on the same Mac, restart the JustFit server with
`HOST=0.0.0.0` and the same non-empty `API_KEY`. In Open WebUI, add an
OpenAI-compatible connection under **Admin Settings → Connections**:

| Setting | Value |
| --- | --- |
| Base URL | `http://host.docker.internal:8080/v1` |
| API key | the value of `JUSTFIT_API_KEY` |
| Model | `justfit-qwen38` |

For Open WebUI running directly on the host, use
`http://127.0.0.1:8080/v1` and keep the server bound to loopback. The required
backend routes are `/v1/models` and `/v1/chat/completions`; this guide does not
claim support for OpenAI Responses API-only features.

## 5. Hermes Agent

Hermes needs at least a 64K context for its coding-agent workflow. Here the
configured context is 73,728 positions: 65,536 for input and an 8,192-token
generation ceiling.

Run `hermes model`, choose **Custom endpoint**, and enter:

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8080/v1` |
| API key | the value of `JUSTFIT_API_KEY` |
| Model | `justfit-qwen38` |
| Context length | `73728` |

The equivalent Hermes configuration is:

```yaml
model:
  default: justfit-qwen38
  provider: custom
  base_url: http://127.0.0.1:8080/v1
  api_key: YOUR_LOCAL_JUSTFIT_KEY
  context_length: 73728
```

Prefer the interactive configuration flow so the key does not end up in copied
examples or version control. If Hermes is on another trusted machine, use the
Mac's reachable address instead of `127.0.0.1`, bind JustFit to `0.0.0.0`, and
retain API-key authentication.

## 6. Verify before a long coding session

```bash
curl -fsS \
  -H "Authorization: Bearer $JUSTFIT_API_KEY" \
  http://127.0.0.1:8080/v1/models

curl -N \
  -H "Authorization: Bearer $JUSTFIT_API_KEY" \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/chat/completions \
  -d '{"model":"justfit-qwen38","messages":[{"role":"user","content":"Reply with READY."}],"max_tokens":32,"stream":true}'
```

A healthy stream ends with `data: [DONE]`. Authentication failures should
return HTTP 401. A 73,728-position pool guarantees addressable KV capacity for
one admitted request under this profile; it does not promise that every prompt
template can supply exactly 65,536 user-visible tokens, because templates and
multimodal inputs also consume positions.

## Validation record

The `justfit-repro-v3` flow was checked on the paper's M4 Pro / 24 GiB host on
2026-09-16 with the production stack stopped so that only one 27B server owned
unified memory:

- unauthenticated `/v1/models` returned HTTP 401; authenticated discovery
  returned the stable `justfit-qwen38` model ID;
- the terminal client completed a streaming request and observed `[DONE]`;
- a request sent from an existing Open WebUI container through its Docker
  network path returned `OWUI_READY`;
- Hermes Agent used `context_length: 73728`, constructed a real 14,944-token
  agent prompt, and returned `HERMES_READY`;
- the Hermes request prefilling rate was 123.8 tok/s, and the guarded server's
  sampled process-footprint peak across the validation was 16,520 MiB.

This is a connectivity and lifecycle smoke test, not a matched throughput
benchmark or a full 64K-prompt coding evaluation. Open WebUI configuration
follows its official
[OpenAI-compatible provider guide](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/),
and Hermes setup follows its official
[provider documentation](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/integrations/providers.md).
