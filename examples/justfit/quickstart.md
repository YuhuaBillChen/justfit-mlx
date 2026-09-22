# JustFit Quick Start

This guide is for someone who wants to **run Qwen3.8-27B locally on a Mac** and
use it from a terminal, Open WebUI, or Hermes Agent. You do not need to
understand the JustFit paper or its benchmark scripts first.

The setup below gives one request a **65,536-token input budget** and an
**8,192-token output ceiling** (73,728 total positions). Keep this page open and
copy each command in order.

> **Use the release tag shown here.** This guide intentionally stays on the
> immutable `justfit-repro-v4` release while using a conservative 64K+8K daily
> profile. The same release contains larger research configurations, but a
> measured capacity boundary is not automatically a beginner default.

> **Time and space:** allow roughly 30–60 minutes for the first installation,
> depending on download speed, and keep at least 30 GB of disk space free.
> The server takes roughly 20 seconds to load after the files are downloaded.

## What you need

- An Apple-silicon Mac with an M3-or-newer chip (M3/M4/M5 family). The fused
  MTP qtile kernel needs a 1024-thread threadgroup that M1 and M2 reject at the
  27B head dimension, so this profile does not run on those chips.
- **24 GB unified memory or more.** The published setup was tested on an
  M4 Pro MacBook Pro with 24 GB. A 16 GB Mac is not a supported target for
  this 27B profile.
- macOS Terminal. Open it from **Applications → Utilities → Terminal**.
- A reliable Internet connection for the initial model download.
- At least 30 GB of free disk space.

Open Terminal and check the machine and disk before continuing:

```bash
uname -m
df -h "$HOME"
```

The first command should print `arm64`.

## 1. Install native Git and Python

First install Apple's command-line tools:

```bash
xcode-select --install
```

If macOS says they are already installed, continue. Confirm that this is an
Apple-silicon shell:

```bash
test "$(uname -m)" = "arm64" || {
  echo "Open a native arm64 Terminal, not an Intel/Rosetta shell."
  exit 1
}
git --version
```

MLX requires an arm64 Python. An older Intel Homebrew installation under
`/usr/local` can silently provide an x86_64 `python3`, for which no MLX wheel
exists. Install the native Apple-silicon edition of
[Homebrew](https://brew.sh/) if `/opt/homebrew/bin/brew` is missing:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install and verify the exact native Python used by this guide:

```bash
/opt/homebrew/bin/brew install python@3.12 git
/opt/homebrew/bin/python3.12 --version
file /opt/homebrew/bin/python3.12
```

The last line must contain `arm64`. Do not continue if it says `x86_64`.

### Raise the Metal wired-memory ceiling

Having 24 GB of unified memory does **not** mean macOS will let Metal wire
21.5 GiB by default. The published M4 Pro configuration used
`iogpu.wired_limit_mb=22016`; without it, the model can hit the lower system
ceiling even though Activity Monitor still shows free unified memory.

Check the current ceiling and raise it only when it is lower than the tested
value:

```bash
current_wired_mb="$(/usr/sbin/sysctl -n iogpu.wired_limit_mb)"
echo "Current Metal wired-memory ceiling: ${current_wired_mb} MiB"

if [ "$current_wired_mb" -lt 22016 ]; then
  sudo /usr/sbin/sysctl -w iogpu.wired_limit_mb=22016
fi

test "$(/usr/sbin/sysctl -n iogpu.wired_limit_mb)" -ge 22016 || {
  echo "Metal wired-memory ceiling is still below 22016 MiB."
  exit 1
}
```

This raises a ceiling; it does not reserve 21.5 GiB immediately. It also
reduces the memory left for macOS and other GPU applications, so run only one
27B server and close other large GPU workloads. The setting resets after a
reboot; repeat this check before restarting JustFit.

The paper's boundary experiments additionally used a separate 21,000 MiB
process-footprint guard. This beginner launcher does not silently install that
machine-specific guard. The tested 64K-input + 8K-output serving smoke peaked
at 16,520 MiB, well below that boundary; the fully cold 240K+16K experiment
remains a separate, explicitly guarded protocol.

## 2. Download the JustFit source

The tag in these commands is immutable, so everyone following this guide gets
the same source:

```bash
cd "$HOME"
git clone https://github.com/YuhuaBillChen/justfit-mlx.git
cd justfit-mlx
git checkout justfit-repro-v4
```

If you cloned the repository previously, update it instead:

```bash
cd "$HOME/justfit-mlx"
git fetch --tags origin
git checkout justfit-repro-v4
```

## 3. Create an isolated Python environment

This keeps JustFit's Python packages separate from the rest of your Mac:

```bash
cd "$HOME/justfit-mlx"
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -U huggingface_hub
```

Your Terminal prompt should now begin with `(.venv)`. Whenever you open a new
Terminal for JustFit, run these two commands again:

```bash
cd "$HOME/justfit-mlx"
source .venv/bin/activate
```

## 4. Download the model and JustFit components

These are public downloads; a Hugging Face login is not required:

```bash
cd "$HOME/justfit-mlx"

hf download mlx-community/Qwen3.8-27B-mxfp4 \
  --revision 97ab0819817ab1c61d7d39f9169fc71999915641 \
  --local-dir ./justfit-qwen38

hf download billchen42/JustFit-Qwen3.8-27B-components \
  --revision 13e0462fa911a1bcbdccbaa759120500c0f90582 \
  --local-dir ./justfit-components
```

The first download is the 27B MXFP4 model. The second contains the exact MTP
and BF16 vision components used by the public JustFit setup.

Verify the downloaded component files:

```bash
shasum -a 256 \
  justfit-components/vision-bf16.safetensors \
  justfit-components/mtp/model.safetensors \
  justfit-components/mtp/config.json
```

The three hashes, in order, should be:

```text
b27ad7963b3752b5cab9d225f9c0b033d0d59066e7ff357dac17d690a6ff7994
9b1d99f402f98940e08891d04815271e3fa94d6bc762f042fd311d3aa0d8bd1d
e0d0d5dc68f59559940e1b2dafc40a34700b9ae7ff963de268225a54550af563
```

Now extract the model's output head and input embedding. These create local
backing files used by JustFit's phase-aware residency manager. The embedding is
already in the model download, so this step does not download another copy:

```bash
python examples/justfit/prepare_components.py \
  --model ./justfit-qwen38 \
  --output ./justfit-extracted \
  --language-only
```

Confirm that all required files exist:

```bash
test -f justfit-qwen38/config.json && echo "model: OK"
test -f justfit-components/mtp/model.safetensors && echo "MTP: OK"
test -f justfit-components/vision-bf16.safetensors && echo "vision: OK"
test -f justfit-extracted/language-head.safetensors && echo "head: OK"
test -f justfit-extracted/input-embedding.safetensors && echo "embedding: OK"
```

You should see five lines ending in `OK`.

## 5. Create a local API key

The key prevents another app on your network from using the model without
permission. The following commands create it in your home directory, outside
the Git repository:

```bash
mkdir -p "$HOME/.config/justfit"
chmod 700 "$HOME/.config/justfit"
python -c 'import secrets; print(secrets.token_urlsafe(32))' \
  > "$HOME/.config/justfit/api-key"
chmod 600 "$HOME/.config/justfit/api-key"
export JUSTFIT_API_KEY="$(cat "$HOME/.config/justfit/api-key")"
```

Do not paste the key into GitHub issues, screenshots, or public configuration
files. To load the same key in a new Terminal later, run:

```bash
export JUSTFIT_API_KEY="$(cat "$HOME/.config/justfit/api-key")"
```

## 6. Start JustFit

Keep this first Terminal open. The command uses one serving lane with a
64K-input + 8K-output envelope:

```bash
cd "$HOME/justfit-mlx"
source .venv/bin/activate
export JUSTFIT_API_KEY="$(cat "$HOME/.config/justfit/api-key")"

MODEL_PATH=justfit-qwen38 \
MTP_PATH="$PWD/justfit-components/mtp" \
VISION_PATH="$PWD/justfit-components/vision-bf16.safetensors" \
LM_HEAD_PATH="$PWD/justfit-extracted/language-head.safetensors" \
INPUT_EMBEDDING_PATH="$PWD/justfit-extracted/input-embedding.safetensors" \
HOST=127.0.0.1 PORT=8080 API_KEY="$JUSTFIT_API_KEY" \
LANES=1 KV_CAPACITY=73728 MAX_TOKENS=8192 OUTPUT_GUARANTEE=8192 \
TOKEN_QUEUE_TIMEOUT=1800 \
examples/justfit/run_server.sh
```

Wait until Terminal prints:

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8080
```

Leave this Terminal running. Press `Control-C` when you want to stop JustFit.

> Run only one copy of the 27B server. Two MLX model servers can exceed unified
> memory and make the Mac unresponsive.

This 64K+8K profile is deliberately below the paper's boundary configuration.
Do not increase `KV_CAPACITY` just because a newer experimental capacity number
appears in a development log: the pool, output guarantee, process guard, APC
mode, and concurrency must be qualified as one configuration.

## 7. Send the first message

Open a **second Terminal window**, then run:

```bash
cd "$HOME/justfit-mlx"
source .venv/bin/activate
export JUSTFIT_API_KEY="$(cat "$HOME/.config/justfit/api-key")"

python examples/justfit/chat_cli.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model justfit-qwen38 \
  --max-tokens 8192
```

At the `you>` prompt, type a message and press Return. Use `/reset` to start a
fresh conversation and `/exit` to leave. The client treats a stream that ends
without `[DONE]` as an error instead of silently accepting a partial answer.

If this works, the model and server are installed correctly. You can stop here
or connect one of the applications below.

## Option A: Open WebUI

[Open WebUI](https://openwebui.com/) provides a ChatGPT-like browser interface.
This part requires [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/).

### A1. Restart JustFit for Docker access

In the Terminal running JustFit, press `Control-C`. Run the same server command
again, but change `HOST=127.0.0.1` to `HOST=0.0.0.0`:

```bash
cd "$HOME/justfit-mlx"
source .venv/bin/activate
export JUSTFIT_API_KEY="$(cat "$HOME/.config/justfit/api-key")"

MODEL_PATH=justfit-qwen38 \
MTP_PATH="$PWD/justfit-components/mtp" \
VISION_PATH="$PWD/justfit-components/vision-bf16.safetensors" \
LM_HEAD_PATH="$PWD/justfit-extracted/language-head.safetensors" \
INPUT_EMBEDDING_PATH="$PWD/justfit-extracted/input-embedding.safetensors" \
HOST=0.0.0.0 PORT=8080 API_KEY="$JUSTFIT_API_KEY" \
LANES=1 KV_CAPACITY=73728 MAX_TOKENS=8192 OUTPUT_GUARANTEE=8192 \
TOKEN_QUEUE_TIMEOUT=1800 \
examples/justfit/run_server.sh
```

The launcher refuses a non-local bind unless an API key is present.

### A2. Start Open WebUI

With Docker Desktop running, open another Terminal and run:

```bash
docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Open [http://localhost:3000](http://localhost:3000) in a browser. The first
local account becomes the administrator for this local installation.

### A3. Connect Open WebUI to JustFit

1. Open **Admin Panel → Settings → Connections**.
2. Add an **OpenAI-compatible** connection.
3. Set the URL to `http://host.docker.internal:8080/v1`.
4. Paste the key displayed by this command:

   ```bash
   cat "$HOME/.config/justfit/api-key"
   ```

5. Save the connection.
6. Start a new chat and select `justfit-qwen38` from the model menu.

Do not forward port 8080 through your router or expose this research server
directly to the public Internet.

## Option B: Hermes Agent

Hermes turns the model into a coding and tool-using terminal agent. JustFit's
Hermes profile declares 73,728 total positions: 65,536 for input and 8,192 for
output.

Install Hermes using its official installer:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
hermes --version
```

Configure the local endpoint:

```bash
hermes model
```

Choose **Custom Endpoint** and enter:

| Question | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8080/v1` |
| API key | the contents of `~/.config/justfit/api-key` |
| Model | `justfit-qwen38` |
| Context length | `73728` |
| API mode / transport | `Chat Completions` |

Hermes uses `context_length` to manage the combined conversation window. The
8,192-token per-response ceiling is enforced by the JustFit server's
`MAX_TOKENS=8192`; current Hermes releases intentionally leave custom-endpoint
output limits to the server.

Start Hermes:

```bash
hermes --tui
```

For a first test, ask:

```text
Inspect the current folder and summarize what this project does. Do not edit anything.
```

Hermes adds its own system prompt and tool descriptions, so even a short
question can create a large model prompt. Long cold prefills may take several
minutes. Keep the JustFit server Terminal open and set any surrounding proxy or
client timeout to at least 1,800 seconds.

## Stop, restart, and update

- Stop JustFit: press `Control-C` in its server Terminal.
- Restart JustFit: repeat [Step 6](#6-start-justfit).
- Stop Open WebUI: `docker stop open-webui`.
- Start it again: `docker start open-webui`.
- Return to the tested source at any time:

  ```bash
  cd "$HOME/justfit-mlx"
  git checkout justfit-repro-v4
  ```

Your model downloads remain in `~/justfit-mlx/justfit-qwen38` and
`~/justfit-mlx/justfit-components`; checking out the tag does not download them
again.

## Troubleshooting

### `hf: command not found`

Reactivate the Python environment:

```bash
cd "$HOME/justfit-mlx"
source .venv/bin/activate
python -m pip install -U huggingface_hub
```

### `401 Unauthorized`

The client and server are using different keys. In every Terminal, reload the
same key:

```bash
export JUSTFIT_API_KEY="$(cat "$HOME/.config/justfit/api-key")"
```

Then restart the client. If you changed the key, restart the server too.

### Open WebUI cannot find the model

Confirm all three points:

1. JustFit was started with `HOST=0.0.0.0` and a non-empty `API_KEY`.
2. Open WebUI uses `http://host.docker.internal:8080/v1`, not `localhost`.
3. The configured model name is exactly `justfit-qwen38`.

### The server says a required path is missing

Run the five `test -f` commands in [Step 4](#4-download-the-model-and-justfit-components).
Repeat only the download or extraction step whose `OK` line is missing.

### The first answer is slow

The first request initializes Metal kernels and loads phase-specific
components. A long prompt also requires a long prefill. Watch the server
Terminal: progress lines mean it is working.

### Disk use grows after repeated long conversations

Persistent APC can keep exact prefix checkpoints on disk so a later request can
reuse a matching prefix. Long prefixes produce large files. This is expected,
but cache retention should be bounded and monitored; a warm APC restore is not
the same measurement as a cold prefill. The immutable v4 runtime uses the
qualified bounded single-pass writer, while capacity evidence still reports
cold and warm-prefix protocols separately.

### The Mac becomes very slow or unresponsive

Stop the request and make sure no second model server, image generator, or
other large GPU workload is running. The 24 GB result assumes one 27B serving
process owns the relevant unified-memory budget.

### A stream closes without `[DONE]`

Treat the answer as incomplete. Check the server Terminal for an exception or
memory-guard event, then retry only after fixing the cause. The included
terminal client reports this condition explicitly.

## What this setup does not claim

- It is not the paper's fully cold 240K-input + 16K-output boundary experiment.
- It does not prove 64K-prompt coding quality; it provides the memory envelope
  required by Hermes and a tested agent connectivity path.
- It is a single-lane profile. Multi-request B2/B4 experiments use a shared
  pool and different admission settings.
- The server is research software, not a hardened public Internet service.

For benchmark reproduction and evidence definitions, continue to the
[full reproduction guide](README.md). For configuration details, see the
[serving reference](../../docs/justfit-serving.md).
