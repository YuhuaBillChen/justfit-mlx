import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "examples" / "justfit" / "prepare_components.py"
RUN_SERVER = ROOT / "examples" / "justfit" / "run_server.sh"


class _Component:
    def __init__(self, value):
        self.value = value

    def parameters(self):
        return {"weight": self.value}


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("justfit_prepare_components", PREPARE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_components_writes_head_and_input_embedding(monkeypatch, tmp_path):
    module = _load_prepare_module()
    head = _Component(mx.array([[1.0, 2.0]]))
    embedding = _Component(mx.array([[3.0, 4.0]]))
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            lm_head=head,
            model=SimpleNamespace(embed_tokens=embedding),
        ),
        vision_tower=_Component(mx.array([[5.0, 6.0]])),
    )
    monkeypatch.setattr(module, "get_model_path", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(module, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PREPARE),
            "--model",
            "unused",
            "--output",
            str(tmp_path / "output"),
            "--language-only",
        ],
    )

    module.main()

    output = tmp_path / "output"
    assert set(mx.load(str(output / "language-head.safetensors"))) == {"weight"}
    assert set(mx.load(str(output / "input-embedding.safetensors"))) == {"weight"}
    assert not (output / "vision-tower.safetensors").exists()


def test_head_only_remains_a_language_only_alias(monkeypatch, tmp_path):
    module = _load_prepare_module()
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            lm_head=_Component(mx.array([[1.0]])),
            model=SimpleNamespace(embed_tokens=_Component(mx.array([[2.0]]))),
        ),
        vision_tower=_Component(mx.array([[3.0]])),
    )
    monkeypatch.setattr(module, "get_model_path", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(module, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PREPARE),
            "--model",
            "unused",
            "--output",
            str(tmp_path / "output"),
            "--head-only",
        ],
    )

    module.main()

    output = tmp_path / "output"
    assert (output / "language-head.safetensors").exists()
    assert (output / "input-embedding.safetensors").exists()
    assert not (output / "vision-tower.safetensors").exists()


def test_run_server_requires_and_exports_input_embedding(tmp_path):
    model = tmp_path / "model"
    mtp = tmp_path / "mtp"
    bin_dir = tmp_path / "bin"
    model.mkdir()
    mtp.mkdir()
    bin_dir.mkdir()
    (model / "config.json").touch()
    (mtp / "config.json").touch()
    for name in ("head", "embedding", "vision"):
        (tmp_path / name).touch()
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'printf "embedding=%s\\n" "$MLX_VLM_VISION_EMBEDDING_SWAP_PATH"\n'
        'printf "head=%s\\n" "$MLX_VLM_LANGUAGE_HEAD_PHASE_SWAP_PATH"\n'
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "MODEL_PATH": str(model),
        "MTP_PATH": str(mtp),
        "LM_HEAD_PATH": str(tmp_path / "head"),
        "VISION_PATH": str(tmp_path / "vision"),
    }

    missing = subprocess.run(
        [str(RUN_SERVER)], env=env, capture_output=True, text=True, check=False
    )
    assert missing.returncode == 2
    assert "missing required environment variable: INPUT_EMBEDDING_PATH" in missing.stderr

    env["INPUT_EMBEDDING_PATH"] = str(tmp_path / "embedding")
    complete = subprocess.run(
        [str(RUN_SERVER)], env=env, capture_output=True, text=True, check=True
    )
    assert f"embedding={tmp_path / 'embedding'}" in complete.stdout
    assert f"head={tmp_path / 'head'}" in complete.stdout
