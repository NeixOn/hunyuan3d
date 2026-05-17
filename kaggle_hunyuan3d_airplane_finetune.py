"""Kaggle launcher for Hunyuan3D-2.1 shape fine-tuning.

This file is intentionally separate from `kaggle_hunyuan3d_airplane_smoke_test.py`.
Keep the smoke-test file as the known-good inference launcher.

The official Hunyuan3D shape training code expects a preprocessed dataset:

    preprocessed/{uid}/geo_data/{uid}_surface.npz
    preprocessed/{uid}/geo_data/{uid}_sdf.npz
    preprocessed/{uid}/render_cond/000.png ... transforms.json

By default this script runs a tiny training-loop sanity check on the official
`tools/mini_trainset/preprocessed` data if it exists in the cloned repo. For a
real airplane fine-tune, first preprocess ShapeNet meshes/renders with the
official `hy3dshape/tools` pipeline, then set:

    HY3D_TRAIN_DATA_LIST=/path/to/preprocessed
    HY3D_TRAIN_VAL_DATA_LIST=/path/to/preprocessed
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_VERSION = "2026-05-17-training-deps-v2"
PROJECT_DIR = Path(__file__).resolve().parent
WORKDIR = Path(os.environ.get("WORKDIR", PROJECT_DIR)).resolve()
REPO_DIR = WORKDIR / "Hunyuan3D-2.1"
SHAPE_DIR = REPO_DIR / "hy3dshape"
CONFIG_DIR = SHAPE_DIR / "configs"

OUTPUT_DIR = Path(
    os.environ.get("HY3D_TRAIN_OUTPUT_DIR", WORKDIR / "hy3d_airplane_finetune_outputs")
).resolve()
GENERATED_CONFIG = OUTPUT_DIR / "kaggle_t4_finetune_config.yaml"

DEFAULT_MINI_DATASET = SHAPE_DIR / "tools" / "mini_trainset" / "preprocessed"
TRAIN_DATA_LIST = Path(
    os.environ.get("HY3D_TRAIN_DATA_LIST", DEFAULT_MINI_DATASET)
).resolve()
VAL_DATA_LIST = Path(
    os.environ.get("HY3D_TRAIN_VAL_DATA_LIST", TRAIN_DATA_LIST)
).resolve()

TRAIN_STEPS = int(os.environ.get("HY3D_TRAIN_STEPS", "20"))
TRAIN_GPUS = int(os.environ.get("HY3D_TRAIN_GPUS", "1"))
TRAIN_BATCH_SIZE = int(os.environ.get("HY3D_TRAIN_BATCH_SIZE", "1"))
TRAIN_NUM_WORKERS = int(os.environ.get("HY3D_TRAIN_NUM_WORKERS", "2"))
TRAIN_VAL_NUM_WORKERS = int(os.environ.get("HY3D_TRAIN_VAL_NUM_WORKERS", "1"))
TRAIN_LR = float(os.environ.get("HY3D_TRAIN_LR", "1e-5"))
TRAIN_AMP_TYPE = os.environ.get("HY3D_TRAIN_AMP_TYPE", "16")
TRAIN_DEEPSPEED = os.environ.get("HY3D_TRAIN_DEEPSPEED", "0") == "1"
ENABLE_MESH_LOGS = os.environ.get("HY3D_TRAIN_ENABLE_MESH_LOGS", "0") == "1"
CUDA_VISIBLE_DEVICES = os.environ.get("HY3D_TRAIN_CUDA_VISIBLE_DEVICES", "0")
INSTALL_TRAIN_DEPS = os.environ.get("HY3D_TRAIN_INSTALL_DEPS", "1") == "1"
FORCE_MATH_ATTENTION = os.environ.get("HY3D_TRAIN_FORCE_MATH_ATTENTION", "1") == "1"
PATCH_DIR = OUTPUT_DIR / "runtime_patches"


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_repo_exists() -> None:
    if REPO_DIR.exists():
        print(f"Using existing Hunyuan3D repo: {REPO_DIR}", flush=True)
        return
    run(["git", "clone", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git", str(REPO_DIR)])


def torch_wheel_tag() -> str:
    import torch

    torch_version = torch.__version__.split("+", 1)[0]
    cuda_version = torch.version.cuda
    if cuda_version:
        cuda_tag = "cu" + cuda_version.replace(".", "")
    else:
        cuda_tag = "cpu"
    return f"torch-{torch_version}+{cuda_tag}"


def ensure_training_dependencies() -> None:
    if not INSTALL_TRAIN_DEPS:
        print("Skipping training dependency installation because HY3D_TRAIN_INSTALL_DEPS=0.", flush=True)
        return

    try:
        import torch_cluster  # noqa: F401

        print("torch_cluster is already installed.", flush=True)
        return
    except ModuleNotFoundError:
        pass

    wheel_tag = torch_wheel_tag()
    wheel_index = f"https://data.pyg.org/whl/{wheel_tag}.html"
    print(f"Installing torch_cluster from PyG wheels: {wheel_index}", flush=True)
    run([sys.executable, "-m", "pip", "install", "torch-cluster", "-f", wheel_index])


def find_base_config() -> Path:
    override = os.environ.get("HY3D_TRAIN_BASE_CONFIG")
    if override:
        path = Path(override).resolve()
        if not path.exists():
            raise FileNotFoundError(f"HY3D_TRAIN_BASE_CONFIG does not exist: {path}")
        return path

    patterns = [
        "*mini-overfitting*512.yaml",
        "*finetuning*512.yaml",
        "*mini-overfitting*4096.yaml",
        "*finetuning*4096.yaml",
    ]
    for pattern in patterns:
        matches = sorted(CONFIG_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No Hunyuan3D training config found in {CONFIG_DIR}")


def validate_dataset(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} dataset does not exist: {path}\n"
            "For a quick training-loop test, make sure the official mini_trainset exists.\n"
            "For airplane fine-tuning, preprocess ShapeNet first and set "
            "HY3D_TRAIN_DATA_LIST/HY3D_TRAIN_VAL_DATA_LIST."
        )
    examples = sorted(path.glob("*"))
    if not examples:
        raise FileNotFoundError(f"{label} dataset is empty: {path}")
    print(f"{label} dataset: {path} ({len(examples)} entries)", flush=True)


def patch_config(base_config: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with base_config.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    training = cfg.setdefault("training", {})
    training["steps"] = TRAIN_STEPS
    training["use_amp"] = True
    training["amp_type"] = TRAIN_AMP_TYPE
    training["base_lr"] = TRAIN_LR
    training["every_n_train_steps"] = max(1, TRAIN_STEPS)
    training["val_check_interval"] = max(1, min(TRAIN_STEPS, int(os.environ.get("HY3D_TRAIN_VAL_INTERVAL", "10"))))
    training["limit_val_batches"] = int(os.environ.get("HY3D_TRAIN_LIMIT_VAL_BATCHES", "1"))
    training["log_every_n_steps"] = int(os.environ.get("HY3D_TRAIN_LOG_EVERY_N_STEPS", "1"))
    training["num_nodes"] = 1
    training["ckpt_path"] = os.environ.get("HY3D_TRAIN_RESUME_CKPT", "")

    dataset_params = cfg.setdefault("dataset", {}).setdefault("params", {})
    dataset_params["train_data_list"] = str(TRAIN_DATA_LIST)
    dataset_params["val_data_list"] = str(VAL_DATA_LIST)
    dataset_params["batch_size"] = TRAIN_BATCH_SIZE
    dataset_params["num_workers"] = TRAIN_NUM_WORKERS
    dataset_params["val_num_workers"] = TRAIN_VAL_NUM_WORKERS

    if not ENABLE_MESH_LOGS:
        cfg.pop("callbacks", None)

    with GENERATED_CONFIG.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False, allow_unicode=False)

    print(f"Base config: {base_config}", flush=True)
    print(f"Kaggle config: {GENERATED_CONFIG}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)


def write_runtime_patches() -> None:
    if not FORCE_MATH_ATTENTION:
        return

    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    sitecustomize = PATCH_DIR / "sitecustomize.py"
    sitecustomize.write_text(
        r'''
import math

import torch
import torch.nn.functional as F


try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


def _math_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if enable_gqa:
        repeat = query.size(-3) // key.size(-3)
        key = key.repeat_interleave(repeat, dim=-3)
        value = value.repeat_interleave(repeat, dim=-3)

    scale_factor = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    if is_causal:
        causal_mask = torch.ones(
            scores.size(-2),
            scores.size(-1),
            dtype=torch.bool,
            device=scores.device,
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = torch.dropout(weights, dropout_p, train=True)
    return torch.matmul(weights, value)


F.scaled_dot_product_attention = _math_scaled_dot_product_attention
print("Using math fallback for torch.nn.functional.scaled_dot_product_attention", flush=True)
'''.lstrip(),
        encoding="utf-8",
    )
    print(f"Runtime patch dir: {PATCH_DIR}", flush=True)


def launch_training() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if FORCE_MATH_ATTENTION:
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(PATCH_DIR)
            if not existing_pythonpath
            else str(PATCH_DIR) + os.pathsep + existing_pythonpath
        )

    cmd = [
        sys.executable,
        "main.py",
        "--fast",
        "--num_nodes",
        "1",
        "--num_gpus",
        str(TRAIN_GPUS),
        "--config",
        str(GENERATED_CONFIG),
        "--output_dir",
        str(OUTPUT_DIR),
    ]
    if TRAIN_DEEPSPEED:
        cmd.append("--deepspeed")

    run(cmd, cwd=SHAPE_DIR, env=env)


def print_environment() -> None:
    print("=== Kaggle Hunyuan3D shape fine-tune launcher ===", flush=True)
    print(f"Script version: {SCRIPT_VERSION}", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Project dir: {PROJECT_DIR}", flush=True)
    print(f"Workdir: {WORKDIR}", flush=True)
    print(f"Repo dir: {REPO_DIR}", flush=True)
    print(f"Train data: {TRAIN_DATA_LIST}", flush=True)
    print(f"Val data: {VAL_DATA_LIST}", flush=True)
    print(f"Steps: {TRAIN_STEPS}", flush=True)
    print(f"GPUs: {TRAIN_GPUS} (CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES})", flush=True)
    print(f"Batch size: {TRAIN_BATCH_SIZE}", flush=True)
    print(f"AMP type: {TRAIN_AMP_TYPE}", flush=True)
    print(f"DeepSpeed: {TRAIN_DEEPSPEED}", flush=True)
    print(f"Mesh logs: {ENABLE_MESH_LOGS}", flush=True)
    print(f"Install training deps: {INSTALL_TRAIN_DEPS}", flush=True)
    print(f"Force math attention: {FORCE_MATH_ATTENTION}", flush=True)
    try:
        run(["nvidia-smi"])
    except Exception as exc:
        print(f"nvidia-smi failed: {exc}", flush=True)


def main() -> None:
    print_environment()
    ensure_repo_exists()
    ensure_training_dependencies()
    validate_dataset(TRAIN_DATA_LIST, "Train")
    validate_dataset(VAL_DATA_LIST, "Validation")
    base_config = find_base_config()
    patch_config(base_config)
    write_runtime_patches()
    launch_training()
    print("Fine-tuning run finished.", flush=True)


if __name__ == "__main__":
    main()
