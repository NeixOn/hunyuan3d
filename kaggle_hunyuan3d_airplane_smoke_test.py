"""
Kaggle smoke test for image-to-3D on ShapeNet airplane renders.

Target environment: Kaggle GPU T4 x2. This script intentionally tests shape
generation first, without the Hunyuan3D-Paint texturing stage, because geometry
is the part we need to validate before spending VRAM on textures.

Run in a Kaggle notebook cell:
    !python /kaggle/working/kaggle_hunyuan3d_airplane_smoke_test.py

If your ShapeNetRendering dataset is mounted under a different Kaggle path,
set SHAPENET_RENDERING_ROOT before running.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


AIRPLANE_SYNSET = "02691156"
WORKDIR = Path(os.environ.get("WORKDIR", "/kaggle/working")).resolve()
REPO_DIR = WORKDIR / "Hunyuan3D-2.1"
OUTPUT_DIR = WORKDIR / "hy3d_airplane_outputs"

MODEL_ID = os.environ.get("HY3D_MODEL_ID", "tencent/Hunyuan3D-2.1")
SUBFOLDER = os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-1")
STEPS = int(os.environ.get("HY3D_STEPS", "30"))
OCTREE_RESOLUTION = int(os.environ.get("HY3D_OCTREE_RESOLUTION", "256"))


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def install_repo() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        run(["git", "clone", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git", str(REPO_DIR)])

    # Kaggle usually already ships a CUDA-enabled torch build. Reinstalling
    # torch is intentionally opt-in because it can consume time and break a
    # working notebook image.
    if os.environ.get("HY3D_INSTALL_TORCH", "0") == "1":
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch==2.5.1",
                "torchvision==0.20.1",
                "torchaudio==2.5.1",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ]
        )

    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=REPO_DIR)


def find_airplane_render() -> Path:
    explicit = os.environ.get("INPUT_IMAGE")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"INPUT_IMAGE does not exist: {path}")

    roots = []
    if os.environ.get("SHAPENET_RENDERING_ROOT"):
        roots.append(Path(os.environ["SHAPENET_RENDERING_ROOT"]))
    roots.extend(
        [
            Path("/kaggle/input/ShapeNetRendering"),
            Path("/kaggle/input/shapenetrendering"),
            Path("/kaggle/input/shape-net-rendering"),
            Path("/kaggle/input"),
        ]
    )

    patterns = [
        f"**/{AIRPLANE_SYNSET}/*/rendering/*.png",
        f"**/{AIRPLANE_SYNSET}/*/*.png",
        f"**/{AIRPLANE_SYNSET}/**/*.jpg",
    ]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                print(f"Using input render: {matches[0]}", flush=True)
                return matches[0]

    raise FileNotFoundError(
        "Could not find an airplane render. Set INPUT_IMAGE=/path/to/render.png ")


def print_environment() -> None:
    print("=== Kaggle Hunyuan3D airplane smoke test ===", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Workdir: {WORKDIR}", flush=True)
    print(f"Repo dir: {REPO_DIR}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Model: {MODEL_ID} / {SUBFOLDER}", flush=True)
    print(f"Steps: {STEPS}", flush=True)
    print(f"Octree resolution: {OCTREE_RESOLUTION}", flush=True)

    try:
        run(["nvidia-smi"])
    except Exception as exc:
        print(f"WARNING: nvidia-smi failed: {exc}", flush=True)


def generate_shape(input_image: Path) -> Path:
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{input_image.stem}_hunyuan3d_shape.glb"

    print("Loading Hunyuan3D shape pipeline...", flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_ID,
        subfolder=SUBFOLDER,
        use_safetensors=True,
    )

    image = Image.open(input_image).convert("RGBA")
    print("Running shape generation. This is the slow/VRAM-heavy part.", flush=True)
    result = pipeline(
        image=image,
        num_inference_steps=STEPS,
        octree_resolution=OCTREE_RESOLUTION,
    )

    mesh = result[0] if isinstance(result, (list, tuple)) else result
    mesh.export(output_path)
    print(f"Saved mesh: {output_path}", flush=True)
    return output_path


def main() -> None:
    print_environment()
    input_image = find_airplane_render()
    install_repo()
    output_path = generate_shape(input_image)
    print("DONE", flush=True)
    print(f"Output file: {output_path}", flush=True)


if __name__ == "__main__":
    main()
