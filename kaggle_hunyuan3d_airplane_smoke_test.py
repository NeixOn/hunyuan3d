"""
Kaggle smoke test for image-to-3D on ShapeNet airplane renders.

Target environment: Kaggle GPU T4 x2. This script intentionally tests shape
generation first, without the Hunyuan3D-Paint texturing stage, because geometry
is the part we need to validate before spending VRAM on textures.

Run in a Kaggle notebook cell:
    !python /kaggle/working/kaggle_hunyuan3d_airplane_smoke_test.py

Default Kaggle paths used by this script:
    reconstruction image:
        /kaggle/working/image/airplan.png
    ShapeNetCore airplane meshes:
        /kaggle/input/datasets/neixon/airplanedataset/02691156/<instance>/models/model_normalized.obj
    ShapeNetRendering airplane views:
        /kaggle/input/datasets/ronak555/shapenetcorerendering-part1/kaggle/tmp/ShapeNetRendering/02691156/<instance>/rendering/00.png

Optional env overrides:
    INPUT_IMAGE=/path/to/image.png
    SHAPENET_CORE_ROOT=/path/to/ShapeNetCore/02691156-or-parent
    SHAPENET_RENDERING_ROOT=/path/to/ShapeNetRendering/02691156-or-parent
    SHAPENET_INSTANCE_ID=10155655850468db78d106ce0a280f87
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
LOCAL_IMAGE_DIR = WORKDIR / "image"

MODEL_ID = os.environ.get("HY3D_MODEL_ID", "tencent/Hunyuan3D-2.1")
SUBFOLDER = os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-1")
STEPS = int(os.environ.get("HY3D_STEPS", "30"))
OCTREE_RESOLUTION = int(os.environ.get("HY3D_OCTREE_RESOLUTION", "256"))
INSTANCE_ID = os.environ.get("SHAPENET_INSTANCE_ID")

DEFAULT_CORE_ROOT = Path("/kaggle/input/datasets/neixon/airplanedataset")
DEFAULT_RENDERING_ROOT = Path(
    "/kaggle/input/datasets/ronak555/shapenetcorerendering-part1/kaggle/tmp/ShapeNetRendering"
)


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


def find_first_existing_file(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def find_reconstruction_image() -> Path:
    explicit = os.environ.get("INPUT_IMAGE")
    if explicit:
        path = Path(explicit)
        if path.exists():
            print(f"Using explicit reconstruction image: {path}", flush=True)
            return path
        raise FileNotFoundError(f"INPUT_IMAGE does not exist: {path}")

    local_candidates = [
        LOCAL_IMAGE_DIR / "airplan.png",
        LOCAL_IMAGE_DIR / "airplane.png",
        LOCAL_IMAGE_DIR / "00.png",
    ]
    local_image = find_first_existing_file(local_candidates)
    if local_image:
        print(f"Using local reconstruction image: {local_image}", flush=True)
        return local_image

    if LOCAL_IMAGE_DIR.exists():
        matches = sorted(
            path
            for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp")
            for path in LOCAL_IMAGE_DIR.glob(suffix)
        )
        if matches:
            print(f"Using first image from {LOCAL_IMAGE_DIR}: {matches[0]}", flush=True)
            return matches[0]

    print(
        "No local image found in /kaggle/working/image; falling back to ShapeNetRendering.",
        flush=True,
    )
    return find_airplane_render()


def normalize_synset_root(root: Path) -> Path:
    return root if root.name == AIRPLANE_SYNSET else root / AIRPLANE_SYNSET


def find_ground_truth_mesh() -> Path | None:
    roots = []
    if os.environ.get("SHAPENET_CORE_ROOT"):
        roots.append(Path(os.environ["SHAPENET_CORE_ROOT"]))
    roots.append(DEFAULT_CORE_ROOT)

    for root in roots:
        synset_root = normalize_synset_root(root)
        if not synset_root.exists():
            continue

        if INSTANCE_ID:
            candidate = synset_root / INSTANCE_ID / "models" / "model_normalized.obj"
            if candidate.exists():
                print(f"Found ground-truth mesh: {candidate}", flush=True)
                return candidate
            continue

        matches = sorted(synset_root.glob("*/models/model_normalized.obj"))
        if matches:
            print(f"Found ground-truth mesh: {matches[0]}", flush=True)
            return matches[0]

    print("Ground-truth mesh not found; continuing with reconstruction only.", flush=True)
    return None


def find_airplane_render() -> Path:
    roots = []
    if os.environ.get("SHAPENET_RENDERING_ROOT"):
        roots.append(Path(os.environ["SHAPENET_RENDERING_ROOT"]))
    roots.extend(
        [
            DEFAULT_RENDERING_ROOT,
            Path("/kaggle/input/ShapeNetRendering"),
            Path("/kaggle/input/shapenetrendering"),
            Path("/kaggle/input/shape-net-rendering"),
            Path("/kaggle/input"),
        ]
    )

    for root in roots:
        if not root.exists():
            continue
        synset_root = normalize_synset_root(root)

        if INSTANCE_ID:
            candidates = [
                synset_root / INSTANCE_ID / "rendering" / "00.png",
                synset_root / INSTANCE_ID / "rendering" / "00.jpg",
            ]
            image = find_first_existing_file(candidates)
            if image:
                print(f"Using ShapeNet render: {image}", flush=True)
                return image

        patterns = [
            "*/rendering/00.png",
            "*/rendering/*.png",
            "*/*.png",
            "**/*.jpg",
        ]
        for pattern in patterns:
            matches = sorted(synset_root.glob(pattern))
            if matches:
                print(f"Using ShapeNet render: {matches[0]}", flush=True)
                return matches[0]

    raise FileNotFoundError(
        "Could not find an input image. Put it under /kaggle/working/image, "
        "or set INPUT_IMAGE=/path/to/image.png."
    )


def print_environment() -> None:
    print("=== Kaggle Hunyuan3D airplane smoke test ===", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Workdir: {WORKDIR}", flush=True)
    print(f"Repo dir: {REPO_DIR}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Local image dir: {LOCAL_IMAGE_DIR}", flush=True)
    print(f"Default ShapeNetCore root: {DEFAULT_CORE_ROOT}", flush=True)
    print(f"Default ShapeNetRendering root: {DEFAULT_RENDERING_ROOT}", flush=True)
    print(f"ShapeNet instance id: {INSTANCE_ID or '<auto>'}", flush=True)
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
    input_image = find_reconstruction_image()
    find_ground_truth_mesh()
    install_repo()
    output_path = generate_shape(input_image)
    print("DONE", flush=True)
    print(f"Output file: {output_path}", flush=True)


if __name__ == "__main__":
    main()
