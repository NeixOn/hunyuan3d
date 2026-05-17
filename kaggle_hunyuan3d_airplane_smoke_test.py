"""
Kaggle smoke test for image-to-3D on ShapeNet airplane renders.

Target environment: Kaggle GPU T4 x2. This script intentionally tests shape
generation first, without the Hunyuan3D-Paint texturing stage, because geometry
is the part we need to validate before spending VRAM on textures.

Run from the cloned project directory:
    python kaggle_hunyuan3d_airplane_smoke_test.py

Default Kaggle paths used by this script:
    reconstruction image:
        ./image/airplan.png
    ShapeNetCore airplane meshes:
        /kaggle/input/datasets/neixon/airplanedataset/02691156/<instance>/models/model_normalized.obj
    ShapeNetRendering airplane views:
        /kaggle/input/datasets/ronak555/shapenetcorerendering-part1/kaggle/tmp/ShapeNetRendering/02691156/<instance>/rendering/00.png

Optional env overrides:
    INPUT_IMAGE=/path/to/image.png
    HY3D_INSTALL_PROFILE=shape|full
    SHAPENET_CORE_ROOT=/path/to/ShapeNetCore/02691156-or-parent
    SHAPENET_RENDERING_ROOT=/path/to/ShapeNetRendering/02691156-or-parent
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


AIRPLANE_SYNSET = "02691156"
PROJECT_DIR = Path(__file__).resolve().parent
WORKDIR = Path(os.environ.get("WORKDIR", PROJECT_DIR)).resolve()
REPO_DIR = WORKDIR / "Hunyuan3D-2.1"
OUTPUT_DIR = WORKDIR / "hy3d_airplane_outputs"
PATCHED_REQUIREMENTS = WORKDIR / "hy3d_requirements_kaggle.txt"
LOCAL_IMAGE_DIR = Path(os.environ.get("LOCAL_IMAGE_DIR", PROJECT_DIR / "image")).resolve()

MODEL_ID = os.environ.get("HY3D_MODEL_ID", "tencent/Hunyuan3D-2.1")
SUBFOLDER = os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-1")
STEPS = int(os.environ.get("HY3D_STEPS", "30"))
OCTREE_RESOLUTION = int(os.environ.get("HY3D_OCTREE_RESOLUTION", "256"))
INSTALL_PROFILE = os.environ.get("HY3D_INSTALL_PROFILE", "shape").lower()
USE_SAFETENSORS_ENV = os.environ.get("HY3D_USE_SAFETENSORS", "0").strip().lower()
USE_SAFETENSORS = USE_SAFETENSORS_ENV in {"1", "true", "yes", "on"}

DEFAULT_CORE_ROOT = Path("/kaggle/input/datasets/neixon/airplanedataset")
DEFAULT_RENDERING_ROOT = Path(
    "/kaggle/input/datasets/ronak555/shapenetcorerendering-part1/kaggle/tmp/ShapeNetRendering"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def requirement_name(line: str) -> str:
    token = line.strip()
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if marker in token:
            token = token.split(marker, 1)[0]
            break
    return token.strip().lower().replace("_", "-")


def build_kaggle_requirements() -> Path:
    source = REPO_DIR / "requirements.txt"
    if os.environ.get("HY3D_REQUIREMENTS_FILE"):
        return Path(os.environ["HY3D_REQUIREMENTS_FILE"]).resolve()

    py312_replacements = {
        "numpy==1.24.4": "numpy==1.26.4",
        "pymeshlab==2022.2.post3": "pymeshlab==2023.12.post3",
        "open3d==0.18.0": "open3d==0.19.0",
        "onnxruntime==1.16.3": "onnxruntime==1.18.0",
    }
    shape_excluded_packages = {
        # Texture upscaling / paint pipeline.
        "realesrgan",
        "basicsr",
        "tb-nightly",
        "cupy-cuda12x",
        # Demo server.
        "gradio",
        "fastapi",
        "uvicorn",
        # Blender / ONNX / training helpers not needed for shape-only inference.
        "bpy",
        "onnxruntime",
        "deepspeed",
        "pythreejs",
    }
    lines = source.read_text(encoding="utf-8").splitlines()
    patched = []
    replacements_used = []
    skipped_for_shape = []
    for line in lines:
        stripped = line.strip()
        if INSTALL_PROFILE == "shape" and stripped and not stripped.startswith(("#", "--")):
            name = requirement_name(stripped)
            if name in shape_excluded_packages:
                patched.append(f"# skipped for shape-only smoke test: {line}")
                skipped_for_shape.append(stripped)
                continue

        replacement = py312_replacements.get(stripped)
        if sys.version_info >= (3, 12) and replacement:
            patched.append(replacement)
            replacements_used.append((stripped, replacement))
            continue
        patched.append(line)

    PATCHED_REQUIREMENTS.write_text("\n".join(patched) + "\n", encoding="utf-8")
    print(f"Using Kaggle requirements file: {PATCHED_REQUIREMENTS}", flush=True)
    print(f"Dependency install profile: {INSTALL_PROFILE}", flush=True)
    for old, new in replacements_used:
        print(f"Patched {old} to {new} for Python 3.12.", flush=True)
    if skipped_for_shape:
        print("Skipped packages for shape-only smoke test:", flush=True)
        for package in skipped_for_shape:
            print(f"  - {package}", flush=True)
    return PATCHED_REQUIREMENTS


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

    if os.environ.get("HY3D_SKIP_DEP_INSTALL", "0") == "1":
        print("Skipping dependency installation because HY3D_SKIP_DEP_INSTALL=1.", flush=True)
        return

    requirements_file = build_kaggle_requirements()
    run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], cwd=REPO_DIR)


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
        f"No local image found in {LOCAL_IMAGE_DIR}; falling back to ShapeNetRendering.",
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

        matches = sorted(synset_root.glob("*/models/model_normalized.obj"))
        if matches:
            print(f"Found example ground-truth mesh: {matches[0]}", flush=True)
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
        f"Could not find an input image. Put it under {LOCAL_IMAGE_DIR}, "
        "or set INPUT_IMAGE=/path/to/image.png."
    )


def print_environment() -> None:
    print("=== Kaggle Hunyuan3D airplane smoke test ===", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Project dir: {PROJECT_DIR}", flush=True)
    print(f"Workdir: {WORKDIR}", flush=True)
    print(f"Repo dir: {REPO_DIR}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Local image dir: {LOCAL_IMAGE_DIR}", flush=True)
    print(f"Default ShapeNetCore root: {DEFAULT_CORE_ROOT}", flush=True)
    print(f"Default ShapeNetRendering root: {DEFAULT_RENDERING_ROOT}", flush=True)
    print(f"Model: {MODEL_ID} / {SUBFOLDER}", flush=True)
    print(
        f"Use safetensors: {USE_SAFETENSORS} "
        f"(HY3D_USE_SAFETENSORS={USE_SAFETENSORS_ENV!r})",
        flush=True,
    )
    print(f"Steps: {STEPS}", flush=True)
    print(f"Octree resolution: {OCTREE_RESOLUTION}", flush=True)

    try:
        run(["nvidia-smi"])
    except Exception as exc:
        print(f"WARNING: nvidia-smi failed: {exc}", flush=True)


def generate_shape(input_image: Path) -> Path:
    shape_dir = REPO_DIR / "hy3dshape"
    for import_dir in (shape_dir, REPO_DIR):
        if str(import_dir) not in sys.path:
            sys.path.insert(0, str(import_dir))

    from PIL import Image
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{input_image.stem}_hunyuan3d_shape.glb"

    print("Loading Hunyuan3D shape pipeline...", flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_ID,
        subfolder=SUBFOLDER,
        use_safetensors=USE_SAFETENSORS,
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
