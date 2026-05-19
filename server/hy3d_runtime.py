from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
from PIL import Image

from .config import MODEL_ID, OCTREE_RESOLUTION, REPO_DIR, STEPS, SUBFOLDER, USE_SAFETENSORS


class Hunyuan3DShapeRuntime:
    def __init__(self) -> None:
        self.pipeline = None

    def _add_repo_to_path(self) -> None:
        shape_dir = REPO_DIR / "hy3dshape"
        for import_dir in (shape_dir, REPO_DIR):
            if str(import_dir) not in sys.path:
                sys.path.insert(0, str(import_dir))

    def load(self) -> None:
        if self.pipeline is not None:
            return
        self._add_repo_to_path()
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        print("Loading Hunyuan3D shape pipeline...", flush=True)
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_ID,
            subfolder=SUBFOLDER,
            use_safetensors=USE_SAFETENSORS,
        )
        print("Hunyuan3D shape pipeline loaded.", flush=True)

    def generate(self, input_image: Path, output_glb: Path) -> Path:
        self.load()
        assert self.pipeline is not None

        output_glb.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(input_image).convert("RGBA")
        torch.set_grad_enabled(False)

        print(
            f"Generating shape: steps={STEPS}, octree_resolution={OCTREE_RESOLUTION}",
            flush=True,
        )
        with torch.inference_mode():
            result = self.pipeline(
                image=image,
                num_inference_steps=STEPS,
                octree_resolution=OCTREE_RESOLUTION,
            )

        mesh = result[0] if isinstance(result, (list, tuple)) else result
        mesh.export(output_glb)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output_glb


runtime = Hunyuan3DShapeRuntime()
