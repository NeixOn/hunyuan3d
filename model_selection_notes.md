# Image-to-3D Model Choice for ShapeNet Airplanes

Date: 2026-05-17

## Shortlist

### 1. TRELLIS.2

Best open candidate by architecture/quality on paper: 4B parameters, O-Voxel
representation, high-resolution PBR assets, MIT license, and full training code.

Problem for this project: the official repo says an NVIDIA GPU with at least
24 GB memory is necessary. Kaggle T4 x2 gives two 16 GB GPUs, not one 32 GB
device, so basic inference/fine-tuning is risky without low-VRAM/offload hacks.

Verdict: best research target, bad first Kaggle T4 smoke test.

### 2. Pixal3D

Very new SIGGRAPH 2026 model. It is built on TRELLIS.2 and improves pixel-level
faithfulness by back-projecting image features into 3D. This is especially
interesting for ShapeNetRendering because the input view alignment matters.

Problem for this project: also inherits TRELLIS.2 heaviness; repo is young; the
public README focuses on inference, not a mature fine-tuning path.

Verdict: strongest quality candidate to watch/test later, but not the first
model I would fight on T4 x2.

### 3. Hunyuan3D 2.1

Best practical first candidate. It has open code/weights, a shape pipeline, a
texture pipeline, low-VRAM mode in the Gradio app, and public training-related
materials. HY3D-Bench also includes Hunyuan3D-Shape-v2-1 Small baseline weights.

Verdict: start here for Kaggle. Test shape generation first, then decide whether
to add texturing or fine-tune/adapt.

### 4. SPAR3D / Stable Fast 3D

Fast, practical feed-forward reconstruction models from Stability AI. Good for
baselines and quick comparisons. Less attractive as the main fine-tuning target
than Hunyuan/TRELLIS-style models because the adaptation path is less direct.

Verdict: keep as baselines.

## Recommended First Experiment

Use `kaggle_hunyuan3d_airplane_smoke_test.py` from the cloned project folder on
Kaggle:

```bash
%cd /path/to/cloned/project
!python kaggle_hunyuan3d_airplane_smoke_test.py
```

On Kaggle Python 3.12, the script writes `hy3d_requirements_kaggle.txt` and
uses `HY3D_INSTALL_PROFILE=shape` by default. This installs only the dependencies
needed for shape-only reconstruction and skips texture/demo/training packages
that are not needed for the smoke test:

```text
realesrgan, basicsr, tb_nightly, cupy-cuda12x, gradio, fastapi, uvicorn,
bpy, onnxruntime, deepspeed, pythreejs
```

For Python 3.12 compatibility, it also patches old Hunyuan3D pins that do not
have suitable wheels:

```text
numpy==1.24.4 -> numpy==1.26.4
pymeshlab==2022.2.post3 -> pymeshlab==2023.12.post3
open3d==0.18.0 -> open3d==0.19.0
onnxruntime==1.16.3 -> onnxruntime==1.18.0
```

If installation failed before this patch, just pull/update this repo and run the
script again; the already cloned `Hunyuan3D-2.1` directory can stay in place.
After dependencies are installed once, you can skip that step with:

```bash
%env HY3D_SKIP_DEP_INSTALL=1
!python kaggle_hunyuan3d_airplane_smoke_test.py
```

The Hunyuan3D-2.1 shape API is imported from `hy3dshape.pipelines`; the older
`hy3dgen.shapegen` import belongs to a different Hunyuan3D code layout.

The Hunyuan3D-2.1 Hugging Face shape weights currently download as
`model.fp16.ckpt`, so the script uses `HY3D_USE_SAFETENSORS=0` by default. Only
set `HY3D_USE_SAFETENSORS=1` if you have a matching `model.fp16.safetensors`
file in the cache/model folder.

To intentionally try the full official requirements instead:

```bash
%env HY3D_INSTALL_PROFILE=full
!python kaggle_hunyuan3d_airplane_smoke_test.py
```

The current script expects the reconstruction input image first at:

```text
./image/airplan.png
```

It also knows the current Kaggle dataset layout:

```text
/kaggle/input/datasets/neixon/airplanedataset/02691156/<instance>/models/model_normalized.obj
/kaggle/input/datasets/ronak555/shapenetcorerendering-part1/kaggle/tmp/ShapeNetRendering/02691156/<instance>/rendering/00.png
```

If you want to force one specific image:

```bash
%env INPUT_IMAGE=/path/to/image.png
!python kaggle_hunyuan3d_airplane_smoke_test.py
```

Lower memory settings:

```bash
%env HY3D_STEPS=20
%env HY3D_OCTREE_RESOLUTION=128
!python kaggle_hunyuan3d_airplane_smoke_test.py
```

## Improvement Path

1. Build a validation set from ShapeNet airplane:
   - input: one render from the 24 views;
   - target: ShapeNetCore mesh;
   - metrics: Chamfer Distance, F-score, normal consistency, visual turntable.

2. Establish baselines:
   - Hunyuan3D 2.1 shape only;
   - Hunyuan3D 2.1 shape + paint if memory allows;
   - SF3D/SPAR3D for fast comparison;
   - TRELLIS.2 or Pixal3D only if a 24 GB+ GPU is available.

3. Adaptation options, from cheapest to most expensive:
   - airplane-specific preprocessing: masks, crop, canonical view, alpha;
   - airplane-specific postprocessing: symmetry cleanup, component filtering;
   - fine-tune only shape flow / late blocks / adapters where possible;
   - train a small class-specific model using ShapeNet meshes converted to the
     model's latent representation.

4. Avoid full-model fine-tuning on T4 x2 unless the official training config
   already fits. Two T4 GPUs do not behave like one 32 GB GPU.
