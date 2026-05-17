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

Use `kaggle_hunyuan3d_airplane_smoke_test.py` on Kaggle:

```bash
!cp /path/to/kaggle_hunyuan3d_airplane_smoke_test.py /kaggle/working/
!python /kaggle/working/kaggle_hunyuan3d_airplane_smoke_test.py
```

If the dataset path is not auto-detected:

```bash
%env SHAPENET_RENDERING_ROOT=/kaggle/input/<your-shapenet-rendering-folder>
!python /kaggle/working/kaggle_hunyuan3d_airplane_smoke_test.py
```

Lower memory settings:

```bash
%env HY3D_STEPS=20
%env HY3D_OCTREE_RESOLUTION=128
!python /kaggle/working/kaggle_hunyuan3d_airplane_smoke_test.py
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