"""
Measure quality metrics for generated 3D meshes.

Basic usage:
    python mesh_quality_metrics.py path/to/model.glb

Optional reference metrics:
    python mesh_quality_metrics.py generated.glb --reference ground_truth.obj

The script prints human-readable metrics and can also save JSON:
    python mesh_quality_metrics.py model.glb --json-out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: trimesh. Install it with `pip install trimesh scipy`."
        ) from exc
    return trimesh


def load_as_mesh(path: Path):
    trimesh = require_trimesh()
    loaded = trimesh.load(path, force=None)

    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for geometry in loaded.geometry.values():
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0:
                meshes.append(geometry)
        if not meshes:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        return trimesh.util.concatenate(meshes)

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported geometry type in {path}: {type(loaded)!r}")
    if len(loaded.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    return loaded


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def edge_manifold_metrics(mesh) -> dict[str, Any]:
    edges = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.sum(counts == 1))
    non_manifold_edges = int(np.sum(counts > 2))
    total_unique_edges = int(len(unique_edges))
    return {
        "unique_edges": total_unique_edges,
        "boundary_edges": boundary_edges,
        "boundary_edge_ratio": boundary_edges / total_unique_edges if total_unique_edges else None,
        "non_manifold_edges": non_manifold_edges,
        "non_manifold_edge_ratio": non_manifold_edges / total_unique_edges if total_unique_edges else None,
    }


def face_quality_metrics(mesh) -> dict[str, Any]:
    areas = np.asarray(mesh.area_faces)
    total_faces = int(len(mesh.faces))
    if total_faces == 0:
        return {
            "degenerate_faces": 0,
            "degenerate_face_ratio": None,
            "face_area_min": None,
            "face_area_median": None,
            "face_area_max": None,
        }

    eps = max(float(np.nanmedian(areas)) * 1e-8, 1e-18)
    degenerate = int(np.sum(areas <= eps))
    return {
        "degenerate_faces": degenerate,
        "degenerate_face_ratio": degenerate / total_faces,
        "face_area_min": safe_float(np.min(areas)),
        "face_area_median": safe_float(np.median(areas)),
        "face_area_max": safe_float(np.max(areas)),
    }


def component_metrics(mesh) -> dict[str, Any]:
    components = mesh.split(only_watertight=False)
    face_counts = sorted((int(len(component.faces)) for component in components), reverse=True)
    total_faces = int(len(mesh.faces))
    largest_ratio = face_counts[0] / total_faces if total_faces and face_counts else None
    small_components = int(sum(1 for count in face_counts[1:] if total_faces and count / total_faces < 0.01))
    return {
        "components": int(len(components)),
        "largest_component_face_ratio": largest_ratio,
        "small_components_under_1pct": small_components,
        "component_face_counts_top5": face_counts[:5],
    }


def duplicate_vertex_metrics(mesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        return {"duplicate_vertices_estimate": 0, "duplicate_vertex_ratio": None}

    scale = max(float(np.linalg.norm(mesh.extents)), 1.0)
    rounded = np.round(vertices / (scale * 1e-8)).astype(np.int64)
    unique_count = len(np.unique(rounded, axis=0))
    duplicates = int(len(vertices) - unique_count)
    return {
        "duplicate_vertices_estimate": duplicates,
        "duplicate_vertex_ratio": duplicates / len(vertices),
    }


def symmetry_score(mesh, axis: int, samples: int, seed: int) -> float | None:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None

    trimesh = require_trimesh()
    if len(mesh.faces) == 0:
        return None

    sample_count = min(samples, max(100, int(len(mesh.faces) * 4)))
    rng = np.random.default_rng(seed)
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, sample_count)
    finally:
        np.random.set_state(state)

    center = np.asarray(mesh.bounding_box.centroid)
    mirrored = points.copy()
    mirrored[:, axis] = 2.0 * center[axis] - mirrored[:, axis]
    distances, _ = cKDTree(points).query(mirrored, k=1)
    diagonal = float(np.linalg.norm(mesh.extents))
    if diagonal <= 0:
        return None
    return safe_float(np.mean(distances) / diagonal)


def intrinsic_metrics(mesh, symmetry_samples: int, seed: int) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds)
    extents = np.asarray(mesh.extents)
    diagonal = float(np.linalg.norm(extents))
    is_watertight = bool(mesh.is_watertight)
    euler_number = int(mesh.euler_number)
    genus = None
    if is_watertight and euler_number <= 2:
        genus = int((2 - euler_number) / 2)

    metrics: dict[str, Any] = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": is_watertight,
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": euler_number,
        "genus_if_watertight": genus,
        "surface_area": safe_float(mesh.area),
        "volume_if_watertight": safe_float(mesh.volume) if is_watertight else None,
        "bounding_box_min": [safe_float(v) for v in bounds[0]],
        "bounding_box_max": [safe_float(v) for v in bounds[1]],
        "bounding_box_extents": [safe_float(v) for v in extents],
        "bounding_box_diagonal": safe_float(diagonal),
        "aspect_ratio_max_to_min_extent": safe_float(
            np.max(extents) / max(np.min(extents), 1e-12)
        ),
    }
    metrics.update(component_metrics(mesh))
    metrics.update(edge_manifold_metrics(mesh))
    metrics.update(face_quality_metrics(mesh))
    metrics.update(duplicate_vertex_metrics(mesh))

    axis_names = ("x", "y", "z")
    for axis, name in enumerate(axis_names):
        metrics[f"symmetry_score_{name}_axis"] = symmetry_score(mesh, axis, symmetry_samples, seed + axis)
    return metrics


def sample_points_and_normals(mesh, count: int, seed: int):
    trimesh = require_trimesh()
    rng = np.random.default_rng(seed)
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        points, face_index = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals)[face_index]
    return points, normals


def reference_metrics(generated, reference, samples: int, threshold: float, seed: int) -> dict[str, Any]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise SystemExit(
            "Reference metrics need scipy. Install it with `pip install scipy`."
        ) from exc

    generated_points, generated_normals = sample_points_and_normals(generated, samples, seed)
    reference_points, reference_normals = sample_points_and_normals(reference, samples, seed + 1)

    reference_tree = cKDTree(reference_points)
    generated_tree = cKDTree(generated_points)

    gen_to_ref_dist, gen_to_ref_idx = reference_tree.query(generated_points, k=1)
    ref_to_gen_dist, ref_to_gen_idx = generated_tree.query(reference_points, k=1)

    chamfer_l2 = float(np.mean(gen_to_ref_dist**2) + np.mean(ref_to_gen_dist**2))
    chamfer_l1 = float(np.mean(gen_to_ref_dist) + np.mean(ref_to_gen_dist))
    precision = float(np.mean(gen_to_ref_dist < threshold))
    recall = float(np.mean(ref_to_gen_dist < threshold))
    f_score = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))

    matched_reference_normals = reference_normals[gen_to_ref_idx]
    normal_cos = np.abs(np.sum(generated_normals * matched_reference_normals, axis=1))

    return {
        "reference_samples": int(samples),
        "reference_threshold": float(threshold),
        "chamfer_l1": chamfer_l1,
        "chamfer_l2": chamfer_l2,
        "f_score": f_score,
        "precision": precision,
        "recall": recall,
        "normal_consistency": safe_float(np.mean(normal_cos)),
    }


def quality_score(metrics: dict[str, Any]) -> dict[str, Any]:
    score = 100.0
    penalties: list[str] = []

    if not metrics["is_watertight"]:
        score -= 15
        penalties.append("mesh is not watertight")
    if not metrics["is_winding_consistent"]:
        score -= 10
        penalties.append("face winding is inconsistent")

    boundary_ratio = metrics.get("boundary_edge_ratio") or 0.0
    non_manifold_ratio = metrics.get("non_manifold_edge_ratio") or 0.0
    degenerate_ratio = metrics.get("degenerate_face_ratio") or 0.0
    duplicate_ratio = metrics.get("duplicate_vertex_ratio") or 0.0
    largest_component_ratio = metrics.get("largest_component_face_ratio") or 1.0

    score -= min(20.0, boundary_ratio * 100)
    score -= min(25.0, non_manifold_ratio * 250)
    score -= min(15.0, degenerate_ratio * 300)
    score -= min(10.0, duplicate_ratio * 100)
    score -= min(20.0, (1.0 - largest_component_ratio) * 50)

    if boundary_ratio > 0.02:
        penalties.append("many boundary edges")
    if non_manifold_ratio > 0.005:
        penalties.append("non-manifold edges detected")
    if degenerate_ratio > 0.01:
        penalties.append("many degenerate faces")
    if largest_component_ratio < 0.95:
        penalties.append("mesh has significant disconnected components")

    return {
        "quality_score_0_100": round(max(0.0, min(100.0, score)), 2),
        "quality_warnings": penalties,
    }


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def print_metrics(metrics: dict[str, Any]) -> None:
    groups = [
        (
            "Geometry",
            [
                "vertices",
                "faces",
                "surface_area",
                "volume_if_watertight",
                "bounding_box_extents",
                "bounding_box_diagonal",
                "aspect_ratio_max_to_min_extent",
            ],
        ),
        (
            "Topology",
            [
                "is_watertight",
                "is_winding_consistent",
                "euler_number",
                "genus_if_watertight",
                "components",
                "largest_component_face_ratio",
                "boundary_edges",
                "boundary_edge_ratio",
                "non_manifold_edges",
                "non_manifold_edge_ratio",
            ],
        ),
        (
            "Mesh Defects",
            [
                "degenerate_faces",
                "degenerate_face_ratio",
                "duplicate_vertices_estimate",
                "duplicate_vertex_ratio",
                "face_area_min",
                "face_area_median",
                "face_area_max",
            ],
        ),
        (
            "Symmetry",
            [
                "symmetry_score_x_axis",
                "symmetry_score_y_axis",
                "symmetry_score_z_axis",
            ],
        ),
        (
            "Reference",
            [
                "chamfer_l1",
                "chamfer_l2",
                "f_score",
                "precision",
                "recall",
                "normal_consistency",
            ],
        ),
        (
            "Overall",
            [
                "quality_score_0_100",
                "quality_warnings",
            ],
        ),
    ]

    for title, keys in groups:
        present = [(key, metrics[key]) for key in keys if key in metrics]
        if not present:
            continue
        print(f"\n{title}:")
        for key, value in present:
            print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute quality metrics for a generated 3D mesh.")
    parser.add_argument("mesh", type=Path, help="Path to generated .glb/.obj/.ply mesh.")
    parser.add_argument("--reference", type=Path, help="Optional ground-truth mesh for Chamfer/F-score.")
    parser.add_argument("--samples", type=int, default=20000, help="Surface samples for reference metrics.")
    parser.add_argument(
        "--symmetry-samples",
        type=int,
        default=8000,
        help="Surface samples for approximate symmetry scores.",
    )
    parser.add_argument(
        "--fscore-threshold",
        type=float,
        default=0.01,
        help="Distance threshold for F-score in mesh units.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--json-out", type=Path, help="Optional path to save a JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mesh.exists():
        raise SystemExit(f"Mesh file does not exist: {args.mesh}")

    mesh = load_as_mesh(args.mesh)
    metrics = {
        "mesh_path": str(args.mesh),
        **intrinsic_metrics(mesh, symmetry_samples=args.symmetry_samples, seed=args.seed),
    }

    if args.reference:
        if not args.reference.exists():
            raise SystemExit(f"Reference file does not exist: {args.reference}")
        reference = load_as_mesh(args.reference)
        metrics["reference_path"] = str(args.reference)
        metrics.update(
            reference_metrics(
                mesh,
                reference,
                samples=args.samples,
                threshold=args.fscore_threshold,
                seed=args.seed,
            )
        )

    metrics.update(quality_score(metrics))
    metrics = make_jsonable(metrics)
    print_metrics(metrics)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON report: {args.json_out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
