"""
Mesh Normalization, Quantization, Reconstruction, and Error Analysis
--------------------------------------------------------------------
Author: Saravanan S
Date: November 2025

This script performs:
  • Mesh loading (.obj)
  • Normalization (MinMax and Unit-Sphere)
  • Quantization (1024 bins)
  • Reconstruction (Dequantize + Denormalize)
  • Error analysis (MSE, MAE per axis)
  • Visualization (bar chart, histogram)
  • Summary CSV generation
"""

import os
import argparse
from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import csv

EPS = 1e-12  # small constant to avoid divide-by-zero


# ---------- Utility Functions ----------
def load_mesh(path):
    """Load mesh from file using trimesh."""
    mesh = trimesh.load(path, force='mesh')
    if not hasattr(mesh, 'vertices'):
        raise ValueError(f"File {path} didn't produce a valid mesh.")
    return mesh


def save_mesh(vertices, faces, path):
    """Save mesh vertices and faces to a file."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)


# ---------- Normalization ----------
def minmax_normalize(vertices):
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    scale = v_max - v_min
    scale[scale == 0] = EPS
    normalized = (vertices - v_min) / scale
    return normalized, v_min, v_max


def minmax_denormalize(normalized, v_min, v_max):
    return normalized * (v_max - v_min) + v_min


def unit_sphere_normalize(vertices):
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    dists = np.linalg.norm(centered, axis=1)
    max_d = np.max(dists)
    if max_d < EPS:
        max_d = EPS
    normalized = centered / max_d
    return normalized, centroid, max_d


def unit_sphere_denormalize(normalized, centroid, max_d):
    return normalized * max_d + centroid


# ---------- Quantization ----------
def quantize(normalized, n_bins=1024):
    q = np.floor(np.clip(normalized, 0.0, 1.0) * (n_bins - 1)).astype(np.int64)
    return q


def dequantize(q, n_bins=1024):
    return q.astype(np.float64) / (n_bins - 1)


# ---------- Error Metrics ----------
def mse(a, b):
    return np.mean((a - b) ** 2)


def mae(a, b):
    return np.mean(np.abs(a - b))


# ---------- Core Processing ----------
def process_mesh(path, out_dir, bins=1024):
    mesh = load_mesh(path)
    vertices = mesh.vertices.copy()
    faces = mesh.faces if mesh.faces is not None else np.array([], dtype=int).reshape(0, 3)

    name = Path(path).stem
    result = {'name': name, 'n_vertices': len(vertices)}

    os.makedirs(out_dir, exist_ok=True)
    nm_dir = os.path.join(out_dir, 'normalized')
    recon_dir = os.path.join(out_dir, 'reconstructed')
    plots_dir = os.path.join(out_dir, 'plots')
    for d in (nm_dir, recon_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    # ---------- Method A: MinMax ----------
    norm_mm, v_min, v_max = minmax_normalize(vertices)
    q_mm = quantize(norm_mm, n_bins=bins)
    deq_mm = dequantize(q_mm, n_bins=bins)
    recon_mm = minmax_denormalize(deq_mm, v_min, v_max)
    save_mesh(norm_mm, faces, os.path.join(nm_dir, f"{name}_minmax_normalized.ply"))
    save_mesh(recon_mm, faces, os.path.join(recon_dir, f"{name}_minmax_reconstructed.ply"))

    mse_mm = mse(vertices, recon_mm)
    mae_mm = mae(vertices, recon_mm)
    mse_mm_axis = np.mean((vertices - recon_mm) ** 2, axis=0)
    mae_mm_axis = np.mean(np.abs(vertices - recon_mm), axis=0)

    # ---------- Method B: Unit Sphere ----------
    norm_us, centroid, max_d = unit_sphere_normalize(vertices)
    norm_us_01 = (norm_us + 1.0) / 2.0
    q_us = quantize(norm_us_01, n_bins=bins)
    deq_us_01 = dequantize(q_us, n_bins=bins)
    deq_us = deq_us_01 * 2.0 - 1.0
    recon_us = unit_sphere_denormalize(deq_us, centroid, max_d)
    save_mesh(norm_us, faces, os.path.join(nm_dir, f"{name}_unitsphere_normalized.ply"))
    save_mesh(recon_us, faces, os.path.join(recon_dir, f"{name}_unitsphere_reconstructed.ply"))

    mse_us = mse(vertices, recon_us)
    mae_us = mae(vertices, recon_us)
    mse_us_axis = np.mean((vertices - recon_us) ** 2, axis=0)
    mae_us_axis = np.mean(np.abs(vertices - recon_us), axis=0)

    # ---------- Store results ----------
    result.update({
        'methodA_mse': float(mse_mm),
        'methodA_mae': float(mae_mm),
        'methodB_mse': float(mse_us),
        'methodB_mae': float(mae_us),
    })

    # ---------- Plot MSE per-axis ----------
    axes = ['x', 'y', 'z']
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    ax.bar(x - 0.2, mse_mm_axis, width=0.4, label='MinMax')
    ax.bar(x + 0.2, mse_us_axis, width=0.4, label='UnitSphere')
    ax.set_xticks(x)
    ax.set_xticklabels(axes)
    ax.set_ylabel("MSE per axis")
    ax.set_title(f"Reconstruction Error per Axis - {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{name}_mse_axis.png"))
    plt.close()

    # ---------- Plot error histograms ----------
    err_mm = np.linalg.norm(vertices - recon_mm, axis=1)
    err_us = np.linalg.norm(vertices - recon_us, axis=1)

    plt.hist(err_mm, bins=80)
    plt.title(f"Per-Vertex L2 Error (MinMax) - {name}")
    plt.xlabel("L2 Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{name}_hist_mm.png"))
    plt.close()

    plt.hist(err_us, bins=80)
    plt.title(f"Per-Vertex L2 Error (UnitSphere) - {name}")
    plt.xlabel("L2 Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{name}_hist_us.png"))
    plt.close()

    return result


# ---------- Batch Execution ----------
def find_obj_files(folder):
    return sorted([str(p) for p in Path(folder).rglob("*.obj")])


def main(args):
    obj_files = find_obj_files(args.input_dir)
    if len(obj_files) == 0:
        print(f"No OBJ files found in {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    summary = []

    for f in obj_files:
        print(f"\nProcessing: {f}")
        try:
            res = process_mesh(f, os.path.join(args.output_dir, Path(f).stem), bins=args.bins)
            summary.append(res)
            print(f" Done: {res['name']} | Vertices: {res['n_vertices']}")
            print(f"  MinMax: MSE={res['methodA_mse']:.6e} MAE={res['methodA_mae']:.6e}")
            print(f"  UnitSphere: MSE={res['methodB_mse']:.6e} MAE={res['methodB_mae']:.6e}")
        except Exception as e:
            print(f"  Error processing {f}: {e}")

        # Save summary CSV
    csv_path = os.path.join(args.output_dir, "summary_errors.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ['name', 'n_vertices', 'methodA_mse', 'methodA_mae', 'methodB_mse', 'methodB_mae']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary:
            writer.writerow({
                'name': s['name'],
                'n_vertices': s['n_vertices'],
                'methodA_mse': s['methodA_mse'],
                'methodA_mae': s['methodA_mae'],
                'methodB_mse': s['methodB_mse'],
                'methodB_mae': s['methodB_mae']
            })
    print(f"\nSummary written to {csv_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="input_meshes")
    parser.add_argument("--output_dir", type=str, default="output_all")
    parser.add_argument("--bins", type=int, default=1024)
    args = parser.parse_args()
    main(args)