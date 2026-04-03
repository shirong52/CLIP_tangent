#!/usr/bin/env python3
"""UMAP visualisation for triplet-based embedding analysis.

Each triplet has three points:
  anchor         (●  black)  - reference scene
  floor_pert     (▲  red)    - same objects, floor color changed  [large-area]
  obj_color_pert (▲  blue)   - same floor, object colors changed  [small-area]

Two panels side-by-side: image space (CLS token) and text space (EOS token).

Arrows: anchor → floor_pert (red), anchor → obj_color_pert (blue).

Hypothesis:
  Image space: floor_pert arrow longer than obj_color_pert arrow (ratio > 1.0)
  Text  space: arrows similar length  (ratio ≈ 1.0, because text changes are comparable)
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── data loading ──────────────────────────────────────────────────────────────

def load_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    arrays = {k: data[k] for k in data.files if k != "metadata"}
    metadata = data["metadata"].tolist()
    return arrays, metadata


# ── distance metrics ──────────────────────────────────────────────────────────

def cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise cosine distance for L2-normalised row vectors."""
    return 1.0 - np.einsum("ij,ij->i", a, b)


def compute_distances(arrays: dict) -> dict:
    return {
        "image": {
            "floor_pert":     cosine_dist(arrays["anchor_img"], arrays["floor_pert_img"]),
            "obj_color_pert": cosine_dist(arrays["anchor_img"], arrays["obj_color_pert_img"]),
        },
        "text": {
            "floor_pert":     cosine_dist(arrays["anchor_txt"], arrays["floor_pert_txt"]),
            "obj_color_pert": cosine_dist(arrays["anchor_txt"], arrays["obj_color_pert_txt"]),
        },
    }


def summarise(distances: dict) -> dict:
    metrics = {}
    for modality, d in distances.items():
        metrics[modality] = {}
        for kind, dists in d.items():
            metrics[modality][kind] = {
                "mean": float(dists.mean()),
                "std":  float(dists.std()),
                "min":  float(dists.min()),
                "max":  float(dists.max()),
            }
        fp = metrics[modality]["floor_pert"]["mean"]
        oc = metrics[modality]["obj_color_pert"]["mean"]
        metrics[modality]["ratio_floor_over_obj"] = float(fp / oc) if oc > 0 else None
    return metrics


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(emb: np.ndarray, n_neighbors: int = 10, min_dist: float = 0.1):
    import umap
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=42, metric="cosine",
    )
    return reducer.fit_transform(emb)


def prepare_umap_stack(arrays: dict, modality: str):
    """Stack anchor / floor_pert / obj_color_pert into one matrix for joint UMAP."""
    sfx = "img" if modality == "image" else "txt"
    n = arrays[f"anchor_{sfx}"].shape[0]
    stack = np.concatenate([
        arrays[f"anchor_{sfx}"],
        arrays[f"floor_pert_{sfx}"],
        arrays[f"obj_color_pert_{sfx}"],
    ], axis=0)
    # labels: 0..n-1 = anchor, n..2n-1 = floor_pert, 2n..3n-1 = obj_color_pert
    roles = (["anchor"] * n + ["floor_pert"] * n + ["obj_color_pert"] * n)
    group_ids = list(range(n)) * 3
    return stack, roles, group_ids


# ── plot ──────────────────────────────────────────────────────────────────────

STYLE = {
    "anchor":         dict(color="#2c2c2c", marker="o", s=70,  label="anchor",               zorder=4),
    "floor_pert":     dict(color="#e74c3c", marker="^", s=70,  label="floor_pert (large-area)", zorder=4),
    "obj_color_pert": dict(color="#2980b9", marker="^", s=70,  label="obj_color_pert (small-area)", zorder=4),
}


def _draw_panel(ax, emb2d, roles, group_ids, distances, title):
    n = len(set(group_ids))
    # index map: (group_id, role) -> row in emb2d
    idx = {}
    for i, (g, r) in enumerate(zip(group_ids, roles)):
        idx[(g, r)] = i

    # draw arrows first (behind points)
    for g in range(n):
        orig = emb2d[idx[(g, "anchor")]]
        fp   = emb2d[idx[(g, "floor_pert")]]
        oc   = emb2d[idx[(g, "obj_color_pert")]]
        ax.annotate("", xy=fp,  xytext=orig,
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", alpha=0.3, lw=0.8))
        ax.annotate("", xy=oc,  xytext=orig,
                    arrowprops=dict(arrowstyle="->", color="#2980b9", alpha=0.3, lw=0.8))

    # scatter per role
    for role, style in STYLE.items():
        mask = [r == role for r in roles]
        pts  = emb2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=style["color"], marker=style["marker"], s=style["s"],
                   alpha=0.80, edgecolors="white", linewidths=0.4,
                   label=style["label"], zorder=style["zorder"])

    # info box
    fp_m = distances["floor_pert"].mean()
    oc_m = distances["obj_color_pert"].mean()
    ratio = fp_m / oc_m if oc_m > 0 else float("nan")
    info = (
        f"floor_pert     \u0394cos: {fp_m:.4f} \u00b1 {distances['floor_pert'].std():.4f}\n"
        f"obj_color_pert \u0394cos: {oc_m:.4f} \u00b1 {distances['obj_color_pert'].std():.4f}\n"
        f"ratio (floor / obj): {ratio:.3f}"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(loc="lower right", fontsize=7.5, markerscale=1.1)


def plot(img2d, img_roles, img_gids,
         txt2d, txt_roles, txt_gids,
         distances, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    _draw_panel(axes[0], img2d, img_roles, img_gids,
                distances["image"], "Image Embedding Space (CLS token)")
    _draw_panel(axes[1], txt2d, txt_roles, txt_gids,
                distances["text"],  "Text Embedding Space (EOS token)")
    fig.suptitle(
        "Triplet Analysis: Floor-Color (large-area) vs Object-Color (small-area) Perturbations\n"
        "Anchor → red arrow: floor changed | Anchor → blue arrow: object color changed",
        fontsize=13, y=1.03,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualisation to {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate UMAP triplet plots and cosine-distance metrics."
    )
    parser.add_argument("--embeddings_path", default="results/embeddings.npz")
    parser.add_argument("--output_image",   default="results/umap_analysis.png")
    parser.add_argument("--output_metrics", default="results/distance_metrics.json")
    parser.add_argument("--n_neighbors", type=int,   default=10)
    parser.add_argument("--min_dist",    type=float, default=0.1)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings_path} ...")
    arrays, metadata = load_embeddings(args.embeddings_path)
    n = arrays["anchor_img"].shape[0]
    print(f"  {n} triplets  |  dim={arrays['anchor_img'].shape[1]}")

    # distances
    print("Computing cosine distances ...")
    distances = compute_distances(arrays)
    metrics   = summarise(distances)

    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.output_metrics}")

    print("\n=== Distance Metrics Summary ===")
    for modality, md in metrics.items():
        print(f"\n[{modality.upper()}]")
        for kind in ("floor_pert", "obj_color_pert"):
            d = md[kind]
            print(f"  {kind:20s}  mean={d['mean']:.4f}  std={d['std']:.4f}")
        print(f"  ratio (floor/obj)      = {md['ratio_floor_over_obj']:.4f}")

    # UMAP
    print("\nPreparing UMAP input ...")
    img_stack, img_roles, img_gids = prepare_umap_stack(arrays, "image")
    txt_stack, txt_roles, txt_gids = prepare_umap_stack(arrays, "text")

    print("Running UMAP on image embeddings ...")
    img2d = run_umap(img_stack, args.n_neighbors, args.min_dist)

    print("Running UMAP on text embeddings ...")
    txt2d = run_umap(txt_stack, args.n_neighbors, args.min_dist)

    print("Generating visualisation ...")
    plot(img2d, img_roles, img_gids,
         txt2d, txt_roles, txt_gids,
         distances, args.output_image)

    print("\nDone!")


if __name__ == "__main__":
    main()
