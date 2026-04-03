#!/usr/bin/env python3
"""Find triplets from the ground_color factor for clean controlled comparison.

Triplet structure (all three from ground_color factor):
  anchor         (A.orig): original image/caption
  floor_pert     (A.pert): same scene, ONLY floor color changed  → large-area perturbation
  obj_color_pert (B.orig): same scene config + same floor color, ONLY object colors differ → small-area perturbation

Matching criterion:
  A and B share the same (anchor_shape, ref_shape, relation, background, ground_color_orig)
  but have different object colors (anchor_color, ref_color).

  python3 scripts/match_paired_samples.py --n_triplets 5
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_data(path: str) -> list:
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_triplets(samples: list, n_triplets: int = 50) -> list:
    """Find triplets within the ground_color factor.

    Group pairs by (anchor_shape, ref_shape, relation, background, original_val=floor_orig).
    Any group with ≥2 pairs can form a triplet: pick pair A as anchor+floor_pert,
    pick pair B (different object colors) as obj_color_pert.
    """
    gc_pairs = [s for s in samples if s.get("factor") == "ground_color"]

    # Group by scene config + original floor color
    groups = defaultdict(list)
    for p in gc_pairs:
        m = p.get("meta", {})
        key = (
            m.get("anchor_shape", ""),
            m.get("ref_shape", ""),
            m.get("relation", ""),
            m.get("background", ""),
            p["original_val"],   # floor color of orig image
        )
        groups[key].append(p)

    triplets = []
    for key, pairs in groups.items():
        if len(pairs) < 2:
            continue

        anchor_shape, ref_shape, relation, background, floor_orig = key

        # Pick pair A and pair B where ONLY ONE color differs
        # XOR logic: (anchor different) XOR (ref different) = exactly one differs
        a = pairs[0]
        b = next(
            (p for p in pairs[1:]
             if (p["meta"].get("anchor_color") != a["meta"].get("anchor_color"))
                != (p["meta"].get("ref_color") != a["meta"].get("ref_color"))),
            None,
        )
        if b is None:
            continue

        triplets.append({
            "scene_key": f"{anchor_shape}|{ref_shape}|{relation}|{background}|floor={floor_orig}",
            # ── anchor (A.orig) ──────────────────────────────────────────
            "anchor_image":   a["image"],
            "anchor_caption": a["caption"],
            # ── floor perturbation (A.pert) ──────────────────────────────
            # same object colors, only floor color changes
            "floor_pert_image":   a["image_plus"],
            "floor_pert_caption": a["caption_plus"],
            "floor_change": f"{a['original_val']}->{a['perturbed_val']}",
            # ── object-color perturbation (B.orig) ──────────────────────
            # same floor color (= floor_orig), only object colors differ
            "obj_color_pert_image":   b["image"],
            "obj_color_pert_caption": b["caption"],
            "obj_color_change": (
                f"anchor {a['meta'].get('anchor_color')}->{b['meta'].get('anchor_color')}, "
                f"ref {a['meta'].get('ref_color')}->{b['meta'].get('ref_color')}"
            ),
            "meta": {
                "anchor_shape": anchor_shape,
                "ref_shape":    ref_shape,
                "relation":     relation,
                "background":   background,
                "floor_orig":   floor_orig,
                "floor_pert":   a["perturbed_val"],
                "A_anchor_color": a["meta"].get("anchor_color"),
                "A_ref_color":    a["meta"].get("ref_color"),
                "B_anchor_color": b["meta"].get("anchor_color"),
                "B_ref_color":    b["meta"].get("ref_color"),
            },
        })

        if len(triplets) >= n_triplets:
            break

    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Find triplets within ground_color factor for controlled UMAP analysis."
    )
    parser.add_argument(
        "--data_path",
        default="/root/autodl-tmp/dataset/composition/controlled_pairs/train.jsonl",
    )
    parser.add_argument("--output_path", default="results/matched_samples.json")
    parser.add_argument("--n_triplets", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading data from {args.data_path} ...")
    samples = load_data(args.data_path)
    print(f"Loaded {len(samples)} samples")

    print("Finding triplets within ground_color factor ...")
    triplets = build_triplets(samples, n_triplets=args.n_triplets)
    print(f"Found {len(triplets)} triplets")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(triplets, f, indent=2)
    print(f"Saved to {output_path}")

    if triplets:
        print("\nSample triplet:")
        t = triplets[0]
        print(f"  anchor         : {t['anchor_caption']}")
        print(f"  floor_pert     : {t['floor_pert_caption']}  [{t['floor_change']}]")
        print(f"  obj_color_pert : {t['obj_color_pert_caption']}  [{t['obj_color_change']}]")


if __name__ == "__main__":
    main()
