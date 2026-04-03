# UMAP Embedding Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create UMAP visualizations to compare how ground_color (large area) and anchor_color (small area) perturbations affect CLIP image vs text embeddings, validating that visual area impacts image embedding distance but not text embedding distance.

**Architecture:** Three-stage pipeline: (1) Match samples with both perturbation types, (2) Extract CLIP embeddings using CLS/EOS tokens, (3) Generate UMAP plots + quantitative distance metrics.

**Tech Stack:** Python, UMAP-learn, matplotlib, PyTorch, HuggingFace Transformers (CLIP)

---

## File Structure

```
scripts/
├── match_paired_samples.py    # Find samples with both factor types
├── extract_embeddings.py      # Extract CLIP embeddings (CLS/EOS tokens)
├── plot_umap_analysis.py      # Generate UMAP visualization + metrics

results/
├── matched_samples.json       # Output of Task 1
├── embeddings.npz             # Output of Task 2
├── umap_analysis.png          # Output of Task 3
└── distance_metrics.json      # Output of Task 3
```

---

## TODOs

### Task 1: Match Samples with Both Perturbation Types

**Files:**
- Create: `scripts/match_paired_samples.py`
- Read: `/root/autodl-tmp/dataset/composition/controlled_pairs/train.jsonl`
- Output: `results/matched_samples.json`

- [ ] **Step 1: Create the matching script**

```python
#!/usr/bin/env python3
"""Find samples that have both ground_color and anchor_color perturbations."""

import json
from pathlib import Path
from collections import defaultdict

def load_data(path: str) -> list:
    """Load JSONL data from controlled_pairs dataset."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def build_original_key(sample: dict) -> str:
    """Build a key to identify the original image configuration.
    
    Uses metadata that should be identical between same-original samples:
    - anchor_color, anchor_shape, ref_color, ref_shape, relation
    """
    meta = sample.get("meta", {})
    # Use fields that define the base scene (excluding ground_color/wall_color)
    key_parts = [
        meta.get("anchor_color", ""),
        meta.get("anchor_shape", ""),
        meta.get("ref_color", ""),
        meta.get("ref_shape", ""),
        meta.get("relation", ""),
    ]
    return "|".join(key_parts)

def match_samples(samples: list, n_per_group: int = 10) -> list:
    """Find groups where same original has both ground_color and anchor_color."""
    
    # Group samples by original configuration and factor type
    groups = defaultdict(lambda: {"ground_color": [], "anchor_color": []})
    
    for sample in samples:
        factor = sample.get("factor")
        if factor not in ["ground_color", "anchor_color"]:
            continue
        
        key = build_original_key(sample)
        groups[key][factor].append(sample)
    
    # Find groups with both types
    matched_groups = []
    for key, factors in groups.items():
        if factors["ground_color"] and factors["anchor_color"]:
            # Take first of each type
            ground_sample = factors["ground_color"][0]
            anchor_sample = factors["anchor_color"][0]
            
            # Verify they share the same original image path
            if ground_sample.get("image") == anchor_sample.get("image"):
                matched_groups.append({
                    "original_key": key,
                    "original_image": ground_sample["image"],
                    "original_caption": ground_sample["caption"],
                    "ground_color_perturbed": ground_sample["image_plus"],
                    "ground_color_caption_perturbed": ground_sample["caption_plus"],
                    "ground_color_changed": f"{ground_sample['original_val']}->{ground_sample['perturbed_val']}",
                    "anchor_color_perturbed": anchor_sample["image_plus"],
                    "anchor_color_caption_perturbed": anchor_sample["caption_plus"],
                    "anchor_color_changed": f"{anchor_sample['original_val']}->{anchor_sample['perturbed_val']}",
                })
        
        if len(matched_groups) >= n_per_group:
            break
    
    return matched_groups

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/root/autodl-tmp/dataset/composition/controlled_pairs/train.jsonl")
    parser.add_argument("--output_path", default="results/matched_samples.json")
    parser.add_argument("--n_groups", type=int, default=10)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}...")
    samples = load_data(args.data_path)
    print(f"Loaded {len(samples)} samples")
    
    print(f"Finding matched groups (need both ground_color and anchor_color)...")
    matched = match_samples(samples, n_per_group=args.n_groups)
    print(f"Found {len(matched)} matched groups")
    
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(matched, f, indent=2)
    
    # Print sample for verification
    if matched:
        print("\nSample matched group:")
        print(json.dumps(matched[0], indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the matching script**

```bash
cd /root/CLIP_tangent && python scripts/match_paired_samples.py
```

Expected output: "Found 10 matched groups" + sample JSON

- [ ] **Step 3: Verify output file exists and has correct structure**

```bash
cat results/matched_samples.json | python -c "import json,sys; d=json.load(sys.stdin); print(f'Groups: {len(d)}'); print(f'Keys: {list(d[0].keys())}')"
```

Expected: Groups: 10, Keys showing original/ground_color/anchor_color fields

---

### Task 2: Extract CLIP Embeddings with CLS/EOS Tokens

**Files:**
- Create: `scripts/extract_embeddings.py`
- Read: `results/matched_samples.json` (from Task 1)
- Read: `/root/autodl-tmp/model/clip_vit_b/` (CLIP model)
- Output: `results/embeddings.npz`

- [ ] **Step 1: Create the embedding extraction script**

```python
#!/usr/bin/env python3
"""Extract CLIP embeddings using CLS token (image) and EOS token (text)."""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

def load_clip_model(model_path: str):
    """Load CLIP model and processor."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor

def get_transform():
    """Standard CLIP image transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

@torch.no_grad()
def extract_image_embedding(model, image_path: str, transform, device) -> np.ndarray:
    """Extract CLS token embedding from image."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get vision model output
    vision_outputs = model.vision_model(image_tensor)
    # CLS token is at position 0
    cls_embedding = vision_outputs.last_hidden_state[:, 0, :]
    # Apply projection
    projected = model.visual_projection(cls_embedding)
    
    return projected.cpu().numpy().squeeze()

@torch.no_grad()
def extract_text_embedding(model, processor, text: str, device) -> np.ndarray:
    """Extract EOS token embedding from text."""
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get text model output
    text_outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Find EOS token position (last non-padding token)
    # For CLIP, we use the last token position
    seq_length = input_ids.shape[1]
    eos_embedding = text_outputs.last_hidden_state[:, seq_length - 1, :]
    # Apply projection
    projected = model.text_projection(eos_embedding)
    
    return projected.cpu().numpy().squeeze()

def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """L2 normalize embedding."""
    return emb / np.linalg.norm(emb)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/clip_vit_b")
    parser.add_argument("--matched_path", default="results/matched_samples.json")
    parser.add_argument("--image_root", default="/root/autodl-tmp/dataset/composition/controlled_pairs")
    parser.add_argument("--output_path", default="results/embeddings.npz")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading CLIP model from {args.model_path}...")
    model, processor = load_clip_model(args.model_path)
    model = model.to(device)
    transform = get_transform()
    
    # Load matched samples
    with open(args.matched_path) as f:
        matched_groups = json.load(f)
    print(f"Processing {len(matched_groups)} matched groups...")
    
    # Storage
    image_embeddings = {}  # {group_idx: {original, ground, anchor}}
    text_embeddings = {}
    metadata = []
    
    image_root = Path(args.image_root)
    
    for idx, group in enumerate(matched_groups):
        print(f"  Processing group {idx + 1}/{len(matched_groups)}")
        
        # Image paths
        orig_img = image_root / group["original_image"]
        ground_img = image_root / group["ground_color_perturbed"]
        anchor_img = image_root / group["anchor_color_perturbed"]
        
        # Extract image embeddings (CLS token)
        img_orig = normalize_embedding(extract_image_embedding(model, str(orig_img), transform, device))
        img_ground = normalize_embedding(extract_image_embedding(model, str(ground_img), transform, device))
        img_anchor = normalize_embedding(extract_image_embedding(model, str(anchor_img), transform, device))
        
        image_embeddings[idx] = {
            "original": img_orig,
            "ground_color": img_ground,
            "anchor_color": img_anchor,
        }
        
        # Extract text embeddings (EOS token)
        txt_orig = normalize_embedding(extract_text_embedding(model, processor, group["original_caption"], device))
        txt_ground = normalize_embedding(extract_text_embedding(model, processor, group["ground_color_caption_perturbed"], device))
        txt_anchor = normalize_embedding(extract_text_embedding(model, processor, group["anchor_color_caption_perturbed"], device))
        
        text_embeddings[idx] = {
            "original": txt_orig,
            "ground_color": txt_ground,
            "anchor_color": txt_anchor,
        }
        
        metadata.append({
            "original_caption": group["original_caption"],
            "ground_color_changed": group["ground_color_changed"],
            "anchor_color_changed": group["anchor_color_changed"],
        })
    
    # Save embeddings
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        metadata=metadata,
    )
    
    print(f"Saved embeddings to {output_path}")
    print(f"  - {len(image_embeddings)} groups of image embeddings")
    print(f"  - Each has: original, ground_color, anchor_color")
    print(f"  - Embedding dim: {img_orig.shape}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Install UMAP-learn dependency**

```bash
pip install umap-learn matplotlib
```

- [ ] **Step 3: Run embedding extraction**

```bash
cd /root/CLIP_tangent && python scripts/extract_embeddings.py
```

Expected: "Saved embeddings to results/embeddings.npz" with embedding dimensions

---

### Task 3: Generate UMAP Visualization + Distance Metrics

**Files:**
- Create: `scripts/plot_umap_analysis.py`
- Read: `results/embeddings.npz` (from Task 2)
- Output: `results/umap_analysis.png`
- Output: `results/distance_metrics.json`

- [ ] **Step 1: Create the UMAP plotting script**

```python
#!/usr/bin/env python3
"""Generate UMAP visualization comparing ground_color vs anchor_color perturbations."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_embeddings(path: str):
    """Load embeddings from npz file."""
    data = np.load(path, allow_pickle=True)
    image_embeddings = data["image_embeddings"].item()
    text_embeddings = data["text_embeddings"].item()
    metadata = data["metadata"]
    return image_embeddings, text_embeddings, metadata

def compute_distances(embeddings: dict, perturbation_type: str) -> list:
    """Compute cosine distance between original and perturbed embeddings."""
    distances = []
    for idx, embs in embeddings.items():
        orig = embs["original"]
        perturbed = embs[perturbation_type]
        # Cosine distance = 1 - cosine similarity
        cosine_sim = np.dot(orig, perturbed) / (np.linalg.norm(orig) * np.linalg.norm(perturbed))
        distances.append(1 - cosine_sim)
    return np.array(distances)

def prepare_umap_data(embeddings: dict):
    """Prepare data for UMAP: collect all embeddings with labels."""
    all_embeddings = []
    labels = []  # "original", "ground_color", "anchor_color"
    group_ids = []
    
    for idx, embs in embeddings.items():
        all_embeddings.append(embs["original"])
        labels.append("original")
        group_ids.append(idx)
        
        all_embeddings.append(embs["ground_color"])
        labels.append("ground_color")
        group_ids.append(idx)
        
        all_embeddings.append(embs["anchor_color"])
        labels.append("anchor_color")
        group_ids.append(idx)
    
    return np.array(all_embeddings), labels, group_ids

def run_umap(embeddings: np.ndarray, n_neighbors: int = 5, min_dist: float = 0.1):
    """Run UMAP dimensionality reduction."""
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    return reducer.fit_transform(embeddings)

def plot_umap_comparison(image_2d, text_2d, labels, group_ids, image_distances, text_distances, output_path):
    """Create side-by-side UMAP plots for image and text embeddings."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color mapping
    color_map = {
        "original": "black",
        "ground_color": "red",
        "anchor_color": "blue",
    }
    
    for ax, (emb_2d, title, distances) in [
        (axes[0], "Image Embeddings (CLS Token)", image_distances),
        (axes[1], "Text Embeddings (EOS Token)", text_distances),
    ]:
        # Plot points by label type
        for label_type in ["original", "ground_color", "anchor_color"]:
            mask = [l == label_type for l in labels]
            points = emb_2d[mask]
            ax.scatter(
                points[:, 0], points[:, 1],
                c=color_map[label_type],
                label=f"{label_type} (n={len(points)})",
                alpha=0.6,
                s=80,
                edgecolors="white",
                linewidths=0.5,
            )
        
        # Draw connecting lines from original to perturbed
        unique_groups = set(group_ids)
        for group_id in unique_groups:
            group_mask = [g == group_id for g in group_ids]
            group_labels = [l for l, m in zip(labels, group_mask) if m]
            group_points = emb_2d[group_mask]
            
            if len(group_points) == 3:  # Should have original, ground, anchor
                orig_idx = group_labels.index("original")
                ground_idx = group_labels.index("ground_color")
                anchor_idx = group_labels.index("anchor_color")
                
                # Line from original to ground_color
                ax.annotate(
                    "",
                    xy=group_points[ground_idx],
                    xytext=group_points[orig_idx],
                    arrowprops=dict(arrowstyle="->", color="red", alpha=0.3, lw=1),
                )
                # Line from original to anchor_color
                ax.annotate(
                    "",
                    xy=group_points[anchor_idx],
                    xytext=group_points[orig_idx],
                    arrowprops=dict(arrowstyle="->", color="blue", alpha=0.3, lw=1),
                )
        
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        
        # Add distance info as text
        dist_text = (
            f"Ground color avg dist: {distances['ground_color'].mean():.4f}\n"
            f"Anchor color avg dist: {distances['anchor_color'].mean():.4f}\n"
            f"Ratio (ground/anchor): {distances['ground_color'].mean() / distances['anchor_color'].mean():.2f}"
        )
        ax.text(0.02, 0.98, dist_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle("CLIP Embedding Space: Effect of Visual Area Change on Perturbation Distance", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")

def compute_and_save_metrics(image_embeddings, text_embeddings, output_path):
    """Compute quantitative distance metrics."""
    
    image_distances = {
        "ground_color": compute_distances(image_embeddings, "ground_color"),
        "anchor_color": compute_distances(image_embeddings, "anchor_color"),
    }
    
    text_distances = {
        "ground_color": compute_distances(text_embeddings, "ground_color"),
        "anchor_color": compute_distances(text_embeddings, "anchor_color"),
    }
    
    metrics = {
        "image": {
            "ground_color_mean": float(image_distances["ground_color"].mean()),
            "ground_color_std": float(image_distances["ground_color"].std()),
            "anchor_color_mean": float(image_distances["anchor_color"].mean()),
            "anchor_color_std": float(image_distances["anchor_color"].std()),
            "ratio_ground_over_anchor": float(image_distances["ground_color"].mean() / image_distances["anchor_color"].mean()),
        },
        "text": {
            "ground_color_mean": float(text_distances["ground_color"].mean()),
            "ground_color_std": float(text_distances["ground_color"].std()),
            "anchor_color_mean": float(text_distances["anchor_color"].mean()),
            "anchor_color_std": float(text_distances["anchor_color"].std()),
            "ratio_ground_over_anchor": float(text_distances["ground_color"].mean() / text_distances["anchor_color"].mean()),
        },
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {output_path}")
    
    # Print summary
    print("\n=== Distance Metrics Summary ===")
    print(f"\nImage Embeddings:")
    print(f"  Ground color distance: {metrics['image']['ground_color_mean']:.4f} ± {metrics['image']['ground_color_std']:.4f}")
    print(f"  Anchor color distance: {metrics['image']['anchor_color_mean']:.4f} ± {metrics['image']['anchor_color_std']:.4f}")
    print(f"  Ratio (ground/anchor): {metrics['image']['ratio_ground_over_anchor']:.2f}")
    
    print(f"\nText Embeddings:")
    print(f"  Ground color distance: {metrics['text']['ground_color_mean']:.4f} ± {metrics['text']['ground_color_std']:.4f}")
    print(f"  Anchor color distance: {metrics['text']['anchor_color_mean']:.4f} ± {metrics['text']['anchor_color_std']:.4f}")
    print(f"  Ratio (ground/anchor): {metrics['text']['ratio_ground_over_anchor']:.2f}")
    
    return image_distances, text_distances

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", default="results/embeddings.npz")
    parser.add_argument("--output_image", default="results/umap_analysis.png")
    parser.add_argument("--output_metrics", default="results/distance_metrics.json")
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_path}...")
    image_embeddings, text_embeddings, metadata = load_embeddings(args.embeddings_path)
    print(f"Loaded {len(image_embeddings)} groups")
    
    # Compute metrics first
    image_distances, text_distances = compute_and_save_metrics(
        image_embeddings, text_embeddings, args.output_metrics
    )
    
    # Prepare UMAP data
    print("\nPreparing UMAP data...")
    image_data, image_labels, image_group_ids = prepare_umap_data(image_embeddings)
    text_data, text_labels, text_group_ids = prepare_umap_data(text_embeddings)
    
    # Run UMAP
    print("Running UMAP on image embeddings...")
    image_2d = run_umap(image_data)
    
    print("Running UMAP on text embeddings...")
    text_2d = run_umap(text_data)
    
    # Plot
    print("Generating visualization...")
    plot_umap_comparison(
        image_2d, text_2d,
        image_labels, text_labels,
        image_group_ids, text_group_ids,
        {"ground_color": image_distances["ground_color"], "anchor_color": image_distances["anchor_color"]},
        {"ground_color": text_distances["ground_color"], "anchor_color": text_distances["anchor_color"]},
        args.output_image,
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the UMAP analysis**

```bash
cd /root/CLIP_tangent && python scripts/plot_umap_analysis.py
```

Expected: UMAP visualization saved to results/umap_analysis.png + metrics JSON

- [ ] **Step 3: Verify outputs exist**

```bash
ls -la results/umap_analysis.png results/distance_metrics.json
```

- [ ] **Step 4: View the metrics**

```bash
cat results/distance_metrics.json
```

---

## Verification Checklist

- [ ] Task 1: matched_samples.json has 10 groups with both perturbation types
- [ ] Task 2: embeddings.npz contains 10 groups × 3 types × 2 modalities (60 embeddings)
- [ ] Task 3: umap_analysis.png shows side-by-side plots with arrows
- [ ] Task 3: distance_metrics.json shows ratio > 1.0 for image, ratio ≈ 1.0 for text

---

## Success Criteria

### Quantitative
- [ ] Image space: `ground_color_distance / anchor_color_distance > 1.2` (hypothesis confirmed)
- [ ] Text space: `ground_color_distance / anchor_color_distance ≈ 0.8-1.2` (hypothesis confirmed)

### Visual
- [ ] Image UMAP: Red arrows (ground_color) longer than blue arrows (anchor_color)
- [ ] Text UMAP: Red and blue arrows similar lengths

---

## Commit Strategy

```bash
# After Task 1
git add scripts/match_paired_samples.py results/matched_samples.json
git commit -m "feat: add sample matching for dual-perturbation groups"

# After Task 2
git add scripts/extract_embeddings.py results/embeddings.npz
git commit -m "feat: add CLIP embedding extraction with CLS/EOS tokens"

# After Task 3
git add scripts/plot_umap_analysis.py results/umap_analysis.png results/distance_metrics.json
git commit -m "feat: add UMAP visualization and distance metrics"
```

---

## Run Commands Summary

```bash
# Install dependencies (once)
pip install umap-learn matplotlib

# Run full pipeline
cd /root/CLIP_tangent

# Task 1: Match samples
python scripts/match_paired_samples.py --n_groups 10

# Task 2: Extract embeddings
python scripts/extract_embeddings.py

# Task 3: Generate visualization
python scripts/plot_umap_analysis.py

# View results
ls -la results/
cat results/distance_metrics.json
```
