# Tangent-Alignment CLIP Fine-tuning

This folder contains a modular training framework for CLIP fine-tuning with first-order cross-modal alignment.

## What is implemented

- `clip_only`: standard CLIP fine-tuning baseline.
- `clip_tangent`: CLIP loss + cross-modal tangent alignment + weak delta norm regularization.
- `clip_consistency`: CLIP loss + generic within-modality consistency baseline + weak delta norm regularization.

Losses are split into independent functions in `src/losses/clip_losses.py` for clean ablation.

## Project layout

- `configs/`: experiment configs
- `src/data/`: dataset loading
- `src/losses/`: objective functions
- `src/modeling/`: model loading and trainable-layer strategy
- `src/utils/`: config, seed, and metrics
- `src/train.py`: training, evaluation, and entrypoint (module or script)

## Expected dataset format

The code expects records from `build_controlled_dataset.py` (`train.jsonl`, `test.jsonl`) with fields:

- `image`, `image_plus`
- `caption`, `caption_plus`
- `factor`

and images under:

- `images/train/...`
- `images/test/...`

## Default paths

The default config `configs/tangent_clip_vitb_controlled_mini.json` uses:

- model: `/root/autodl-tmp/model/clip_vit_b`
- dataset: `/root/autodl-tmp/dataset/composition/controlled_pairs_mini`

## Run

```bash
cd /root/CLIP_tangent/train_tangent_clip
python -m src.train --config configs/tangent_clip_vitb_controlled_mini.json
```

## Tips

- If your dataset generation is still running, wait until both JSONL files and image folders are complete.
- For a frozen/near-frozen baseline, set:
  - `"trainable_strategy": "head_only"`
- For full fine-tuning, set:
  - `"trainable_strategy": "full"`
- For ablation:
  - `"mode": "clip_only"`
  - `"mode": "clip_consistency"`

## Standalone Evaluation (Separated from Training)

A standalone evaluation suite is available under `src/eval/`.

Run entrypoint:

```bash
cd /root/CLIP_tangent/train_tangent_clip
python -m src.eval.run --task <task> --model-path <checkpoint_or_best_model_dir>
```

### 1) Controlled pairs retrieval (your generated dataset)

Default paths already point to:
- `/root/autodl-tmp/dataset/composition/controlled_pairs/test.jsonl`
- `/root/autodl-tmp/dataset/composition/controlled_pairs`

```bash
python -m src.eval.run \
  --task controlled \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model
```

### 2) COCO/Flickr30k retrieval

COCO val2017:

```bash
python -m src.eval.run \
  --task retrieval \
  --retrieval-dataset coco \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model
```

Flickr30k test split:

```bash
python -m src.eval.run \
  --task retrieval \
  --retrieval-dataset flickr30k \
  --flickr-split test \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model
```

### 3) Winoground compositional evaluation

Supports image loading directly from `images.zip` (no manual unzip required):

```bash
python -m src.eval.run \
  --task winoground \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model
```

### 4) SugarCrepe evaluation

If SugarCrepe is available, place annotation json/jsonl under:
- `/root/autodl-tmp/dataset/composition/sugarcrepe`
or pass explicit annotation file with `--sugarcrepe-annotation`.

```bash
python -m src.eval.run \
  --task sugarcrepe \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model \
  --sugarcrepe-annotation /path/to/sugarcrepe_annotations.json
```

### 5) Run all benchmarks together

`all` mode runs:
- controlled pairs
- retrieval on COCO + Flickr30k
- winoground
- sugarcrepe (auto-skip if missing)

```bash
python -m src.eval.run \
  --task all \
  --model-path /root/autodl-tmp/output/tangent_vitb_controlled_mini/best_model \
  --output-json /root/autodl-tmp/output/tangent_vitb_controlled_mini/eval_all.json
```
