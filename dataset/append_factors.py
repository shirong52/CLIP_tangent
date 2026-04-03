"""
append_factors.py
=================
追加新因素到已有数据集，不重跑已有因素。

设计原则：
  - 读取现有 train.jsonl / test.jsonl，找出已有的最大 pair_idx
  - 只生成 --factors 指定的新因素
  - 新记录直接追加（append）到现有 JSONL 文件末尾
  - stats.json 同步更新

用法（追加 ground_color 和 wall_color）：
  python append_factors.py \
      --out_dir /root/autodl-tmp/dataset/composition/controlled_pairs \
      --blender /root/autodl-tmp/blender-3.6.9-linux-x64/blender \
      --factors ground_color wall_color \
      --n_train 2000 \
      --n_test  500 \
      --seed 100

用法（重新生成 color 因素，使其包含 ground_color）：
  python append_factors.py \
      --out_dir /root/autodl-tmp/dataset/composition/controlled_pairs \
      --blender /root/autodl-tmp/blender-3.6.9-linux-x64/blender \
      --factors color \
      --n_train 2000 \
      --n_test  500 \
      --seed 200 \
      --replace
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

# 复用 build_controlled_dataset.py 里的所有生成器和工具函数
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_controlled_dataset import (
    FACTOR_GENERATORS,
    TRAIN_COLORS, TEST_COLORS,
    TRAIN_BACKGROUNDS, TEST_BACKGROUNDS,
    TRAIN_GROUND_COLORS, TEST_GROUND_COLORS,
    TRAIN_WALL_COLORS, TEST_WALL_COLORS,
    generate_ground_color_pair, generate_wall_color_pair,
    generate_color_pair, generate_anchor_color_pair,
)


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def get_max_pair_idx(jsonl_path: Path) -> int:
    """从现有 JSONL 中读出最大的 pair_idx 数字，用于生成不冲突的新 ID。"""
    if not jsonl_path.exists():
        return -1
    max_idx = -1
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r   = json.loads(line)
                pid = r.get("pair_id", "")
                # pair_id 格式：train_relation_00123
                parts = pid.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    max_idx = max(max_idx, int(parts[1]))
            except Exception:
                pass
    return max_idx


def get_existing_factors(jsonl_path: Path) -> set:
    """返回现有 JSONL 中已有的因素名称集合。"""
    factors = set()
    if not jsonl_path.exists():
        return factors
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                factors.add(r.get("factor", ""))
            except Exception:
                pass
    return factors


def render_scene(scene_spec: dict, out_path: str,
                 blender_bin: str, render_script: str,
                 timeout: int = 120) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w",
                                     delete=False, encoding="utf-8") as f:
        json.dump(scene_spec, f)
        tmp_json = f.name
    try:
        cmd = [blender_bin, "--background", "--python", render_script,
               "--", "--scene_json", tmp_json, "--out", out_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"  ⚠ 渲染失败: {out_path}\n    {result.stderr[-300:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ⚠ 渲染超时: {out_path}")
        return False
    finally:
        os.unlink(tmp_json)


# ──────────────────────────────────────────────
# 核心追加逻辑
# ──────────────────────────────────────────────

def _strip_factors(jsonl_path: Path, factors_to_remove: set):
    """Remove all records whose 'factor' field is in factors_to_remove."""
    if not jsonl_path.exists():
        return
    kept = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("factor") not in factors_to_remove:
                kept.append(line)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept))
        if kept:
            f.write("\n")


def append_split(split: str, factors: list, n_per_factor: int,
                 rng: random.Random, out_dir: Path,
                 blender_bin: str, render_script: str,
                 dry_run: bool = False,
                 replace: bool = False) -> list[dict]:

    jsonl_path = out_dir / f"{split}.jsonl"
    img_dir    = out_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)

    # 检查已有因素，防止重复追加
    existing = get_existing_factors(jsonl_path)
    if replace:
        # --replace 模式：先从 JSONL 中删除同名因素的旧记录
        to_remove = [f for f in factors if f in existing]
        if to_remove:
            print(f"  [{split}] --replace 模式：删除旧记录 {to_remove} ...")
            _strip_factors(jsonl_path, set(to_remove))
            existing -= set(to_remove)
        to_add = factors
    else:
        to_add   = [f for f in factors if f not in existing]
        skipped  = [f for f in factors if f in existing]
        if skipped:
            print(f"  ⚠ [{split}] 以下因素已存在，跳过: {skipped}")
    if not to_add:
        print(f"  [{split}] 所有指定因素已存在，无需追加。")
        return []

    # 从已有 pair_idx 最大值开始编号，避免 ID 冲突
    start_idx = get_max_pair_idx(jsonl_path) + 1
    print(f"  [{split}] 起始 pair_idx = {start_idx}")

    # 颜色池和背景池
    color_pool  = TRAIN_COLORS  if split == "train" else TEST_COLORS
    bg_pool     = TRAIN_BACKGROUNDS if split == "train" else TEST_BACKGROUNDS
    ground_pool = TRAIN_GROUND_COLORS if split == "train" else TEST_GROUND_COLORS
    wall_pool   = TRAIN_WALL_COLORS   if split == "train" else TEST_WALL_COLORS

    # 新因素对应的生成器（带额外颜色池参数的包装）
    def gen_ground(rng, cp, bp):
        return generate_ground_color_pair(rng, cp, bp, ground_color_pool=ground_pool)

    def gen_wall(rng, cp, bp):
        return generate_wall_color_pair(rng, cp, bp, wall_color_pool=wall_pool)

    def gen_color(rng, cp, bp):
        # 等概率混合 ref_color 和 anchor_color 两种变体，与原始数据集比例一致
        if rng.random() < 0.5:
            return generate_color_pair(rng, cp, bp, ground_color_pool=ground_pool)
        else:
            return generate_anchor_color_pair(rng, cp, bp, ground_color_pool=ground_pool)

    new_generators = {
        "ground_color": gen_ground,
        "wall_color":   gen_wall,
        "color":        gen_color,
    }

    new_records = []
    pair_idx    = start_idx

    for factor in to_add:
        if factor not in new_generators:
            print(f"  ⚠ 未知因素 '{factor}'，跳过")
            continue

        generator    = new_generators[factor]
        factor_count = 0
        print(f"\n  [{split}] 追加因素: {factor} | 目标: {n_per_factor} 对")

        while factor_count < n_per_factor:
            pair     = generator(rng, color_pool, bg_pool)
            stem     = f"{split}_{factor}_{pair_idx:05d}"
            img_orig = img_dir / f"{stem}_orig.png"
            img_pert = img_dir / f"{stem}_pert.png"

            if not dry_run:
                ok_orig = render_scene(pair["scene_orig"], str(img_orig),
                                       blender_bin, render_script)
                ok_pert = render_scene(pair["scene_pert"], str(img_pert),
                                       blender_bin, render_script)
                if not (ok_orig and ok_pert):
                    pair_idx += 1
                    continue
            else:
                json_dir = out_dir / "scene_specs" / split
                json_dir.mkdir(parents=True, exist_ok=True)
                with open(json_dir / f"{stem}_orig.json", "w") as f:
                    json.dump(pair["scene_orig"], f, indent=2)
                with open(json_dir / f"{stem}_pert.json", "w") as f:
                    json.dump(pair["scene_pert"], f, indent=2)

            record = {
                "pair_id":       f"{split}_{factor}_{pair_idx:05d}",
                "split":         split,
                "factor":        pair["factor"],
                "changed_field": pair["changed_field"],
                "original_val":  pair["original_val"],
                "perturbed_val": pair["perturbed_val"],
                "image":         str(img_orig.relative_to(out_dir)),
                "image_plus":    str(img_pert.relative_to(out_dir)),
                "caption":       pair["caption"],
                "caption_plus":  pair["caption_plus"],
                "meta":          pair["meta"],
                "hard_negatives":[],
                "scene_orig":    pair["scene_orig"],
                "scene_pert":    pair["scene_pert"],
            }
            new_records.append(record)
            factor_count += 1
            pair_idx     += 1

            if factor_count % 100 == 0:
                print(f"    {factor_count}/{n_per_factor} 对完成")

        print(f"  ✓ [{split}/{factor}] 完成 {factor_count} 对")

    # 追加写入 JSONL（不覆盖，mode="a"）
    if new_records:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for r in new_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  ✓ 已追加 {len(new_records)} 条 → {jsonl_path}")

    return new_records


def update_stats(out_dir: Path):
    """重新统计所有因素数量，更新 stats.json。"""
    stats_path = out_dir / "stats.json"
    stats = {}
    for split in ("train", "test"):
        jsonl_path = out_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            continue
        factor_dist = {}
        total = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                fac = r.get("factor", "unknown")
                factor_dist[fac] = factor_dist.get(fac, 0) + 1
                total += 1
        stats[split] = {"total": total, "factor_dist": factor_dist}

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ stats.json 已更新 → {stats_path}")

    # 打印汇总
    print(f"\n{'='*50}")
    print(f"  更新后的数据集统计")
    print(f"{'='*50}")
    for split, stat in stats.items():
        print(f"\n  [{split}] 总计 {stat['total']} 对")
        for factor, cnt in sorted(stat["factor_dist"].items()):
            print(f"    {factor:<15s}: {cnt:>5} 对")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="追加新因素到现有数据集")
    parser.add_argument("--out_dir",  required=True,
                        help="现有数据集根目录（含 train.jsonl / test.jsonl）")
    parser.add_argument("--blender",  default="blender")
    parser.add_argument("--render_script", default=None)
    parser.add_argument("--factors",  nargs="+",
                        default=["ground_color", "wall_color"],
                        help="要追加的新因素列表")
    parser.add_argument("--n_train",  type=int, default=2000)
    parser.add_argument("--n_test",   type=int, default=500)
    parser.add_argument("--seed",     type=int, default=100,
                        help="建议用与原始生成不同的 seed，避免重复样本")
    parser.add_argument("--dry_run",  action="store_true")
    parser.add_argument("--replace",  action="store_true",
                        help="允许删除已有同名因素记录后重新生成（用于修正已有因素）")
    args = parser.parse_args()

    rng         = random.Random(args.seed)
    out_dir     = Path(args.out_dir)
    render_script = args.render_script or str(
        Path(__file__).parent / "blender_render.py"
    )

    print(f"\n{'='*50}")
    print(f"  追加因素: {args.factors}")
    print(f"  数据集目录: {out_dir}")
    print(f"  训练/测试: {args.n_train}/{args.n_test} 对/因素")
    print(f"  干运行: {args.dry_run}")
    print(f"{'='*50}\n")

    for split, n in [("train", args.n_train), ("test", args.n_test)]:
        print(f"\n[{split.upper()}]")
        append_split(
            split         = split,
            factors       = args.factors,
            n_per_factor  = n,
            rng           = rng,
            out_dir       = out_dir,
            blender_bin   = args.blender,
            render_script = render_script,
            dry_run       = args.dry_run,
            replace       = args.replace,
        )

    update_stats(out_dir)

    if args.dry_run:
        print("\n  ⚠ 干运行模式：已生成场景 JSON，未实际渲染。")


if __name__ == "__main__":
    main()