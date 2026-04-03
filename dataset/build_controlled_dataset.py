"""
build_controlled_dataset.py
============================
完全受控的合成配对数据集生成器。

设计原则：
  每对图像 (x, x⁺) 仅改变一个语义因素，其余场景元素完全相同。
  这从根本上解决了跨场景配对的背景噪声问题。

生成策略：
  对于每个因素（关系/颜色/形状），构建一个"模板场景"，
  然后只改变目标因素的值，渲染两张图像，构成一个配对。

用法：
  # 完整生成（训练集 + 测试集）
  python build_controlled_dataset.py \
      --out_dir /root/autodl-tmp/controlled_pairs \
      --blender /path/to/blender \
      --n_train 2000 \
      --n_test  500 \
      --seed 42

  # 仅生成场景 JSON（不渲染，用于调试）
  python build_controlled_dataset.py \
      --out_dir /root/autodl-tmp/controlled_pairs \
      --dry_run

输出结构：
  controlled_pairs/
    images/
      train/  scene_00001_orig.png  scene_00001_pert.png  ...
      test/   ...
    train.jsonl   # 训练配对记录
    test.jsonl    # 测试配对记录
    stats.json    # 统计信息
"""

import argparse
import json
import math
import os
import random
import subprocess
import tempfile
from itertools import product
from pathlib import Path

# ──────────────────────────────────────────────
# 场景元素定义
# ──────────────────────────────────────────────

SHAPES    = ["cube", "sphere", "cylinder", "cone"]
COLORS    = ["red", "blue", "green", "yellow", "purple", "cyan"]
# 四种关系：left/right 通过位置区分，near/far 通过 ref 对象大小区分
RELATIONS   = ["left", "right", "near", "far"]
BACKGROUNDS = ["gray", "white", "beige", "dark"]

# left/right：ref 沿 x 轴偏移，anchor 固定在原点
# 语义：relation="left" 表示 anchor 在 ref 左边，即 anchor.x < ref.x
#       所以 ref.x 应该是正值，让 ref 在右边，anchor 在左边
# near/far：  ref 与 anchor 位置相同（同一点），通过 size 编码距离感
RELATION_TO_OFFSET = {
    "left":  ( 1.5, 0.0),   # ref 在右边，anchor 在左边
    "right": (-1.5, 0.0),    # ref 在左边，anchor 在右边
    "near":  ( 1.0, 0.0),   # 略偏右避免完全重叠，size 会更大
    "far":   ( 1.0, 0.0),   # 位置相同，size 更小
}

# near/far 对应的 ref 对象尺寸（通过大小模拟远近感）
RELATION_TO_REF_SIZE = {
    "left":  1.0,    # 正常大小
    "right": 1.0,
    "near":  1.4,    # 近：ref 看起来更大
    "far":   0.6,    # 远：ref 看起来更小
}

# 互为对立的关系
RELATION_OPPOSITES = {
    "left": "right", "right": "left",
    "near": "far",   "far":   "near",
}

# 训练集和测试集不相交的因素组合划分
# 策略：按颜色组合划分——某些颜色对只出现在训练集，其余只在测试集
TRAIN_COLORS = ["red", "blue", "green", "yellow"]          # 训练集颜色（4种）
TEST_COLORS  = ["purple", "cyan", "orange", "pink"]        # 测试集独占颜色（4种，至少需要3种才能做颜色扰动）

TRAIN_BACKGROUNDS = ["gray", "white"]
TEST_BACKGROUNDS  = ["beige", "dark"]

# 地面颜色池（训练/测试不相交）
TRAIN_GROUND_COLORS = ["gray", "beige", "brown", "green"]
TEST_GROUND_COLORS  = ["sand", "clay", "slate", "moss"]

# 墙面颜色池（训练/测试不相交）
TRAIN_WALL_COLORS   = ["white", "cream", "sky", "mint"]
TEST_WALL_COLORS    = ["rose", "lavender", "peach", "fog"]


# ──────────────────────────────────────────────
# 描述模板
# ──────────────────────────────────────────────

RELATION_TEXT = {
    "left":  "to the left of",
    "right": "to the right of",
    "near":  "larger than",    # 用大小描述远近关系
    "far":   "smaller than",
}

def make_caption(anchor_color, anchor_shape, relation, ref_color, ref_shape, background,
                 ground_color=None, wall_color=None):
    rel_text = RELATION_TEXT[relation]
    bg_text  = {"gray": "a gray surface", "white": "a white surface",
                "beige": "a beige surface", "dark": "a dark surface"}.get(background, f"a {background} surface")
    caption = f"a {anchor_color} {anchor_shape} {rel_text} a {ref_color} {ref_shape} on {bg_text}"
    if ground_color:
        caption += f" with a {ground_color} floor"
    if wall_color:
        caption += f" and a {wall_color} wall"
    return caption


# ──────────────────────────────────────────────
# 场景 JSON 构建
# ──────────────────────────────────────────────

def build_scene_spec(anchor_color, anchor_shape,
                     ref_color, ref_shape,
                     relation, background,
                     resolution=256,
                     camera_azimuth=45.0,
                     ref_size=None,
                     ground_color=None,
                     wall_color=None) -> dict:
    """
    构建单个场景的完整规格。
    anchor 固定在原点，ref 根据 relation 放置。
    near/far 关系通过 ref_size 大小而非位置来编码距离感。
    ground_color：覆盖地面颜色（None 则用 background 决定）
    wall_color：  添加背景墙颜色（None 则无背景墙）
    """
    dx, dy   = RELATION_TO_OFFSET[relation]
    ref_size = ref_size if ref_size is not None else RELATION_TO_REF_SIZE[relation]
    spec = {
        "background":   background,
        "ground_color": ground_color,   # None = 用默认背景色
        "wall_color":   wall_color,     # None = 不渲染墙
        "resolution":   resolution,
        "camera": {
            "distance":  8.0,
            "elevation": 18.0,
            "azimuth":   camera_azimuth,
        },
        "objects": [
            {
                "shape": anchor_shape,
                "color": anchor_color,
                "size":  1.0,
                "x": 0.0, "y": 0.0, "z": 0.0,
            },
            {
                "shape": ref_shape,
                "color": ref_color,
                "size":  ref_size,
                "x": dx, "y": dy, "z": 0.0,
            },
        ],
    }
    return spec


# ──────────────────────────────────────────────
# 配对生成：每次只改变一个因素
# ──────────────────────────────────────────────

def generate_relation_pair(rng, color_pool, bg_pool) -> dict:
    """
    关系扰动对：两张图完全相同，只有空间关系不同。
    anchor 和 ref 的颜色、形状、背景完全一致。
    """
    anchor_color  = rng.choice(color_pool)
    anchor_shape  = rng.choice(SHAPES)
    ref_color     = rng.choice([c for c in color_pool if c != anchor_color])
    ref_shape     = rng.choice(SHAPES)
    background    = rng.choice(bg_pool)
    rel_orig      = rng.choice(RELATIONS)
    rel_pert      = RELATION_OPPOSITES[rel_orig]
    azimuth       = rng.uniform(30, 60)   # 轻微随机视角，增加多样性

    # near/far 关系：orig 和 pert 的 ref_size 互换（大↔小），位置不变
    spec_orig = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color, ref_shape,
                                  rel_orig, background,
                                  camera_azimuth=azimuth,
                                  ref_size=RELATION_TO_REF_SIZE[rel_orig])
    spec_pert = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color, ref_shape,
                                  rel_pert, background,
                                  camera_azimuth=azimuth,
                                  ref_size=RELATION_TO_REF_SIZE[rel_pert])
    return {
        "factor":        "relation",
        "changed_field": "relation",
        "original_val":  rel_orig,
        "perturbed_val": rel_pert,
        "caption":       make_caption(anchor_color, anchor_shape,
                                       rel_orig, ref_color, ref_shape, background),
        "caption_plus":  make_caption(anchor_color, anchor_shape,
                                       rel_pert, ref_color, ref_shape, background),
        "scene_orig":    spec_orig,
        "scene_pert":    spec_pert,
        "meta": {
            "anchor_color": anchor_color, "anchor_shape": anchor_shape,
            "ref_color":    ref_color,    "ref_shape":    ref_shape,
            "background":   background,
        }
    }


def generate_color_pair(rng, color_pool, bg_pool,
                        ground_color_pool=None) -> dict:
    """
    颜色扰动对：只改变 ref 对象的颜色，其余完全相同。
    背景关系固定用 left/right，避免 near/far 的 size 变化干扰颜色因素。

    为保证与 ground_color 因素的场景结构一致（均含地面），
    此处也为两个场景随机分配一个 **固定** ground_color（orig/pert 相同）。
    如此一来，"物体颜色变化" vs "地面颜色变化" 的比较在图像结构上完全对等。
    """
    anchor_color  = rng.choice(color_pool)
    anchor_shape  = rng.choice(SHAPES)
    ref_shape     = rng.choice(SHAPES)
    background    = rng.choice(bg_pool)
    relation      = rng.choice(["left", "right"])   # 固定用位置关系
    azimuth       = rng.uniform(30, 60)

    # 地面颜色：固定（orig 和 pert 相同），只保证场景有地面
    gpool         = ground_color_pool or TRAIN_GROUND_COLORS
    ground_fixed  = rng.choice(gpool)

    # 原始颜色和扰动颜色从 color_pool 中选，确保不同
    # 需要至少3种颜色：anchor色、ref原始色、ref扰动色各不相同
    if len(color_pool) < 3:
        raise ValueError(
            f"颜色扰动需要至少3种颜色，当前 color_pool 只有 {len(color_pool)} 种: {color_pool}\n"
            f"请在 TEST_COLORS 中添加更多颜色。"
        )
    available      = [c for c in color_pool if c != anchor_color]
    ref_color_orig = rng.choice(available)
    ref_color_pert = rng.choice([c for c in available if c != ref_color_orig])

    spec_orig = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color_orig, ref_shape,
                                  relation, background,
                                  camera_azimuth=azimuth,
                                  ground_color=ground_fixed)
    spec_pert = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color_pert, ref_shape,   # 只改 ref 颜色
                                  relation, background,
                                  camera_azimuth=azimuth,
                                  ground_color=ground_fixed)   # 地面不变
    return {
        "factor":        "color",
        "changed_field": "ref_color",
        "original_val":  ref_color_orig,
        "perturbed_val": ref_color_pert,
        "caption":       make_caption(anchor_color, anchor_shape,
                                       relation, ref_color_orig, ref_shape, background,
                                       ground_color=ground_fixed),
        "caption_plus":  make_caption(anchor_color, anchor_shape,
                                       relation, ref_color_pert, ref_shape, background,
                                       ground_color=ground_fixed),
        "scene_orig":    spec_orig,
        "scene_pert":    spec_pert,
        "meta": {
            "anchor_color": anchor_color, "anchor_shape": anchor_shape,
            "ref_shape":    ref_shape,    "relation":     relation,
            "background":   background,   "ground_color": ground_fixed,
        }
    }


def generate_shape_pair(rng, color_pool, bg_pool) -> dict:
    """
    形状扰动对：只改变 ref 对象的形状，其余完全相同。
    背景关系固定用 left/right。
    """
    anchor_color  = rng.choice(color_pool)
    anchor_shape  = rng.choice(SHAPES)
    ref_color     = rng.choice([c for c in color_pool if c != anchor_color])
    background    = rng.choice(bg_pool)
    relation      = rng.choice(["left", "right"])
    azimuth       = rng.uniform(30, 60)

    ref_shape_orig = rng.choice(SHAPES)
    ref_shape_pert = rng.choice([s for s in SHAPES if s != ref_shape_orig])

    spec_orig = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color, ref_shape_orig,
                                  relation, background,
                                  camera_azimuth=azimuth)
    spec_pert = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color, ref_shape_pert,   # 只改 ref 形状
                                  relation, background,
                                  camera_azimuth=azimuth)
    return {
        "factor":        "shape",
        "changed_field": "ref_shape",
        "original_val":  ref_shape_orig,
        "perturbed_val": ref_shape_pert,
        "caption":       make_caption(anchor_color, anchor_shape,
                                       relation, ref_color, ref_shape_orig, background),
        "caption_plus":  make_caption(anchor_color, anchor_shape,
                                       relation, ref_color, ref_shape_pert, background),
        "scene_orig":    spec_orig,
        "scene_pert":    spec_pert,
        "meta": {
            "anchor_color": anchor_color, "anchor_shape": anchor_shape,
            "ref_color":    ref_color,    "relation":     relation,
            "background":   background,
        }
    }


def generate_anchor_color_pair(rng, color_pool, bg_pool,
                               ground_color_pool=None) -> dict:
    """
    anchor 颜色扰动对：只改变 anchor 对象的颜色。

    与 generate_color_pair 相同，为保证与 ground_color 因素的场景结构一致，
    此处也为两个场景随机分配一个固定的 ground_color（orig/pert 相同）。
    """
    anchor_shape  = rng.choice(SHAPES)
    ref_color     = rng.choice(color_pool)
    ref_shape     = rng.choice(SHAPES)
    background    = rng.choice(bg_pool)
    relation      = rng.choice(["left", "right"])
    azimuth       = rng.uniform(30, 60)

    # 地面颜色：固定（orig 和 pert 相同）
    gpool         = ground_color_pool or TRAIN_GROUND_COLORS
    ground_fixed  = rng.choice(gpool)

    if len(color_pool) < 3:
        raise ValueError(
            f"颜色扰动需要至少3种颜色，当前 color_pool 只有 {len(color_pool)} 种: {color_pool}"
        )
    available          = [c for c in color_pool if c != ref_color]
    anchor_color_orig  = rng.choice(available)
    anchor_color_pert  = rng.choice([c for c in available if c != anchor_color_orig])

    spec_orig = build_scene_spec(anchor_color_orig, anchor_shape,
                                  ref_color, ref_shape,
                                  relation, background,
                                  camera_azimuth=azimuth,
                                  ground_color=ground_fixed)
    spec_pert = build_scene_spec(anchor_color_pert, anchor_shape,
                                  ref_color, ref_shape,
                                  relation, background,
                                  camera_azimuth=azimuth,
                                  ground_color=ground_fixed)   # 地面不变
    return {
        "factor":        "color",
        "changed_field": "anchor_color",
        "original_val":  anchor_color_orig,
        "perturbed_val": anchor_color_pert,
        "caption":       make_caption(anchor_color_orig, anchor_shape,
                                       relation, ref_color, ref_shape, background,
                                       ground_color=ground_fixed),
        "caption_plus":  make_caption(anchor_color_pert, anchor_shape,
                                       relation, ref_color, ref_shape, background,
                                       ground_color=ground_fixed),
        "scene_orig":    spec_orig,
        "scene_pert":    spec_pert,
        "meta": {
            "anchor_shape": anchor_shape, "ref_color":  ref_color,
            "ref_shape":    ref_shape,    "relation":   relation,
            "background":   background,   "ground_color": ground_fixed,
        }
    }


# ──────────────────────────────────────────────
# 困难负样本（hard negatives）
# 仅改变一个因素，用于评估时的判别测试
# ──────────────────────────────────────────────

def generate_hard_negatives(pair: dict, rng, color_pool) -> list[dict]:
    """
    给定一个配对，生成所有单因素困难负样本。
    每个负样本只改变一个因素，其余与原始场景完全相同。
    返回的负样本用于评估时的排序测试。
    """
    hard_negs = []
    orig = pair["scene_orig"]
    objs = orig["objects"]
    anchor = objs[0]
    ref    = objs[1]

    # 负样本1：关系取反
    for alt_rel in RELATIONS:
        if alt_rel == pair["meta"].get("relation") or \
           alt_rel == pair.get("original_val"):
            continue
        spec = build_scene_spec(
            anchor["color"], anchor["shape"],
            ref["color"], ref["shape"],
            alt_rel, orig["background"],
            camera_azimuth=orig["camera"]["azimuth"]
        )
        hard_negs.append({
            "type":    "relation_neg",
            "changed": alt_rel,
            "caption": make_caption(anchor["color"], anchor["shape"],
                                     alt_rel, ref["color"], ref["shape"],
                                     orig["background"]),
            "scene":   spec,
        })

    # 负样本2：ref 颜色替换
    for alt_color in color_pool:
        if alt_color == ref["color"]:
            continue
        spec = build_scene_spec(
            anchor["color"], anchor["shape"],
            alt_color, ref["shape"],
            pair["meta"].get("relation", "left"),
            orig["background"],
            camera_azimuth=orig["camera"]["azimuth"]
        )
        hard_negs.append({
            "type":    "color_neg",
            "changed": alt_color,
            "caption": make_caption(anchor["color"], anchor["shape"],
                                     pair["meta"].get("relation", "left"),
                                     alt_color, ref["shape"],
                                     orig["background"]),
            "scene":   spec,
        })

    return hard_negs[:4]   # 每个配对最多保留 4 个困难负样本


# ──────────────────────────────────────────────
# Blender 渲染调用
# ──────────────────────────────────────────────

def render_scene(scene_spec: dict, out_path: str,
                 blender_bin: str, render_script: str,
                 timeout: int = 120) -> bool:
    """
    将 scene_spec 写成临时 JSON，调用 Blender 渲染，输出到 out_path。
    返回是否成功。
    """
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w",
                                     delete=False, encoding="utf-8") as f:
        json.dump(scene_spec, f)
        tmp_json = f.name

    try:
        cmd = [
            blender_bin,
            "--background",
            "--python", render_script,
            "--",
            "--scene_json", tmp_json,
            "--out",        out_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            print(f"  ⚠ 渲染失败: {out_path}")
            print(f"    stderr: {result.stderr[-500:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ⚠ 渲染超时: {out_path}")
        return False
    finally:
        os.unlink(tmp_json)


# ──────────────────────────────────────────────
# 因素5：地面颜色扰动
# 只改变地面颜色，物体、关系、墙面完全相同
# ──────────────────────────────────────────────

def generate_ground_color_pair(rng, color_pool, bg_pool,
                                ground_color_pool=None) -> dict:
    """
    地面颜色扰动对：只改变 ground_color，其余完全相同。
    物体颜色使用 color_pool，地面颜色使用独立的 ground_color_pool。
    """
    # 物体属性
    anchor_color = rng.choice(color_pool)
    anchor_shape = rng.choice(SHAPES)
    ref_color    = rng.choice([c for c in color_pool if c != anchor_color])
    ref_shape    = rng.choice(SHAPES)
    relation     = rng.choice(["left", "right"])
    background   = rng.choice(bg_pool)
    azimuth      = rng.uniform(30, 60)

    # 地面颜色：从独立颜色池中选两种不同的
    gpool = ground_color_pool or TRAIN_GROUND_COLORS
    ground_orig = rng.choice(gpool)
    ground_pert = rng.choice([c for c in gpool if c != ground_orig])

    spec_orig = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color, ref_shape,
                                  relation, background,
                                  camera_azimuth=azimuth,
                                  ground_color=ground_orig)
    spec_pert = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color, ref_shape,
                                  relation, background,
                                  camera_azimuth=azimuth,
                                  ground_color=ground_pert)   # 只改地面颜色
    return {
        "factor":        "ground_color",
        "changed_field": "ground_color",
        "original_val":  ground_orig,
        "perturbed_val": ground_pert,
        "caption":       make_caption(anchor_color, anchor_shape,
                                       relation, ref_color, ref_shape,
                                       background, ground_color=ground_orig),
        "caption_plus":  make_caption(anchor_color, anchor_shape,
                                       relation, ref_color, ref_shape,
                                       background, ground_color=ground_pert),
        "scene_orig":    spec_orig,
        "scene_pert":    spec_pert,
        "meta": {
            "anchor_color": anchor_color, "anchor_shape": anchor_shape,
            "ref_color":    ref_color,    "ref_shape":    ref_shape,
            "relation":     relation,     "background":   background,
        }
    }


# ──────────────────────────────────────────────
# 因素6：墙面颜色扰动
# 添加一面背景墙，只改变墙的颜色
# ──────────────────────────────────────────────

def generate_wall_color_pair(rng, color_pool, bg_pool,
                              wall_color_pool=None) -> dict:
    """
    墙面颜色扰动对：场景中添加一面背景墙，只改变墙的颜色，其余完全相同。
    """
    anchor_color = rng.choice(color_pool)
    anchor_shape = rng.choice(SHAPES)
    ref_color    = rng.choice([c for c in color_pool if c != anchor_color])
    ref_shape    = rng.choice(SHAPES)
    relation     = rng.choice(["left", "right"])
    background   = rng.choice(bg_pool)
    azimuth      = rng.uniform(30, 60)

    wpool     = wall_color_pool or TRAIN_WALL_COLORS
    wall_orig = rng.choice(wpool)
    wall_pert = rng.choice([c for c in wpool if c != wall_orig])

    spec_orig = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color, ref_shape,
                                  relation, background,
                                  camera_azimuth=azimuth,
                                  wall_color=wall_orig)
    spec_pert = build_scene_spec(anchor_color, anchor_shape,
                                  ref_color, ref_shape,
                                  relation, background,
                                  camera_azimuth=azimuth,
                                  wall_color=wall_pert)
    return {
        "factor":        "wall_color",
        "changed_field": "wall_color",
        "original_val":  wall_orig,
        "perturbed_val": wall_pert,
        "caption":       make_caption(anchor_color, anchor_shape,
                                       relation, ref_color, ref_shape,
                                       background, wall_color=wall_orig),
        "caption_plus":  make_caption(anchor_color, anchor_shape,
                                       relation, ref_color, ref_shape,
                                       background, wall_color=wall_pert),
        "scene_orig":    spec_orig,
        "scene_pert":    spec_pert,
        "meta": {
            "anchor_color": anchor_color, "anchor_shape": anchor_shape,
            "ref_color":    ref_color,    "ref_shape":    ref_shape,
            "relation":     relation,     "background":   background,
        }
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

FACTOR_GENERATORS = {
    "relation":     generate_relation_pair,
    "color":        generate_color_pair,
    "shape":        generate_shape_pair,
    "anchor_color": generate_anchor_color_pair,
    "ground_color": generate_ground_color_pair,
    "wall_color":   generate_wall_color_pair,
}


def build_split(split: str, n_per_factor: int, rng: random.Random,
                out_dir: Path, blender_bin: str, render_script: str,
                dry_run: bool = False, add_hard_negatives: bool = True) -> list[dict]:
    """
    生成一个 split（train 或 test）的全部配对。
    """
    # 根据 split 选择不同的颜色和背景池（保证训练/测试组合不相交）
    color_pool = TRAIN_COLORS if split == "train" else TEST_COLORS
    bg_pool    = TRAIN_BACKGROUNDS if split == "train" else TEST_BACKGROUNDS

    img_dir = out_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)

    records  = []
    pair_idx = 0

    for factor, generator in FACTOR_GENERATORS.items():
        print(f"\n  [{split}] 因素: {factor} | 目标: {n_per_factor} 对")
        factor_count = 0

        while factor_count < n_per_factor:
            pair = generator(rng, color_pool, bg_pool)

            # 文件名
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
                # dry_run：只写 JSON，不渲染
                json_dir = out_dir / "scene_specs" / split
                json_dir.mkdir(parents=True, exist_ok=True)
                with open(json_dir / f"{stem}_orig.json", "w") as f:
                    json.dump(pair["scene_orig"], f, indent=2)
                with open(json_dir / f"{stem}_pert.json", "w") as f:
                    json.dump(pair["scene_pert"], f, indent=2)

            # 困难负样本（仅记录 JSON，不渲染——评估时按需渲染）
            hard_negs = []
            if add_hard_negatives:
                hard_negs = generate_hard_negatives(pair, rng, color_pool)

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
                "hard_negatives":hard_negs,
                # 保存场景规格，方便复现
                "scene_orig":    pair["scene_orig"],
                "scene_pert":    pair["scene_pert"],
            }
            records.append(record)
            factor_count += 1
            pair_idx     += 1

            if factor_count % 100 == 0:
                print(f"    {factor_count}/{n_per_factor} 对完成")

        print(f"  ✓ [{split}/{factor}] 完成 {factor_count} 对")

    return records


def write_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  已写出 {len(records)} 条 → {path}")


def main():
    parser = argparse.ArgumentParser(description="完全受控的合成配对数据集生成器")
    parser.add_argument("--out_dir",  required=True,
                        help="输出根目录")
    parser.add_argument("--blender",  default="blender",
                        help="Blender 可执行文件路径（默认：blender，需在 PATH 中）")
    parser.add_argument("--render_script", default=None,
                        help="blender_render.py 路径（默认：与本脚本同目录）")
    parser.add_argument("--n_train",  type=int, default=2000,
                        help="训练集每因素对数（总量 = n_train × 因素数）")
    parser.add_argument("--n_test",   type=int, default=500,
                        help="测试集每因素对数")
    parser.add_argument("--resolution", type=int, default=256,
                        help="渲染分辨率（默认 256×256）")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--dry_run",  action="store_true",
                        help="只生成场景 JSON，不实际渲染（调试用）")
    parser.add_argument("--factors",  nargs="+",
                        default=["relation", "color", "shape", "anchor_color"],
                        help="要生成的扰动因素列表")
    parser.add_argument("--no_hard_negatives", action="store_true",
                        help="不生成困难负样本")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_script = args.render_script or str(
        Path(__file__).parent / "blender_render.py"
    )

    # 把 resolution 写入各 generator 默认值（通过全局变量传递）
    # 实际通过 build_scene_spec 的 resolution 参数传入
    global _RESOLUTION
    _RESOLUTION = args.resolution

    # 过滤因素生成器
    global FACTOR_GENERATORS
    FACTOR_GENERATORS = {k: v for k, v in FACTOR_GENERATORS.items()
                         if k in args.factors}

    print(f"\n{'='*55}")
    print(f"  完全受控合成数据集生成器")
    print(f"{'='*55}")
    print(f"  输出目录: {out_dir}")
    print(f"  因素:     {list(FACTOR_GENERATORS.keys())}")
    print(f"  训练/测试: {args.n_train}/{args.n_test} 对/因素")
    print(f"  分辨率:   {args.resolution}×{args.resolution}")
    print(f"  干运行:   {args.dry_run}")
    if not args.dry_run:
        print(f"  Blender:  {args.blender}")
    print()

    all_stats = {}

    for split, n in [("train", args.n_train), ("test", args.n_test)]:
        print(f"\n[{split.upper()}]")
        records = build_split(
            split             = split,
            n_per_factor      = n,
            rng               = rng,
            out_dir           = out_dir,
            blender_bin       = args.blender,
            render_script     = render_script,
            dry_run           = args.dry_run,
            add_hard_negatives= not args.no_hard_negatives,
        )

        jsonl_path = out_dir / f"{split}.jsonl"
        write_jsonl(records, jsonl_path)

        # 统计
        factor_dist = {}
        for r in records:
            factor_dist[r["factor"]] = factor_dist.get(r["factor"], 0) + 1
        all_stats[split] = {
            "total":       len(records),
            "factor_dist": factor_dist,
        }

    # 写出统计
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # 汇总打印
    print(f"\n{'='*55}")
    print(f"  生成完成！")
    for split, stat in all_stats.items():
        print(f"\n  [{split}] 总计 {stat['total']} 对")
        for factor, cnt in stat["factor_dist"].items():
            print(f"    {factor:<15s}: {cnt:>5} 对")
    print(f"\n  输出目录: {out_dir.resolve()}")

    if args.dry_run:
        print("\n  ⚠ 干运行模式：已生成场景 JSON，未实际渲染。")
        print("    确认场景规格正确后，去掉 --dry_run 重新运行。")


if __name__ == "__main__":
    main()