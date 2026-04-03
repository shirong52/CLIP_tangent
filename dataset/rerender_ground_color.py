"""
rerender_ground_color.py
========================
批量重新渲染 ground_color 因素的图像，修复 left/right 位置关系 bug。

使用批量渲染模式，一次性启动 Blender，大幅提升速度。

用法：
  python dataset/rerender_ground_color.py \
      --out_dir /root/autodl-tmp/dataset/composition/controlled_pairs \
      --blender blender
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def collect_ground_color_tasks(jsonl_path: str, img_dir: Path) -> list[dict]:
    """从 jsonl 文件中收集需要重新渲染的 ground_color 场景。"""
    tasks = []
    
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record.get("factor") != "ground_color":
                continue
            
            pair_id = record["pair_id"]
            orig_path = img_dir / f"{pair_id}_orig.png"
            pert_path = img_dir / f"{pair_id}_pert.png"
            
            tasks.append({
                "scene": record["scene_orig"],
                "out": str(orig_path),
            })
            tasks.append({
                "scene": record["scene_pert"],
                "out": str(pert_path),
            })
    
    return tasks


def run_batch_render(tasks: list[dict], blender_bin: str, render_script: str,
                     batch_size: int = 50, skip_existing: bool = False):
    """批量渲染，每 batch_size 个任务启动一个 Blender 进程。"""
    if skip_existing:
        tasks = [t for t in tasks if not Path(t["out"]).exists()]
    
    total = len(tasks)
    print(f"  需要渲染: {total} 张图像")
    
    for batch_start in range(0, total, batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        batch_end = batch_start + len(batch)
        
        print(f"  渲染批次: {batch_start}-{batch_end}/{total}")
        
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as f:
            json.dump(batch, f)
            batch_json = f.name
        
        try:
            cmd = [
                blender_bin,
                "--background",
                "--python", render_script,
                "--",
                "--batch_json", batch_json,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                print(f"  ⚠ 批次渲染失败")
                print(f"    stderr: {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠ 批次渲染超时")
        finally:
            os.unlink(batch_json)
    
    print(f"  ✓ 渲染完成")


def update_jsonl(jsonl_path: str, output_path: str):
    """更新 jsonl 文件中的 scene_orig 和 scene_pert（修正坐标）。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset.build_controlled_dataset import build_scene_spec, RELATION_TO_OFFSET
    
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record.get("factor") == "ground_color":
                meta = record["meta"]
                orig_val = record["original_val"]
                pert_val = record["perturbed_val"]
                azimuth = record["scene_orig"]["camera"]["azimuth"]
                
                record["scene_orig"] = build_scene_spec(
                    meta["anchor_color"], meta["anchor_shape"],
                    meta["ref_color"], meta["ref_shape"],
                    meta["relation"], meta["background"],
                    camera_azimuth=azimuth,
                    ground_color=orig_val,
                )
                record["scene_pert"] = build_scene_spec(
                    meta["anchor_color"], meta["anchor_shape"],
                    meta["ref_color"], meta["ref_shape"],
                    meta["relation"], meta["background"],
                    camera_azimuth=azimuth,
                    ground_color=pert_val,
                )
            records.append(record)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ 已更新 {len(records)} 条记录 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="批量重新渲染 ground_color 图像")
    parser.add_argument("--out_dir", required=True, help="数据集根目录")
    parser.add_argument("--blender", default="blender", help="Blender 可执行文件路径")
    parser.add_argument("--render_script", default=None, help="blender_render.py 路径")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="每批渲染图像数（默认 50）")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已存在的文件（支持断点续渲）")
    parser.add_argument("--update_jsonl_only", action="store_true",
                        help="只更新 jsonl，不渲染")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    render_script = args.render_script or str(
        Path(__file__).parent / "blender_render.py"
    )
    
    print(f"\n{'='*55}")
    print(f"  重新渲染 ground_color 图像")
    print(f"{'='*55}")
    
    for split in ["train", "test"]:
        jsonl_path = out_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            continue
        
        print(f"\n[{split.upper()}] 更新 jsonl...")
        update_jsonl(str(jsonl_path), str(jsonl_path))
    
    if args.update_jsonl_only:
        print("\n✓ 仅更新 jsonl 模式，完成。")
        return
    
    for split in ["train", "test"]:
        jsonl_path = out_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            continue
        
        print(f"\n[{split.upper()}] 收集场景...")
        tasks = collect_ground_color_tasks(str(jsonl_path), out_dir / "images" / split)
        
        print(f"[{split.upper()}] 开始批量渲染...")
        run_batch_render(
            tasks,
            args.blender,
            render_script,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
        )
    
    print(f"\n{'='*55}")
    print(f"  ✓ 完成！")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
