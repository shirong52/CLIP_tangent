"""
blender_render.py
=================
批量渲染模式：一次 Blender 进程处理多个场景，彻底消除进程启动开销。
使用 EEVEE 引擎，比 Cycles 快 10-20 倍。

调用方式：
  blender --background --python blender_render.py -- --batch_json /path/batch.json

batch.json 格式（列表，每项是一个渲染任务）：
[
  {"scene": {...场景规格...}, "out": "/path/to/output.png"},
  {"scene": {...},            "out": "/path/to/output2.png"},
  ...
]

也支持单场景模式（向后兼容）：
  blender --background --python blender_render.py -- --scene_json /path/scene.json --out /path/out.png
"""

import bpy
import sys
import json
import math
import os
from pathlib import Path

# ── 解析命令行参数 ──
argv = sys.argv
try:
    idx  = argv.index("--")
    args = argv[idx + 1:]
except ValueError:
    args = []

def get_arg(flag, default=None):
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return default

BATCH_JSON = get_arg("--batch_json")
SCENE_JSON = get_arg("--scene_json")   # 单场景兼容模式
OUT_PATH   = get_arg("--out")

# ──────────────────────────────────────────────
# 颜色表（线性空间 RGBA）
# ──────────────────────────────────────────────
COLOR_MAP = {
    # 训练集物体颜色
    "red":      (0.8,  0.1,  0.1,  1.0),
    "blue":     (0.1,  0.2,  0.8,  1.0),
    "green":    (0.1,  0.7,  0.2,  1.0),
    "yellow":   (0.9,  0.8,  0.1,  1.0),
    # 测试集物体颜色
    "purple":   (0.6,  0.1,  0.7,  1.0),
    "cyan":     (0.1,  0.7,  0.8,  1.0),
    "orange":   (0.9,  0.4,  0.05, 1.0),
    "pink":     (0.9,  0.4,  0.6,  1.0),
    # 地面颜色（训练集）
    "gray":     (0.45, 0.45, 0.45, 1.0),
    "beige":    (0.76, 0.65, 0.53, 1.0),
    "brown":    (0.50, 0.30, 0.15, 1.0),
    "ground_green": (0.25, 0.50, 0.20, 1.0),
    # 地面颜色（测试集）
    "sand":     (0.82, 0.72, 0.50, 1.0),
    "clay":     (0.68, 0.40, 0.28, 1.0),
    "slate":    (0.35, 0.38, 0.42, 1.0),
    "moss":     (0.30, 0.42, 0.22, 1.0),
    # 墙面颜色（训练集）
    "white":    (0.90, 0.90, 0.90, 1.0),
    "cream":    (0.95, 0.92, 0.80, 1.0),
    "sky":      (0.60, 0.78, 0.92, 1.0),
    "mint":     (0.72, 0.92, 0.82, 1.0),
    # 墙面颜色（测试集）
    "rose":     (0.95, 0.75, 0.78, 1.0),
    "lavender": (0.78, 0.72, 0.92, 1.0),
    "peach":    (0.98, 0.82, 0.68, 1.0),
    "fog":      (0.82, 0.84, 0.88, 1.0),
}

BACKGROUND_MAP = {
    "gray":  (0.45, 0.45, 0.45),
    "white": (0.85, 0.85, 0.85),
    "beige": (0.75, 0.65, 0.55),
    "dark":  (0.12, 0.12, 0.15),
}


# ──────────────────────────────────────────────
# 场景构建函数
# ──────────────────────────────────────────────

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def make_material(name, rgba):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value  = rgba
    bsdf.inputs["Roughness"].default_value   = 0.4
    bsdf.inputs["Specular"].default_value    = 0.3
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def add_object(spec):
    shape = spec["shape"]
    color = spec["color"]
    size  = spec.get("size", 1.0)
    x, y, z = spec["x"], spec["y"], spec.get("z", 0.0)

    if shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=size, location=(x, y, z + size / 2))
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=size / 2, location=(x, y, z + size / 2),
            segments=32, ring_count=16)
        bpy.ops.object.shade_smooth()
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            radius=size / 2, depth=size,
            location=(x, y, z + size / 2), vertices=32)
        bpy.ops.object.shade_smooth()
    elif shape == "cone":
        bpy.ops.mesh.primitive_cone_add(
            radius1=size / 2, depth=size,
            location=(x, y, z + size / 2), vertices=32)
    else:
        raise ValueError(f"未知形状: {shape}")

    obj  = bpy.context.active_object
    rgba = COLOR_MAP.get(color, (0.5, 0.5, 0.5, 1.0))
    mat  = make_material(f"mat_{color}_{shape}_{id(obj)}", rgba)
    obj.data.materials.append(mat)
    return obj


def setup_ground(bg_color, ground_color=None):
    """
    ground_color 优先于 bg_color 决定地面颜色。
    若 ground_color 为 None，则使用 BACKGROUND_MAP 中的背景色。
    """
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground"

    if ground_color and ground_color in COLOR_MAP:
        rgba = COLOR_MAP[ground_color]
    else:
        rgb  = BACKGROUND_MAP.get(bg_color, (0.45, 0.45, 0.45))
        rgba = (*rgb, 1.0)

    mat  = bpy.data.materials.new("mat_ground")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value    = rgba
    bsdf.inputs["Roughness"].default_value = 0.8
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    plane.data.materials.append(mat)


def setup_wall(wall_color):
    """
    在场景背后（y=8）添加一面竖墙，颜色由 wall_color 决定。
    墙足够大（宽16×高8），在当前相机角度下充满画面上半部分。
    """
    if not wall_color:
        return
    # 墙放在场景后方，稍微倾斜朝向相机
    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=(0, 7.0, 4.0)
    )
    wall = bpy.context.active_object
    wall.name = "Wall"
    wall.scale = (8.0, 0.1, 4.5)   # 宽16, 厚0.2, 高9
    bpy.ops.object.transform_apply(scale=True)

    rgba = COLOR_MAP.get(wall_color, (0.9, 0.9, 0.9, 1.0))
    mat  = bpy.data.materials.new("mat_wall")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value    = rgba
    bsdf.inputs["Roughness"].default_value = 0.7
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    wall.data.materials.append(mat)


def setup_camera(distance, elevation_deg, azimuth_deg):
    import mathutils
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)
    x = distance * math.cos(elev) * math.cos(azim)
    y = distance * math.cos(elev) * math.sin(azim)
    z = distance * math.sin(elev)

    cam_loc    = mathutils.Vector((x, y, z))
    target_loc = mathutils.Vector((0.0, 0.0, 0.5))
    direction  = target_loc - cam_loc
    rot_quat   = direction.to_track_quat("-Z", "Y")

    bpy.ops.object.camera_add(location=(x, y, z))
    cam_obj = bpy.context.active_object
    cam_obj.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = cam_obj
    return cam_obj


def setup_lighting():
    world = bpy.context.scene.world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value    = (0.2, 0.2, 0.2, 1.0)
        bg_node.inputs["Strength"].default_value = 0.8

    bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 2.0
    sun.data.angle  = 0.1

    bpy.ops.object.light_add(type="AREA", location=(3, -3, 5))
    area = bpy.context.active_object
    area.data.energy = 80
    area.data.size   = 6


def setup_render_settings(resolution):
    """一次性设置渲染参数，批量模式下只调用一次。"""
    scene = bpy.context.scene

    # EEVEE：比 Cycles 快 10-20 倍
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 16
    scene.eevee.use_soft_shadows   = True
    scene.eevee.shadow_cube_size   = "512"
    scene.eevee.use_gtao           = True

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format  = "PNG"
    scene.render.image_settings.color_mode   = "RGB"
    scene.render.image_settings.color_depth  = "8"

    # 色彩管理
    for transform in ("Filmic", "AgX", "Standard"):
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma    = 1.0
    scene.display_settings.display_device = "sRGB"


def render_one(scene_spec, out_path):
    """渲染单个场景到 out_path。"""
    clear_scene()
    setup_ground(
        scene_spec.get("background", "gray"),
        ground_color=scene_spec.get("ground_color")   # None = 用背景色
    )
    setup_wall(scene_spec.get("wall_color"))           # None = 不加墙
    for obj_spec in scene_spec["objects"]:
        add_object(obj_spec)
    cam = scene_spec.get("camera", {})
    setup_camera(
        distance      = cam.get("distance",  8.0),
        elevation_deg = cam.get("elevation", 18.0),
        azimuth_deg   = cam.get("azimuth",   45.0),
    )
    setup_lighting()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    # ── 批量模式（主路径）──
    if BATCH_JSON:
        with open(BATCH_JSON) as f:
            tasks = json.load(f)

        if not tasks:
            print("batch_json 为空，退出")
            return

        # 用第一个任务的分辨率初始化渲染设置（批量内分辨率相同）
        resolution = tasks[0]["scene"].get("resolution", 256)
        setup_render_settings(resolution)

        total = len(tasks)
        for i, task in enumerate(tasks):
            out_path = task["out"]
            if Path(out_path).exists():
                continue   # 已渲染，跳过（支持断点续渲）
            render_one(task["scene"], out_path)
            if (i + 1) % 10 == 0:
                print(f"  批量进度: {i+1}/{total}")

        print(f"批量渲染完成: {total} 张")

    # ── 单场景兼容模式 ──
    elif SCENE_JSON and OUT_PATH:
        with open(SCENE_JSON) as f:
            scene_spec = json.load(f)
        resolution = scene_spec.get("resolution", 256)
        setup_render_settings(resolution)
        render_one(scene_spec, OUT_PATH)
        print(f"渲染完成: {OUT_PATH}")

    else:
        print("ERROR: 需要 --batch_json 或 (--scene_json + --out)")
        sys.exit(1)


main()