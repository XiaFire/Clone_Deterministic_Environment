import bpy
import numpy as np
import math
import os
import sys
import h5py
import argparse
from tqdm import tqdm
import io
from contextlib import redirect_stdout, redirect_stderr


# ================= 配置区域 =================
# 输入文件路径
_HERE = os.path.dirname(os.path.abspath(__file__))
OBJ_PATH = os.path.join(_HERE, "assets", "fence.obj")   # override with --obj
HDF5_PATH = os.path.join(_HERE, "data", "mc.h5")        # override with --h5
OUTPUT_DIR = os.path.join("datasets", "mc", "obs")       # override with --out

# 渲染设置
RESOLUTION_X = 256
RESOLUTION_Y = 256
EYE_HEIGHT = 1.62  # Minecraft 史蒂夫的眼睛高度
FLOOR_Y = 4.0      # 你之前的代码中 tp 高度是 4.0
FOV_DEGREES = 70.0 # Minecraft 默认 FOV

# 坐标系修正 (非常关键!)
# 大多数导出器(Mineways/jMc2Obj)会将 MC的Y轴(高度) 映射为 Blender的Z轴
# MC的 Z轴 映射为 Blender的 -Y 或 Y
# 你可能需要根据实际渲染结果调整下面这个参数
INVERT_MC_Z = True # 如果渲染出来前后反了，改这个
OFFSET_X = 0.0     # 如果地图原点有偏移，在这里修正
OFFSET_Y = 0.0     # Blender的Y对应MC的Z
# ===========================================

def reset_scene():
    """清空场景中所有物体"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def fix_transparency_settings():
    """
    自动修复树叶、草、水等材质的透明度设置
    """
    print("Fixing transparency settings for Eevee...")
    
    for mat in bpy.data.materials:
        # 1. 开启节点使用 (防止报错)
        if not mat.use_nodes:
            mat.use_nodes = True
            
        # 2. 识别需要透明的材质
        # Minecraft 导出名通常包含这些关键词
        mat_name = mat.name.lower()
        needs_transparency = False
        
        keywords = ['leaves', 'grass', 'fern', 'flower', 'crop', 'web', 'sapling', 'torch', 'water', 'glass']
        
        if any(k in mat_name for k in keywords):
            needs_transparency = True
            
        if needs_transparency:
            # === 核心修复 settings ===
            
            # 模式 A: Alpha Clip (推荐用于树叶/草)
            # 只有完全不透明的地方才显示，边缘锐利，不模糊，渲染最快，且不会有排序错误
            mat.blend_method = 'CLIP' 
            mat.shadow_method = 'CLIP' # 让影子也是镂空的
            mat.alpha_threshold = 0.5  # 阈值，通常0.5刚好
            
            # 特殊处理：水和玻璃可能需要 Hashed 或 Blend，但在 MC 里 Clip 通常也够用且最稳
            if 'water' in mat_name or 'glass' in mat_name:
                # 如果你想让水半透明，可以用 HASHED
                # mat.blend_method = 'HASHED' 
                pass

            # 3. 确保 Alpha 通道已连接 (双重保险)
            # 有时候 obj 导入进来，Alpha 通道没连上，这里强制连一下
            try:
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                bsdf = nodes.get("Principled BSDF")
                
                # 寻找图片节点
                tex_node = None
                for n in nodes:
                    if n.type == 'TEX_IMAGE':
                        tex_node = n
                        break
                
                if bsdf and tex_node:
                    # 连接 Alpha 输出 -> BSDF Alpha 输入
                    links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
            except:
                pass


def setup_render_engine():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    
    # 开启屏幕空间 Ambient Occlusion
    scene.eevee.use_gtao = True       # 打开 GTAO
    scene.eevee.gtao_distance = 10.0   # AO 影响距离，可调
    scene.eevee.gtao_factor = 1.0     # AO 强度，可调
    scene.eevee.gtao_quality = 1.0    # 质量，0~1，高质量更慢
    scene.eevee.use_gtao_bent_normals = True  # 更自然阴影方向

    # -------------------------
    # 环境光
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    
    bg = world.node_tree.nodes.new(type="ShaderNodeBackground")
    bg.inputs[0].default_value = (0.3, 0.5, 1.0, 1.0)  # 天空颜色
    bg.inputs[1].default_value = 0.8  # 强度，可调
    world.node_tree.links.new(
        bg.outputs[0],
        world.node_tree.nodes["World Output"].inputs[0]
    )

    # 可选：开启柔和阴影
    scene.eevee.use_soft_shadows = True
    scene.eevee.shadow_cube_size = '1024'
    scene.eevee.shadow_cascade_size = '1024'
    # Sun 光
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(50), 0, math.radians(30))
    light_data.energy = 1
    light_data.angle = 0.8 # 角度为0 = 锐利阴影

    # accelerate rendering
    scene.render.use_persistent_data = True
    scene.eevee.taa_render_samples = 8  # 渲染输出时的采样 (Output)
    scene.eevee.taa_samples = 8         # 视口采样 (Viewport)

def make_it_look_like_mc():
    print("Applying Minecraft Visual Style...")
    scene = bpy.context.scene
    
    # 1. --- 修正色彩管理 (最重要！) ---
    # 从 Filmic/AgX 改回 Standard，找回高饱和度
    if scene.view_settings.view_transform != 'Standard':
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'High Contrast' # 可选：加一点对比度更像

    # 3. --- 修正纹理模糊 (变回像素风) ---
    # 遍历所有材质的所有纹理节点，强制设为“最近邻” (Closest)
    for mat in bpy.data.materials:
        if not mat.use_nodes: continue
        
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                # Linear = 模糊 (默认), Closest = 硬边像素 (MC风格)
                node.interpolation = 'Closest' 

def load_obj(filepath):
    """导入 OBJ 文件 (适配 Blender 4.0+ API)"""
    # Blender 4.0+ 使用 wm.obj_import (基于 C++，速度更快)
    # 注意参数名变化：
    # axis_forward -> forward_axis
    # axis_up      -> up_axis
    # 值也变成了全大写单词，例如 '-Z' 变成了 'NEGATIVE_Z'
    
    bpy.ops.wm.obj_import(
        filepath=filepath, 
        forward_axis='NEGATIVE_Z',  # 对应 Minecraft 的前方
        up_axis='Y'                 # 对应 Minecraft 的上方
    )
    
    # 额外提示：新的导入器可能会默认把物体选中
    # 为了保险，可以强制将导入的物体设为 active 方便后续操作（虽然此脚本只需要它待着不动）
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

def mc_yaw_to_blender_rad(mc_yaw):
    """
    Minecraft Yaw 转 Blender Euler Rotation Z
    MC: 0=南(+Z), 90=西(-X), 180=北(-Z), 270=东(+X)
    Blender Camera: 默认看向 -Z
    """
    # 这是一个经验公式，可能需要根据你的 traj 数据微调 (+/- 90度)
    # 通常 MC Yaw 转换需要取负值并加偏移
    # 假设 traj 里的 theta 是角度制
    return math.radians(mc_yaw - 180)

def parse_args():
    # Only parse arguments after '--' if present, to avoid conflict with blender args
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        # If no '--', assuming no custom args passed or they are mixed. 
        # But usually blender script args are after '--'. 
        # If running purely python without blender CLI wrapper, sys.argv is fine.
        # Check if 'blender' is in the executable path to be safe.
        if "blender" in sys.executable.lower():
             argv = [] # Default to empty if inside blender but no '--' separator
        pass

    parser = argparse.ArgumentParser(description='Blender Render Script')
    parser.add_argument('--idx', type=int, default=0, help='Index of the split (0-based)')
    parser.add_argument('--split', type=int, default=1, help='Total number of splits')
    parser.add_argument('--obj', type=str, default=None, help='Map OBJ (default: assets/fence.obj)')
    parser.add_argument('--h5', type=str, default=None, help='Trajectory HDF5 (default: data/mc.h5)')
    parser.add_argument('--out', type=str, default=None, help='Output obs directory (default: datasets/mc/obs)')
    parser.add_argument('--limit', type=int, default=None, help='Render only the first N trajectories of this split')
    
    # Allow unknown args because blender might pass some implicitly if we are not careful
    return parser.parse_known_args(argv)[0]

def load_trajectories_hdf5(file_path, trajectory_indices=None):
    """从 HDF5 文件读取轨迹"""
    trajectories = []
    
    with h5py.File(file_path, 'r') as f:
        # 读取配置信息
        config = dict(f.attrs)
        
        trajectories_group = f['trajectories']
        
        # 获取轨迹键名并排序以保持顺序
        if trajectory_indices is None:
            traj_keys = sorted(trajectories_group.keys())
        else:
            traj_keys = [f'trajectory_{i:06d}' for i in trajectory_indices]
        
        for key in traj_keys:
            if key in trajectories_group:
                traj_data = trajectories_group[key]['data'][:]
                # 转换回原始格式
                trajectory = [(point['x'], point['y'], point['theta'], point['action']) 
                             for point in traj_data]
                trajectories.append(trajectory)
    
    return trajectories, config

def render_trajectory_as_animation(traj, output_dir, cam_obj):
    scene = bpy.context.scene
    
    # 1. 设置动画长度
    num_frames = len(traj)
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    scene.frame_step = 1
    
    # 2. 清理旧动画数据 (防止上一条轨迹干扰)
    if cam_obj.animation_data:
        cam_obj.animation_data_clear()
        
    # 3. 批量插入关键帧 (这比逐帧渲染快得多)
    # 提示：为了极速，可以不每帧 update view_layer，只在最后 update
    print(f"Keyframing {num_frames} frames...")
    
    for frame_idx, row in enumerate(traj):
        mc_i, mc_j, mc_yaw = row[0], row[1], row[2]
        
        # --- 坐标计算 ---
        mc_x = (19.5 - mc_j) * -1
        mc_z = (mc_i - 29.5) * -1
        b_x = mc_x + OFFSET_X
        b_y = mc_z + OFFSET_Y
        b_z = FLOOR_Y + EYE_HEIGHT
        
        # 设置当前帧
        scene.frame_set(frame_idx)
        
        # 设置位置
        cam_obj.location = (b_x, b_y, b_z)
        cam_obj.keyframe_insert(data_path="location", frame=frame_idx)
        
        # 设置旋转
        cam_obj.rotation_mode = 'XYZ'
        cam_obj.rotation_euler[0] = math.radians(90)
        cam_obj.rotation_euler[1] = 0
        cam_obj.rotation_euler[2] = mc_yaw_to_blender_rad(mc_yaw)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame_idx)

    # 4. 设置输出路径
    # Blender 动画渲染会自动在路径后加帧号 (例如 0000.png, 0001.png)
    # 如果你一定要 0.png, 1.png，建议渲染完后用 os.rename 批量重命名，或者接受 0000 格式
    scene.render.filepath = os.path.join(output_dir, "") # 结尾要有斜杠

    # 5. 一次性渲染整个轨迹
    # print("Rendering animation...")
    # bpy.ops.render.render(animation=True)
    # 将 stdout 和 stderr 重定向到黑洞，不再打印到终端
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        bpy.ops.render.render(animation=True)

def main():
    global OBJ_PATH, HDF5_PATH, OUTPUT_DIR
    _args = parse_args()
    OBJ_PATH = _args.obj or OBJ_PATH
    HDF5_PATH = _args.h5 or HDF5_PATH
    OUTPUT_DIR = _args.out or OUTPUT_DIR
    # 1. 准备环境
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    reset_scene()
    setup_render_engine()
    
    # 2. 导入地图
    print(f"Loading OBJ: {OBJ_PATH}...")
    load_obj(OBJ_PATH)
    fix_transparency_settings()
    make_it_look_like_mc()
    
    # 3. 创建相机
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new(name='Camera', object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # 设置 FOV
    cam_data.angle = math.radians(FOV_DEGREES)

    # 4. 读取轨迹
    if not os.path.exists(HDF5_PATH):
        print(f"Error: HDF5 file not found at {HDF5_PATH}")
        return

    # 解析参数
    args = parse_args()
    print(f"Args: idx={args.idx}, split={args.split}")

    print(f"Loading trajectories from {HDF5_PATH}...")
    try:
        trajectories, config = load_trajectories_hdf5(HDF5_PATH)
        print(f"Loaded {len(trajectories)} trajectories.")
    except Exception as e:
        print(f"Error loading HDF5: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 逐个轨迹处理
    total_trajs_all = len(trajectories)
    
    # Calculate split range
    start = args.idx * total_trajs_all // args.split
    end = (args.idx + 1) * total_trajs_all // args.split
    
    subset_trajectories = trajectories[start:end]
    subset_indices = range(start, end)
    if args.limit is not None:
        subset_trajectories = subset_trajectories[:args.limit]
        subset_indices = subset_indices[:args.limit]
    
    print(f"Processing split {args.idx}/{args.split}: indices [{start}, {end}), count={len(subset_trajectories)}")
    
    for idx, (original_idx, traj) in tqdm(enumerate(zip(subset_indices, subset_trajectories))):
        traj_idx = original_idx 
        # 为每个轨迹创建一个子目录
        traj_output_dir = os.path.join(OUTPUT_DIR, f"{traj_idx}")
        
        if os.path.exists(traj_output_dir):
            # 简单的检查：如果目录存在且不为空，可能已经处理过
            if os.listdir(traj_output_dir):
                print(f"Skipping {traj_idx}, directory exists and not empty.")
                continue
        else:
            os.makedirs(traj_output_dir)
            
        print(f"Processing trajectory {traj_idx} (Local: {idx+1}/{len(subset_trajectories)}) (Length: {len(traj)})")
        
        render_trajectory_as_animation(traj, traj_output_dir, cam_obj)
        # traj_np = np.array(traj, dtype=np.float32)
        # for i in tqdm(range(len(traj_np))):
        #     row = traj_np[i]
        #     mc_i = row[0] 
        #     mc_j = row[1] 
        #     mc_yaw = row[2]
            
        #     # --- 坐标变换 ---
        #     mc_x = 19.5 - mc_j
        #     mc_z = mc_i - 29.5
            
        #     b_x = mc_x + OFFSET_X
        #     b_y = (-mc_z if INVERT_MC_Z else mc_z) + OFFSET_Y
        #     b_z = FLOOR_Y + EYE_HEIGHT
            
        #     cam_obj.location = (b_x, b_y, b_z)
            
        #     # --- 旋转变换 ---
        #     cam_obj.rotation_mode = 'XYZ'
        #     cam_obj.rotation_euler[0] = math.radians(90) 
        #     cam_obj.rotation_euler[1] = 0
        #     cam_obj.rotation_euler[2] = mc_yaw_to_blender_rad(mc_yaw)
            
        #     # --- 更新场景并渲染 ---
        #     bpy.context.view_layer.update()
            
        #     # 设置输出路径: OUTPUT_DIR/{traj_idx}/{i}.png
        #     output_file = os.path.join(traj_output_dir, f"{i}.png")
        #     bpy.context.scene.render.filepath = output_file
            
        #     # 执行渲染 (保存到文件)
        #     bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()



# Original launch (20 parallel Blender processes):
# for i in $(seq 0 19); do
#     blender-4.0.0-linux-x64/blender -b -P 02_render_blender.py -- --idx=$i --split=20 &
# done
