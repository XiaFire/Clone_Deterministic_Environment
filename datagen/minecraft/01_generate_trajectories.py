"""
Trajectory generation for Minecraft-DETERMINISTIC (MC-DET)
----------------------------------------------------------
Generates (x, y, theta_degrees, action) trajectories on the fenced Minecraft map
(assets/maze.npy, 1 = walkable block) with an A*-guided policy.
Output: one HDF5 file, trajectories/trajectory_XXXXXX/data (structured array).
"""
import os
import json
import math
import heapq
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import h5py

# ===================== Configuration =====================
HERE = Path(__file__).resolve().parent



# Trajectory generation parameters
TRAJ_CONFIG = {
    'step_size': 0.3,
    'p_noise': 0.,
    'max_steps': 1000,
    'delta': 15,
    'min_distance': 10,
    'safe_distance': 0.1,
    'center_bias': 0.7,
    'stagnation_threshold': 10,  # 连续10次不动就开始解卡
    'stagnation_distance': 0.01,
    'unstuck_attempts': 5,       # 最多尝试5次解卡
    'exploration_steps': 8       # 每次解卡时的探索步数
}


# ===================== A* Pathfinding =====================
class AStar:
    """A* pathfinding algorithm for grid-based navigation"""
    
    @staticmethod
    def find_path(grid, start, end):
        """Find optimal path from start to end using A* algorithm"""
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        open_heap = []
        heapq.heappush(open_heap, (0, 0, start[0], start[1]))
        came_from = {}
        g_values = {start: 0}
        
        def heuristic(a, b):
            rand = random.random()
            manhattan = abs(a[0] - b[0]) + abs(a[1] - b[1])
            euclidean = (a[0] - b[0])**2 + (a[1] - b[1])**2
            return (1 - rand) * manhattan + rand * euclidean
        
        while open_heap:
            current_f, current_g, x, y = heapq.heappop(open_heap)
            if (x, y) == end:
                # Reconstruct path
                path = []
                while (x, y) in came_from:
                    path.append((x, y))
                    x, y = came_from[(x, y)]
                path.append(start)
                return path[::-1]
            
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                
                # Boundary check
                if nx < 0 or ny < 0 or nx >= grid.shape[0] or ny >= grid.shape[1]:
                    continue
                if grid[nx][ny] == 0:
                    continue
                
                # Diagonal movement check
                if dx != 0 and dy != 0:
                    if not (grid[x + dx, y] == 1 or grid[x, y + dy] == 1):
                        continue
                
                tentative_g = current_g + 1
                if (nx, ny) not in g_values or tentative_g < g_values[(nx, ny)]:
                    came_from[(nx, ny)] = (x, y)
                    g_values[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic((nx, ny), end)
                    heapq.heappush(open_heap, (f, tentative_g, nx, ny))
        
        return []

# ===================== Utility Functions =====================
def setup_environment(maze_path):
    """Initialize the maze environment and return necessary components"""
    maze = np.load(maze_path)
    
    return maze

def is_valid_position(maze, x, y, safe_distance=None):
    """Check if position (x, y) is valid in the maze"""
    if safe_distance is None:
        safe_distance = TRAJ_CONFIG['safe_distance']
    
    ix, iy = int(x), int(y)
    
    # Boundary check
    if (x < safe_distance or x >= maze.shape[0] - safe_distance or 
        y < safe_distance or y >= maze.shape[1] - safe_distance):
        return False
    
    # Grid boundary checks
    if y.is_integer() and iy != 0:
        if maze[ix, iy] == 0 or maze[ix, iy-1] == 0:
            return False
    
    if x.is_integer() and ix != 0:
        if maze[ix, iy] == 0 or maze[ix-1, iy] == 0:
            return False
    
    # Current cell check
    if maze[ix, iy] == 0:
        return False
    
    # Safety distance checks for adjacent obstacles
    directions = [
        (ix + 1, iy, lambda: x >= (ix + 1) - safe_distance),  # Right
        (ix - 1, iy, lambda: x <= ix + safe_distance),        # Left
        (ix, iy + 1, lambda: y >= (iy + 1) - safe_distance),  # Up
        (ix, iy - 1, lambda: y <= iy + safe_distance)         # Down
    ]
    
    for check_x, check_y, condition in directions:
        if (0 <= check_x < maze.shape[0] and 0 <= check_y < maze.shape[1] and 
            maze[check_x, check_y] == 0 and condition()):
            return False
    
    return True

def generate_valid_points(maze, num_points=10000):
    """Generate a set of valid positions in the maze"""
    valid_points = []
    for _ in tqdm(range(num_points), desc="Generating valid points"):
        x, y = random.random() * maze.shape[0], random.random() * maze.shape[1]
        if is_valid_position(maze, x, y):
            valid_points.append((x, y))
    return valid_points

def generate_start_goal_from_samples(valid_points, min_distance=None):
    """Select start and goal positions from valid points"""
    if min_distance is None:
        min_distance = TRAJ_CONFIG['min_distance']
    
    while True:
        start = random.choice(valid_points)
        goal = random.choice(valid_points)
        if math.hypot(start[0] - goal[0], start[1] - goal[1]) > min_distance:
            return start, goal

# ===================== Trajectory Generation =====================
def generate_trajectory(maze, start, goal, valid_points):
    """Generate a trajectory from start to goal using A* guidance with noise"""
    config = TRAJ_CONFIG
    current_pos = start
    current_goal = goal  # 当前目标，可能会变化
    current_theta = random.randint(0, 360)
    trajectory = []
    pre_large_cell = None
    steps = 0
    last_pos = current_pos
    last_theta = current_theta + 100
    
    # 卡住检测
    stagnation_counter = 0
    visited_cells = set()
    goal_changes = 0
    max_goal_changes = 3  # 最多换3次目标
    
    while steps < config['max_steps']:
        # Check if current goal is reached
        if math.hypot(current_pos[0] - current_goal[0], current_pos[1] - current_goal[1]) < config['step_size']:
            print(f"Reached goal at step {steps}")
            break
        
        # Current and goal cells for A*
        large_cell = (int(current_pos[0]), int(current_pos[1]))
        goal_large = (int(current_goal[0]), int(current_goal[1]))
        
        # Update path if needed
        if pre_large_cell is None or large_cell != pre_large_cell:
            path = AStar.find_path(maze, large_cell, goal_large)
        pre_large_cell = large_cell
        
        if not path:
            print("No path found, changing goal...")
            current_goal = _change_goal(current_pos, current_goal, valid_points, config)
            goal_changes += 1
            stagnation_counter = 0
            visited_cells.clear()
            if goal_changes >= max_goal_changes:
                print(f"Changed goal {max_goal_changes} times, terminating trajectory")
                return trajectory
            continue
        
        # 检查是否卡住
        distance_moved = max(abs(current_pos[0] - last_pos[0]), abs(current_pos[1] - last_pos[1]), abs(current_theta - last_theta))
        if distance_moved < config['stagnation_distance']:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        
        # 卡住了就换目标！
        if stagnation_counter >= config['stagnation_threshold']:
            if goal_changes >= max_goal_changes:
                print(f"Changed goal {max_goal_changes} times, terminating trajectory")
                return trajectory
            
            print(f"Stuck for {stagnation_counter} steps, changing goal (attempt {goal_changes + 1})")
            current_goal = _change_goal(current_pos, current_goal, valid_points, config)
            goal_changes += 1
            stagnation_counter = 0
            visited_cells.clear()  # 重新开始探索
            continue
        
        last_pos = current_pos
        last_theta = current_theta
        # 正常的轨迹生成
        target = _determine_target(current_pos, large_cell, current_goal, path, visited_cells, config)
        action = _select_action(current_pos, current_theta, target, config)
        current_pos, current_theta = _execute_action(
            current_pos, current_theta, action, maze, config
        )
        
        trajectory.append((*current_pos, current_theta, action))
        steps += 1
    
    print(f"Trajectory completed: {len(trajectory)} steps, {goal_changes} goal changes")
    return trajectory

def _change_goal(current_pos, old_goal, valid_points, config):
    """换一个新的目标点"""
    min_distance = config['min_distance']
    max_attempts = 50
    
    for _ in range(max_attempts):
        new_goal = random.choice(valid_points)
        
        # 确保新目标距离当前位置足够远
        distance_from_current = math.hypot(
            new_goal[0] - current_pos[0], 
            new_goal[1] - current_pos[1]
        )
        
        # 确保新目标与旧目标不同
        distance_from_old = math.hypot(
            new_goal[0] - old_goal[0], 
            new_goal[1] - old_goal[1]
        )
        
        if distance_from_current > min_distance and distance_from_old > min_distance * 0.5:
            print(f"  New goal: ({new_goal[0]:.2f}, {new_goal[1]:.2f})")
            return new_goal
    
    # 如果找不到合适的新目标，至少选一个远一点的
    best_goal = max(valid_points, 
                   key=lambda p: math.hypot(p[0] - current_pos[0], p[1] - current_pos[1]))
    print(f"  Fallback goal: ({best_goal[0]:.2f}, {best_goal[1]:.2f})")
    return best_goal

def _determine_target(current_pos, large_cell, goal, path, visited_cells, config):
    """Determine the next target position based on current strategy"""
    prefer_center = True
    center_bias = config['center_bias']
    
    if (prefer_center and random.random() < center_bias) and not large_cell in visited_cells:
        cell_center = (large_cell[0] + 0.5, large_cell[1] + 0.5)
        
        # If close to center, move to next cell
        if math.hypot(current_pos[0] - cell_center[0], current_pos[1] - cell_center[1]) < 0.2:
            visited_cells.add(large_cell)
            if len(path) > 1:
                next_center = (path[1][0] + 0.5, path[1][1] + 0.5)
                return next_center
            else:
                return goal
        else:
            return cell_center
    else:
        # Find next unvisited cell
        if len(path) > 1 and path[1] in visited_cells:
            for i in range(2, len(path)):
                if path[i] not in visited_cells:
                    return (path[i][0] + 0.5, path[i][1] + 0.5)
            return goal
        else:
            return goal if len(path) <= 1 else (path[1][0] + 0.5, path[1][1] + 0.5)

def _select_action(current_pos, current_theta, target, config):
    """Select action based on noise probability and target direction"""
    if random.random() < config['p_noise']:
        return random.choices([1, 2, 3], weights=[0.8, 0.1, 0.1])[0]
    
    # Calculate desired direction
    dx = target[0] - current_pos[0]
    dy = target[1] - current_pos[1]
    desired_theta = math.atan2(dy, dx) * 180 / math.pi % 360
    
    # Find best action
    delta = config['delta']
    theta_options = [current_theta, (current_theta - delta) % 360, (current_theta + delta) % 360]
    differences = [
        min((t - desired_theta) % 360, 360 - (t - desired_theta) % 360) 
        for t in theta_options
    ]
    
    return [1, 2, 3][differences.index(min(differences))]

def _execute_action(current_pos, current_theta, action, maze, config):
    """Execute the selected action and return new position and orientation"""
    if action == 2:  # Turn left
        current_theta = (current_theta - config['delta']) % 360
    elif action == 3:  # Turn right
        current_theta = (current_theta + config['delta']) % 360
    elif action == 1:  # Move forward
        rad = math.radians(current_theta)
        new_x = current_pos[0] + config['step_size'] * math.cos(rad)
        new_y = current_pos[1] + config['step_size'] * math.sin(rad)
        if is_valid_position(maze, new_x, new_y):
            current_pos = (new_x, new_y)
    
    return current_pos, current_theta

# ===================== Visualization =====================
def plot_trajectory(maze, trajectory, start, goal, idx, save_path):
    """Plot trajectory on maze map and save as image"""
    plt.figure(figsize=(10, 10))
    print(maze.shape)
    plt.imshow(maze, cmap='gray_r', origin='lower', extent=[0, maze.shape[1], 0, maze.shape[0]])
    
    x = [p[0] for p in trajectory]
    y = [p[1] for p in trajectory]
    
    plt.plot(x, y, 'r-', linewidth=2, label='Trajectory')
    plt.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start')
    plt.scatter(x[-1], y[-1], c='blue', s=100, marker='x', label='End')
    plt.scatter(start[0], start[1], c='red', s=100, marker='*', label='Target Start')
    plt.scatter(goal[0], goal[1], c='blue', s=100, marker='*', label='Target Goal')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Navigation Trajectory')
    plt.legend()
    plt.savefig(save_path / f'mc_trajectory_{idx}.png')
    plt.close()

# ===================== Main Process =====================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate navigation trajectories for Memory Maze')
    parser.add_argument('--num_trajectories', type=int, default=10000, 
                       help='Number of trajectories to generate')
    parser.add_argument('--num_valid_points', type=int, default=10000,
                       help='Number of valid points to pre-generate')
    parser.add_argument('--plot_interval', type=int, default=100,
                       help='Interval for saving trajectory plots (0 disables plots)')
    parser.add_argument('--maze', type=str, default=str(HERE / 'assets' / 'maze.npy'),
                       help='Occupancy grid of the fenced Minecraft map')
    parser.add_argument('--output', type=str, default='datasets/mc/info.hdf5',
                       help='Output HDF5 path')
    parser.add_argument('--plot_dir', type=str, default='datasets/mc/plots',
                       help='Directory for trajectory plots')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (the released data/mc.h5 was generated without a fixed seed)')
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    SAVE_PATH = Path(args.plot_dir)
    if args.plot_interval > 0:
        os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    
    # Setup environment and paths
    maze = setup_environment(args.maze)
    original_maze = maze
    
    # Generate valid points
    print("Generating valid positions...")
    valid_points = generate_valid_points(original_maze, args.num_valid_points)
    print(f"Generated {len(valid_points)} valid points")
    
    # Generate trajectories
    print(f"Generating {args.num_trajectories} trajectories...")
    all_trajectories = []
    
    for i in tqdm(range(args.num_trajectories), desc="Generating trajectories"):
        start, goal = generate_start_goal_from_samples(valid_points)
        # 传递valid_points，以便换目标时使用
        trajectory = generate_trajectory(original_maze, start, goal, valid_points)
        
        if not trajectory:
            print(f"Trajectory {i} is empty, skipping.")
            continue
        
        all_trajectories.append(trajectory)
        
        # Plot trajectory periodically
        if args.plot_interval > 0 and i % args.plot_interval == 0:
            print(f"Trajectory {i} saved and plotted.")
            traj_positions = [[x, y] for x, y, _, _ in trajectory]
            plot_trajectory(original_maze.T, traj_positions, start, goal, i, SAVE_PATH)
    
    # 保存为 HDF5 格式
    hdf5_file = Path(args.output)
    save_trajectories_hdf5(all_trajectories, hdf5_file, TRAJ_CONFIG)
    print(f"All trajectories saved to {hdf5_file}")

def save_trajectories_hdf5(trajectories, output_file, config=None):
    """保存轨迹为 HDF5 格式"""
    with h5py.File(output_file, 'w') as f:
        # 保存配置信息作为属性
        if config:
            for key, value in config.items():
                f.attrs[key] = value
        
        # 创建轨迹组
        trajectories_group = f.create_group('trajectories')
        
        # 保存每条轨迹
        for i, traj in enumerate(trajectories):
            traj_group = trajectories_group.create_group(f'trajectory_{i:06d}')
            
            # 转换为numpy数组
            traj_array = np.array(traj, dtype=[
                ('x', 'f4'), ('y', 'f4'), 
                ('theta', 'f4'), ('action', 'i4')
            ])
            
            # 简单压缩存储，不使用shuffle
            traj_group.create_dataset('data', 
                                    data=traj_array, 
                                    compression='gzip',
                                    compression_opts=6)  # 适中的压缩级别
            
            # 添加轨迹元数据
            traj_group.attrs['length'] = len(traj)
            if len(traj) > 0:
                traj_group.attrs['start_pos'] = [traj[0][0], traj[0][1]]
                traj_group.attrs['end_pos'] = [traj[-1][0], traj[-1][1]]

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

if __name__ == '__main__':
    main()