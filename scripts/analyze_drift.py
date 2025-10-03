#!/usr/bin/env python3
"""
V4b 转向偏差诊断分析器
检查系统性左偏、转向行为和侧向漂移
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from stable_baselines3 import PPO
from quadruped_env import QuadrupedEnv
import argparse
import matplotlib

# 设置matplotlib字体
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  

def analyze_drift(model_path, episodes=5):
    """分析转向偏差和系统性漂移"""
    
    # 创建环境
    env = QuadrupedEnv(render_mode=False)
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 数据收集
    all_trajectories = []
    all_yaw_data = []
    all_lateral_data = []
    all_angular_data = []
    
    print(f"🔍 开始转向偏差分析，共{episodes}个episode...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_steps = 0
        
        # 单个episode的数据
        episode_trajectory = {'x': [], 'y': [], 'z': []}
        episode_yaw = []
        episode_lateral_vel = []
        episode_angular_vel = []
        episode_time = []
        
        print(f"  📊 分析Episode {episode+1}/{episodes}...")
        
        while episode_steps < 1000:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            # 获取位置和朝向信息
            pos, orn = p.getBasePositionAndOrientation(env.robot_id)
            vel, ang_vel = p.getBaseVelocity(env.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            
            # 记录数据
            episode_trajectory['x'].append(pos[0])
            episode_trajectory['y'].append(pos[1])
            episode_trajectory['z'].append(pos[2])
            
            episode_yaw.append(euler[2])  # yaw角度
            episode_lateral_vel.append(vel[1])  # Y轴速度(侧向)
            episode_angular_vel.append(ang_vel[2])  # Z轴角速度(转向)
            episode_time.append(episode_steps)
            
            episode_steps += 1
            
            if done or truncated:
                break
        
        # 保存episode数据
        all_trajectories.append(episode_trajectory)
        all_yaw_data.append(episode_yaw)
        all_lateral_data.append(episode_lateral_vel)
        all_angular_data.append(episode_angular_vel)
    
    env.close()
    
    # 转向偏差分析
    print("\n" + "="*60)
    print("🎯 V4b 转向偏差诊断报告")
    print("="*60)
    
    # 1. 轨迹分析
    print("\n📍 轨迹偏移分析:")
    final_positions = []
    path_curvatures = []
    
    for i, traj in enumerate(all_trajectories):
        start_x, start_y = traj['x'][0], traj['y'][0]
        end_x, end_y = traj['x'][-1], traj['y'][-1]
        
        # 最终位置偏移
        lateral_drift = end_y - start_y
        forward_distance = end_x - start_x
        final_positions.append((forward_distance, lateral_drift))
        
        # 路径曲率(Y坐标的变化程度)
        y_positions = np.array(traj['y'])
        curvature = np.std(y_positions)
        path_curvatures.append(curvature)
        
        print(f"  Episode {i+1}: 前进={forward_distance:.3f}m, 侧偏={lateral_drift:.3f}m, 曲率={curvature:.3f}")
    
    # 统计分析
    avg_lateral_drift = np.mean([pos[1] for pos in final_positions])
    avg_forward = np.mean([pos[0] for pos in final_positions])
    avg_curvature = np.mean(path_curvatures)
    
    print(f"\n📊 轨迹统计:")
    print(f"  平均前进距离: {avg_forward:.3f} m")
    print(f"  平均侧向偏移: {avg_lateral_drift:.3f} m")
    print(f"  平均路径曲率: {avg_curvature:.3f}")
    
    # 2. 朝向变化分析
    print(f"\n🧭 朝向变化分析:")
    all_yaw_changes = []
    
    for i, yaw_data in enumerate(all_yaw_data):
        yaw_array = np.array(yaw_data)
        
        # 计算朝向变化趋势(线性拟合斜率)
        time_points = np.arange(len(yaw_array))
        yaw_slope = np.polyfit(time_points, yaw_array, 1)[0]
        
        # 朝向变化总量
        total_yaw_change = yaw_array[-1] - yaw_array[0]
        yaw_variance = np.var(yaw_array)
        
        all_yaw_changes.append(yaw_slope)
        
        print(f"  Episode {i+1}: 朝向趋势={yaw_slope:.6f} rad/step, 总变化={total_yaw_change:.3f} rad")
    
    avg_yaw_slope = np.mean(all_yaw_changes)
    print(f"\n  平均朝向偏移趋势: {avg_yaw_slope:.6f} rad/step")
    
    # 3. 侧向速度分析
    print(f"\n↔️  侧向速度分析:")
    all_lateral_means = []
    
    for i, lateral_data in enumerate(all_lateral_data):
        lateral_array = np.array(lateral_data)
        lateral_mean = np.mean(lateral_array)
        lateral_std = np.std(lateral_array)
        
        all_lateral_means.append(lateral_mean)
        print(f"  Episode {i+1}: 平均侧向速度={lateral_mean:.4f} m/s, 标准差={lateral_std:.4f}")
    
    avg_lateral_velocity = np.mean(all_lateral_means)
    print(f"\n  总体平均侧向速度: {avg_lateral_velocity:.4f} m/s")
    
    # 4. 转向问题诊断
    print(f"\n🚨 转向问题诊断:")
    
    # 判断是否存在系统性左偏
    if avg_lateral_drift < -0.1:
        print("❌ 检测到系统性左偏！机器人倾向于向左偏移")
        drift_severity = "严重" if avg_lateral_drift < -0.3 else "中等"
        print(f"   偏移严重程度: {drift_severity} (偏移量: {avg_lateral_drift:.3f}m)")
    elif avg_lateral_drift > 0.1:
        print("❌ 检测到系统性右偏！机器人倾向于向右偏移")
        drift_severity = "严重" if avg_lateral_drift > 0.3 else "中等"
        print(f"   偏移严重程度: {drift_severity} (偏移量: {avg_lateral_drift:.3f}m)")
    else:
        print("✅ 未检测到明显的系统性侧向偏移")
    
    # 判断朝向稳定性
    if abs(avg_yaw_slope) > 0.001:
        direction = "逆时针" if avg_yaw_slope > 0 else "顺时针"
        print(f"❌ 检测到系统性{direction}转向趋势！")
        print(f"   转向速率: {abs(avg_yaw_slope):.6f} rad/step")
    else:
        print("✅ 朝向控制相对稳定")
    
    # 判断侧向速度
    if abs(avg_lateral_velocity) > 0.05:
        direction = "左" if avg_lateral_velocity < 0 else "右"
        print(f"❌ 检测到持续{direction}向运动！")
        print(f"   侧向速度: {avg_lateral_velocity:.4f} m/s")
    else:
        print("✅ 侧向运动控制良好")
    
    # 5. 生成可视化
    create_drift_plots(all_trajectories, all_yaw_data, all_lateral_data, all_angular_data)
    
    # 6. 修复建议
    print(f"\n💡 V4c修复建议:")
    if avg_lateral_drift < -0.1 or avg_lateral_velocity < -0.05:
        print("🔧 建议1: 添加左右腿对称性奖励")
        print("🔧 建议2: 增强侧向运动惩罚权重")
        print("🔧 建议3: 检查左前腿支撑力奖励")
    
    if abs(avg_yaw_slope) > 0.001:
        print("🔧 建议4: 强化朝向一致性奖励")
        print("🔧 建议5: 添加角速度稳定性约束")

def create_drift_plots(trajectories, yaw_data, lateral_data, angular_data):
    """生成转向偏差可视化图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('V4b Drift Analysis', fontsize=16, fontweight='bold')
    
    # 1. 轨迹图
    for i, traj in enumerate(trajectories):
        axes[0, 0].plot(traj['x'], traj['y'], label=f'Episode {i+1}', alpha=0.7)
        axes[0, 0].scatter(traj['x'][0], traj['y'][0], color='green', s=50, marker='o')  # 起点
        axes[0, 0].scatter(traj['x'][-1], traj['y'][-1], color='red', s=50, marker='x')  # 终点
    
    axes[0, 0].set_xlabel('X Position (m)')
    axes[0, 0].set_ylabel('Y Position (m)')
    axes[0, 0].set_title('Trajectory Paths')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. 朝向变化
    for i, yaw in enumerate(yaw_data):
        time_steps = np.arange(len(yaw))
        axes[0, 1].plot(time_steps, yaw, label=f'Episode {i+1}', alpha=0.7)
    
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Yaw Angle (rad)')
    axes[0, 1].set_title('Yaw Angle Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. 侧向速度
    for i, lateral in enumerate(lateral_data):
        time_steps = np.arange(len(lateral))
        axes[1, 0].plot(time_steps, lateral, label=f'Episode {i+1}', alpha=0.7)
    
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Lateral Velocity (m/s)')
    axes[1, 0].set_title('Lateral Velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. 角速度
    for i, angular in enumerate(angular_data):
        time_steps = np.arange(len(angular))
        axes[1, 1].plot(time_steps, angular, label=f'Episode {i+1}', alpha=0.7)
    
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 1].set_title('Angular Velocity (Z-axis)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('v4b_drift_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 转向偏差分析图表已保存: v4b_drift_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V4b 转向偏差和系统性漂移分析')
    parser.add_argument('--model', type=str, required=True, help='V4b模型路径')
    parser.add_argument('--episodes', type=int, default=3, help='分析的episode数量 (默认3)')
    
    args = parser.parse_args()
    
    print("🚀 启动V4b转向偏差诊断分析器...")
    analyze_drift(args.model, args.episodes)
    print("✅ 分析完成！")