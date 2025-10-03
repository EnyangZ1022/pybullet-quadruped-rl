#!/usr/bin/env python3
"""
V4b 对称性诊断分析器
检查左右腿关节角度和速度的对称性
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

def analyze_symmetry(model_path, episodes=5):
    """分析左右腿对称性"""
    
    # 创建环境
    env = QuadrupedEnv(render_mode=False)
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 数据收集结构
    left_leg_data = {
        'hip': {'angles': [], 'velocities': []},
        'thigh': {'angles': [], 'velocities': []},
        'calf': {'angles': [], 'velocities': []}
    }
    
    right_leg_data = {
        'hip': {'angles': [], 'velocities': []},
        'thigh': {'angles': [], 'velocities': []},
        'calf': {'angles': [], 'velocities': []}
    }
    
    print(f"🔍 开始对称性分析，共{episodes}个episode...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_steps = 0
        
        print(f"  📊 分析Episode {episode+1}/{episodes}...")
        
        while episode_steps < 1000:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            # 获取关节状态
            joint_positions = []
            joint_velocities = []
            
            for joint_id in env.joint_indices:
                joint_state = p.getJointState(env.robot_id, joint_id)
                joint_positions.append(joint_state[0])  # 角度
                joint_velocities.append(joint_state[1])  # 速度
            
            # 关节索引映射：
            # [0,1,2]: FL_hip, FL_thigh, FL_calf
            # [3,4,5]: FR_hip, FR_thigh, FR_calf  
            # [6,7,8]: RL_hip, RL_thigh, RL_calf
            # [9,10,11]: RR_hip, RR_thigh, RR_calf
            
            # 左腿数据 (FL + RL)
            left_leg_data['hip']['angles'].extend([joint_positions[0], joint_positions[6]])
            left_leg_data['thigh']['angles'].extend([joint_positions[1], joint_positions[7]])
            left_leg_data['calf']['angles'].extend([joint_positions[2], joint_positions[8]])
            
            left_leg_data['hip']['velocities'].extend([joint_velocities[0], joint_velocities[6]])
            left_leg_data['thigh']['velocities'].extend([joint_velocities[1], joint_velocities[7]])
            left_leg_data['calf']['velocities'].extend([joint_velocities[2], joint_velocities[8]])
            
            # 右腿数据 (FR + RR)
            right_leg_data['hip']['angles'].extend([joint_positions[3], joint_positions[9]])
            right_leg_data['thigh']['angles'].extend([joint_positions[4], joint_positions[10]])
            right_leg_data['calf']['angles'].extend([joint_positions[5], joint_positions[11]])
            
            right_leg_data['hip']['velocities'].extend([joint_velocities[3], joint_velocities[9]])
            right_leg_data['thigh']['velocities'].extend([joint_velocities[4], joint_velocities[10]])
            right_leg_data['calf']['velocities'].extend([joint_velocities[5], joint_velocities[11]])
            
            episode_steps += 1
            
            if done or truncated:
                break
    
    env.close()
    
    # 对称性分析
    print("\n" + "="*60)
    print("🎯 V4 左右腿对称性诊断报告")
    print("="*60)
    
    joint_types = ['hip', 'thigh', 'calf']
    overall_symmetry_score = 0
    
    for joint_type in joint_types:
        left_angles = np.array(left_leg_data[joint_type]['angles'])
        right_angles = np.array(right_leg_data[joint_type]['angles'])
        left_vels = np.array(left_leg_data[joint_type]['velocities'])
        right_vels = np.array(right_leg_data[joint_type]['velocities'])
        
        # 角度对称性指标
        angle_diff_mean = np.mean(np.abs(left_angles - right_angles))
        angle_diff_std = np.std(np.abs(left_angles - right_angles))
        angle_corr = np.corrcoef(left_angles, right_angles)[0, 1]
        
        # 速度对称性指标
        vel_diff_mean = np.mean(np.abs(left_vels - right_vels))
        vel_diff_std = np.std(np.abs(left_vels - right_vels))
        vel_corr = np.corrcoef(left_vels, right_vels)[0, 1]
        
        # 统计描述
        left_angle_range = [np.min(left_angles), np.max(left_angles)]
        right_angle_range = [np.min(right_angles), np.max(right_angles)]
        
        print(f"\n📊 {joint_type.upper()} 关节对称性分析:")
        print(f"  角度范围: 左腿[{left_angle_range[0]:.3f}, {left_angle_range[1]:.3f}] | "
              f"右腿[{right_angle_range[0]:.3f}, {right_angle_range[1]:.3f}]")
        print(f"  角度差异: 均值={angle_diff_mean:.4f} rad, 标准差={angle_diff_std:.4f} rad")
        print(f"  角度相关性: {angle_corr:.4f}")
        print(f"  速度差异: 均值={vel_diff_mean:.4f} rad/s, 标准差={vel_diff_std:.4f} rad/s")
        print(f"  速度相关性: {vel_corr:.4f}")
        
        # 对称性评级
        if angle_diff_mean < 0.08 and angle_corr > 0.85:
            symmetry_grade = "优秀"
            grade_icon = "✅"
            joint_score = 100
        elif angle_diff_mean < 0.15 and angle_corr > 0.70:
            symmetry_grade = "良好"
            grade_icon = "🟢"
            joint_score = 80
        elif angle_diff_mean < 0.25 and angle_corr > 0.50:
            symmetry_grade = "一般"
            grade_icon = "🟡"
            joint_score = 60
        else:
            symmetry_grade = "差"
            grade_icon = "❌"
            joint_score = 40
        
        print(f"  {grade_icon} 对称性评级: {symmetry_grade} ({joint_score}分)")
        overall_symmetry_score += joint_score
        
        # 问题诊断
        if angle_diff_mean > 0.2:
            print(f"  ⚠️  警告: {joint_type}关节左右差异过大，可能影响步态平衡")
        if angle_corr < 0.5:
            print(f"  ⚠️  警告: {joint_type}关节左右运动模式不协调")
    
    # 总体评估
    overall_symmetry_score /= 3
    print(f"\n🏆 总体对称性得分: {overall_symmetry_score:.1f}/100")
    
    if overall_symmetry_score >= 85:
        print("✅ 优秀！左右腿高度对称，无需调整")
    elif overall_symmetry_score >= 70:
        print("🟢 良好！轻微不对称，建议微调")
    elif overall_symmetry_score >= 50:
        print("🟡 一般！存在明显不对称，需要优化")
    else:
        print("❌ 较差！严重不对称，必须修复")
    
    # 生成可视化
    create_symmetry_plots(left_leg_data, right_leg_data, joint_types)

def create_symmetry_plots(left_data, right_data, joint_types):
    """生成对称性可视化图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('V4b 左右腿关节对称性分析', fontsize=16, fontweight='bold')
    
    for i, joint_type in enumerate(joint_types):
        left_angles = left_data[joint_type]['angles']
        right_angles = right_data[joint_type]['angles']
        left_vels = left_data[joint_type]['velocities']
        right_vels = right_data[joint_type]['velocities']
        
        # 角度散点图
        axes[0, i].scatter(left_angles, right_angles, alpha=0.6, s=1, c='blue')
        min_val = min(min(left_angles), min(right_angles))
        max_val = max(max(left_angles), max(right_angles))
        axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完全对称线')
        axes[0, i].set_xlabel(f'左腿 {joint_type} 角度 (rad)', fontsize=10)
        axes[0, i].set_ylabel(f'右腿 {joint_type} 角度 (rad)', fontsize=10)
        axes[0, i].set_title(f'{joint_type.upper()} 角度对称性', fontsize=12, fontweight='bold')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # 速度散点图
        axes[1, i].scatter(left_vels, right_vels, alpha=0.6, s=1, c='green')
        min_val = min(min(left_vels), min(right_vels))
        max_val = max(max(left_vels), max(right_vels))
        axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完全对称线')
        axes[1, i].set_xlabel(f'左腿 {joint_type} 速度 (rad/s)', fontsize=10)
        axes[1, i].set_ylabel(f'右腿 {joint_type} 速度 (rad/s)', fontsize=10)
        axes[1, i].set_title(f'{joint_type.upper()} 速度对称性', fontsize=12, fontweight='bold')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v4c_symmetry_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 对称性分析图表已保存: v4c_symmetry_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V4b 四足机器人左右腿对称性诊断分析')
    parser.add_argument('--model', type=str, required=True, help='V4b模型路径')
    parser.add_argument('--episodes', type=int, default=3, help='分析的episode数量 (默认3)')
    
    args = parser.parse_args()
    
    print("🚀 启动V4c对称性诊断分析器...")
    analyze_symmetry(args.model, args.episodes)
    print("✅ 分析完成！")