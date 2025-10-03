#!/usr/bin/env python3
"""
V4b 支撑力诊断分析器
检查各腿支撑质量，特别关注左前腿支撑不足问题
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

def analyze_support(model_path, episodes=5):
    """分析四腿支撑力和支撑质量"""
    
    # 创建环境
    env = QuadrupedEnv(render_mode=False)
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 数据收集结构
    leg_names = ['FL', 'FR', 'RL', 'RR']  # Front Left, Front Right, Rear Left, Rear Right
    
    leg_data = {
        'FL': {'hip': [], 'thigh': [], 'calf': [], 'support_quality': []},
        'FR': {'hip': [], 'thigh': [], 'calf': [], 'support_quality': []},
        'RL': {'hip': [], 'thigh': [], 'calf': [], 'support_quality': []},
        'RR': {'hip': [], 'thigh': [], 'calf': [], 'support_quality': []}
    }
    
    body_orientation_data = {'roll': [], 'pitch': [], 'yaw': []}
    support_patterns = []
    
    print(f"🔍 开始支撑力分析，共{episodes}个episode...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_steps = 0
        
        print(f"  📊 分析Episode {episode+1}/{episodes}...")
        
        while episode_steps < 1000:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            # 获取关节状态和身体姿态
            joint_positions = []
            joint_velocities = []
            
            for joint_id in env.joint_indices:
                joint_state = p.getJointState(env.robot_id, joint_id)
                joint_positions.append(joint_state[0])  # 角度
                joint_velocities.append(joint_state[1])  # 速度
            
            # 获取身体姿态
            pos, orn = p.getBasePositionAndOrientation(env.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            
            body_orientation_data['roll'].append(euler[0])
            body_orientation_data['pitch'].append(euler[1])
            body_orientation_data['yaw'].append(euler[2])
            
            # 分析各腿关节角度 (每腿3个关节: hip, thigh, calf)
            leg_joints = {
                'FL': [joint_positions[0], joint_positions[1], joint_positions[2]],   # 0,1,2
                'FR': [joint_positions[3], joint_positions[4], joint_positions[5]],   # 3,4,5
                'RL': [joint_positions[6], joint_positions[7], joint_positions[8]],   # 6,7,8
                'RR': [joint_positions[9], joint_positions[10], joint_positions[11]]  # 9,10,11
            }
            
            # 记录各腿关节数据和计算支撑质量
            current_support_pattern = []
            
            for leg_name in leg_names:
                hip_angle, thigh_angle, calf_angle = leg_joints[leg_name]
                
                leg_data[leg_name]['hip'].append(hip_angle)
                leg_data[leg_name]['thigh'].append(thigh_angle)
                leg_data[leg_name]['calf'].append(calf_angle)
                
                # 支撑质量评估
                support_quality = calculate_support_quality(hip_angle, thigh_angle, calf_angle)
                leg_data[leg_name]['support_quality'].append(support_quality)
                
                # 判断是否为支撑腿
                is_supporting = support_quality > 0.6
                current_support_pattern.append(is_supporting)
            
            support_patterns.append(current_support_pattern)
            episode_steps += 1
            
            if done or truncated:
                break
    
    env.close()
    
    # 支撑力分析
    print("\n" + "="*60)
    print("🎯 V4b 支撑力诊断报告")
    print("="*60)
    
    # 1. 各腿支撑质量分析
    print("\n🦿 各腿支撑质量分析:")
    
    leg_support_scores = {}
    leg_failure_rates = {}
    
    for leg_name in leg_names:
        support_qualities = np.array(leg_data[leg_name]['support_quality'])
        
        avg_support = np.mean(support_qualities)
        support_std = np.std(support_qualities)
        failure_rate = np.sum(support_qualities < 0.3) / len(support_qualities) * 100
        
        leg_support_scores[leg_name] = avg_support
        leg_failure_rates[leg_name] = failure_rate
        
        print(f"  {leg_name}腿: 平均支撑质量={avg_support:.3f}, 标准差={support_std:.3f}, 失效率={failure_rate:.1f}%")
        
        # 支撑质量评级
        if avg_support > 0.8:
            grade = "优秀 ✅"
        elif avg_support > 0.6:
            grade = "良好 🟢"
        elif avg_support > 0.4:
            grade = "一般 🟡"
        else:
            grade = "差 ❌"
        
        print(f"       支撑评级: {grade}")
    
    # 2. 左前腿特别分析
    print(f"\n🔍 左前腿(FL)专项诊断:")
    
    fl_hip = np.array(leg_data['FL']['hip'])
    fl_thigh = np.array(leg_data['FL']['thigh'])
    fl_calf = np.array(leg_data['FL']['calf'])
    fl_support = np.array(leg_data['FL']['support_quality'])
    
    print(f"  关节角度范围:")
    print(f"    Hip:   [{np.min(fl_hip):.3f}, {np.max(fl_hip):.3f}] rad")
    print(f"    Thigh: [{np.min(fl_thigh):.3f}, {np.max(fl_thigh):.3f}] rad")
    print(f"    Calf:  [{np.min(fl_calf):.3f}, {np.max(fl_calf):.3f}] rad")
    
    # 检测异常角度
    fl_abnormal_hip = np.sum(np.abs(fl_hip) > 1.0) / len(fl_hip) * 100
    fl_abnormal_thigh = np.sum(np.abs(fl_thigh) > 2.5) / len(fl_thigh) * 100
    fl_abnormal_calf = np.sum(np.abs(fl_calf) > 2.5) / len(fl_calf) * 100
    
    print(f"  异常角度比例:")
    print(f"    Hip异常 (>1.0 rad): {fl_abnormal_hip:.1f}%")
    print(f"    Thigh异常 (>2.5 rad): {fl_abnormal_thigh:.1f}%")
    print(f"    Calf异常 (>2.5 rad): {fl_abnormal_calf:.1f}%")
    
    # 3. 支撑模式分析
    print(f"\n🔄 支撑模式分析:")
    
    support_patterns_array = np.array(support_patterns)
    
    # 统计各种支撑模式
    four_leg_support = np.sum(np.sum(support_patterns_array, axis=1) == 4) / len(support_patterns_array) * 100
    three_leg_support = np.sum(np.sum(support_patterns_array, axis=1) == 3) / len(support_patterns_array) * 100
    two_leg_support = np.sum(np.sum(support_patterns_array, axis=1) == 2) / len(support_patterns_array) * 100
    unstable_support = np.sum(np.sum(support_patterns_array, axis=1) < 2) / len(support_patterns_array) * 100
    
    print(f"  四腿支撑: {four_leg_support:.1f}%")
    print(f"  三腿支撑: {three_leg_support:.1f}%")
    print(f"  两腿支撑: {two_leg_support:.1f}%")
    print(f"  不稳定支撑(<2腿): {unstable_support:.1f}%")
    
    # 各腿参与支撑的比例
    leg_support_participation = np.mean(support_patterns_array, axis=0) * 100
    
    print(f"\n  各腿支撑参与度:")
    for i, leg_name in enumerate(leg_names):
        print(f"    {leg_name}腿: {leg_support_participation[i]:.1f}%")
    
    # 4. 身体姿态与支撑关系分析
    print(f"\n🏃 身体姿态分析:")
    
    roll_data = np.array(body_orientation_data['roll'])
    pitch_data = np.array(body_orientation_data['pitch'])
    
    avg_roll = np.mean(roll_data)
    avg_pitch = np.mean(pitch_data)
    roll_std = np.std(roll_data)
    pitch_std = np.std(pitch_data)
    
    print(f"  平均Roll角: {avg_roll:.4f} rad ({np.degrees(avg_roll):.2f}°)")
    print(f"  平均Pitch角: {avg_pitch:.4f} rad ({np.degrees(avg_pitch):.2f}°)")
    print(f"  Roll稳定性: {roll_std:.4f} rad")
    print(f"  Pitch稳定性: {pitch_std:.4f} rad")
    
    # 检测身体倾斜问题
    if abs(avg_roll) > 0.1:
        roll_direction = "左" if avg_roll < 0 else "右"
        print(f"  ❌ 检测到身体{roll_direction}倾！可能与腿部支撑不均有关")
    
    if abs(avg_pitch) > 0.1:
        pitch_direction = "前" if avg_pitch > 0 else "后"
        print(f"  ❌ 检测到身体{pitch_direction}倾！可能与前后腿支撑差异有关")
    
    # 5. 问题诊断与修复建议
    print(f"\n🚨 支撑问题诊断:")
    
    # 找出最差的腿
    worst_leg = min(leg_support_scores.keys(), key=lambda x: leg_support_scores[x])
    best_leg = max(leg_support_scores.keys(), key=lambda x: leg_support_scores[x])
    
    print(f"  最差支撑腿: {worst_leg} (支撑质量: {leg_support_scores[worst_leg]:.3f})")
    print(f"  最佳支撑腿: {best_leg} (支撑质量: {leg_support_scores[best_leg]:.3f})")
    
    # 特别关注左前腿
    if worst_leg == 'FL':
        print(f"  ❌ 确认左前腿支撑最差！这解释了观察到的左前方下沉")
        print(f"     左前腿失效率: {leg_failure_rates['FL']:.1f}%")
    
    # 支撑不平衡检测
    support_imbalance = max(leg_support_scores.values()) - min(leg_support_scores.values())
    if support_imbalance > 0.3:
        print(f"  ❌ 检测到严重支撑不平衡！差异: {support_imbalance:.3f}")
    
    # 生成可视化
    create_support_plots(leg_data, leg_names, body_orientation_data, support_patterns_array)
    
    # 修复建议
    print(f"\n💡 V4c支撑力修复建议:")
    if worst_leg == 'FL':
        print("🔧 建议1: 增加左前腿特定支撑奖励")
        print("🔧 建议2: 优化左前腿关节角度约束")
        print("🔧 建议3: 添加身体Roll角平衡奖励")
    
    print("🔧 建议4: 添加四腿支撑平衡奖励")
    print("🔧 建议5: 惩罚单腿过度负载")

def calculate_support_quality(hip_angle, thigh_angle, calf_angle):
    """
    计算单腿支撑质量
    基于关节角度的合理性和稳定性
    """
    quality = 1.0
    
    # Hip角度评估 (理想范围: -0.5 to 0.5)
    if abs(hip_angle) < 0.5:
        hip_score = 1.0
    elif abs(hip_angle) < 1.0:
        hip_score = 0.8
    else:
        hip_score = 0.4
    
    # Thigh角度评估 (理想范围: -1.0 to 2.0, 支撑时偏向正值)
    if -1.0 <= thigh_angle <= 2.0:
        thigh_score = 1.0
    elif -2.0 <= thigh_angle <= 2.5:
        thigh_score = 0.7
    else:
        thigh_score = 0.3
    
    # Calf角度评估 (理想范围: -1.0 to 1.0, 支撑时接近0)
    if abs(calf_angle) < 1.0:
        calf_score = 1.0
    elif abs(calf_angle) < 2.0:
        calf_score = 0.6
    else:
        calf_score = 0.2
    
    # 综合评分
    quality = (hip_score * 0.3 + thigh_score * 0.4 + calf_score * 0.3)
    
    return quality

def create_support_plots(leg_data, leg_names, orientation_data, support_patterns):
    """生成支撑力分析可视化图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('V4b Support Force Analysis', fontsize=16, fontweight='bold')
    
    # 1. 各腿支撑质量对比
    support_means = [np.mean(leg_data[leg]['support_quality']) for leg in leg_names]
    support_stds = [np.std(leg_data[leg]['support_quality']) for leg in leg_names]
    
    bars = axes[0, 0].bar(leg_names, support_means, yerr=support_stds, capsize=5, 
                          color=['red' if leg == 'FL' else 'blue' for leg in leg_names])
    axes[0, 0].set_ylabel('Support Quality')
    axes[0, 0].set_title('Average Support Quality by Leg')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, mean_val in zip(bars, support_means):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom')
    
    # 2. 左前腿详细分析
    fl_support = leg_data['FL']['support_quality']
    time_steps = np.arange(len(fl_support))
    
    axes[0, 1].plot(time_steps, fl_support, color='red', alpha=0.7)
    axes[0, 1].axhline(y=0.6, color='orange', linestyle='--', label='Good Support Threshold')
    axes[0, 1].axhline(y=0.3, color='red', linestyle='--', label='Poor Support Threshold')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Support Quality')
    axes[0, 1].set_title('FL Leg Support Quality Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 支撑模式分布
    support_counts = np.sum(support_patterns, axis=1)
    unique, counts = np.unique(support_counts, return_counts=True)
    
    axes[0, 2].bar(unique, counts, color='green', alpha=0.7)
    axes[0, 2].set_xlabel('Number of Supporting Legs')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Support Pattern Distribution')
    axes[0, 2].set_xticks(range(5))
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 各腿关节角度分布 (以FL为例)
    joint_types = ['hip', 'thigh', 'calf']
    colors = ['blue', 'green', 'red']
    
    for i, joint_type in enumerate(joint_types):
        fl_angles = leg_data['FL'][joint_type]
        axes[1, 0].hist(fl_angles, bins=30, alpha=0.6, color=colors[i], label=f'FL {joint_type}')
    
    axes[1, 0].set_xlabel('Joint Angle (rad)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('FL Leg Joint Angle Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 身体姿态变化
    time_steps = np.arange(len(orientation_data['roll']))
    
    axes[1, 1].plot(time_steps, np.degrees(orientation_data['roll']), label='Roll', color='red')
    axes[1, 1].plot(time_steps, np.degrees(orientation_data['pitch']), label='Pitch', color='blue')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Angle (degrees)')
    axes[1, 1].set_title('Body Orientation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 6. 各腿支撑参与度
    leg_participation = np.mean(support_patterns, axis=0) * 100
    
    bars = axes[1, 2].bar(leg_names, leg_participation, 
                          color=['red' if leg == 'FL' else 'blue' for leg in leg_names])
    axes[1, 2].set_ylabel('Support Participation (%)')
    axes[1, 2].set_title('Leg Support Participation Rate')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, participation in zip(bars, leg_participation):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{participation:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('v4c_support_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 支撑力分析图表已保存: v4c_support_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V4c 支撑力和支撑质量诊断分析')
    parser.add_argument('--model', type=str, required=True, help='V4cb模型路径')
    parser.add_argument('--episodes', type=int, default=3, help='分析的episode数量 (默认3)')
    
    args = parser.parse_args()
    
    print("🚀 启动V4c支撑力诊断分析器...")
    analyze_support(args.model, args.episodes)
    print("✅ 分析完成！")