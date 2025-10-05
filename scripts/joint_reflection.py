import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quadruped_env import QuadrupedEnv
import numpy as np
import pybullet as p

def test_improved_joint_mapping():
    """改进的关节映射测试 - 修复后验证"""
    print('🔍 改进的关节映射测试...')
    
    env = QuadrupedEnv(render_mode=None)
    
    # 首先验证缩放修改
    print('📊 动作缩放验证:')
    calf_joints = [2, 6, 10, 14]
    for joint_id in calf_joints:
        if joint_id in env.joint_action_scales:
            scale = env.joint_action_scales[joint_id]
            limits = env.joint_limits[joint_id]
            ratio = scale / limits['range']
            print(f'  关节{joint_id} (calf): range={limits["range"]:.3f}, scale={scale:.3f}, ratio={ratio:.1%}')
    
    print('\n🎯 关节映射测试:')
    print(f'joint_indices: {env.joint_indices}')
    
    mapping_results = {}
    correct_count = 0
    
    for action_idx in range(12):
        print(f'\n--- 测试动作索引 {action_idx} ---')
        
        # 重置环境
        obs, _ = env.reset()
        
        # 获取初始状态
        initial_states = {}
        for joint_id in env.joint_indices:
            joint_state = p.getJointState(env.robot_id, joint_id)
            initial_states[joint_id] = joint_state[0]
        
        # 施加单一动作
        action = np.zeros(12)
        action[action_idx] = 1.0
        
        # 执行50步
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"  ⚠️ 机器人在第{step}步倒下")
                break
        
        # 计算所有关节变化
        changes = {}
        max_change = 0
        max_change_joint = None
        
        for joint_id in env.joint_indices:
            joint_state = p.getJointState(env.robot_id, joint_id)
            change = abs(joint_state[0] - initial_states[joint_id])
            changes[joint_id] = change
            
            if change > max_change:
                max_change = change
                max_change_joint = joint_id
        
        # 期望的关节ID
        expected_joint_id = env.joint_indices[action_idx]
        
        print(f'  动作{action_idx} -> 期望关节{expected_joint_id}, 实际响应关节{max_change_joint}')
        print(f'  最大变化: {max_change:.4f} rad')
        
        # 显示所有显著变化（>0.01 rad）
        significant_changes = [(jid, change) for jid, change in changes.items() if change > 0.01]
        if len(significant_changes) > 1:
            print(f'  📋 显著变化: {significant_changes}')
        
        # 验证映射
        mapping_correct = (expected_joint_id == max_change_joint and max_change > 0.05)
        
        if mapping_correct:
            print(f'  ✅ 映射正确')
            correct_count += 1
        else:
            print(f'  ❌ 映射问题 - 期望{expected_joint_id}, 实际{max_change_joint}')
            if max_change < 0.05:
                print(f'    (变化太小: {max_change:.4f} < 0.05)')
        
        mapping_results[action_idx] = {
            'expected': expected_joint_id,
            'actual': max_change_joint,
            'change': max_change,
            'correct': mapping_correct
        }
    
    # 总结报告
    print(f'\n{"="*50}')
    print(f'📊 关节映射总结')
    print(f'{"="*50}')
    print(f'正确映射: {correct_count}/12')
    print(f'映射成功率: {correct_count/12*100:.1f}%')
    
    # 列出问题关节
    problem_joints = [idx for idx, result in mapping_results.items() if not result['correct']]
    if problem_joints:
        print(f'\n❌ 问题动作索引: {problem_joints}')
        for idx in problem_joints:
            result = mapping_results[idx]
            expected = result['expected']
            actual = result['actual']
            change = result['change']
            print(f'  动作{idx}: 期望关节{expected} -> 实际关节{actual} (变化{change:.4f})')
    else:
        print('\n✅ 所有关节映射正确!')
    
    env.close()
    return mapping_results

if __name__ == "__main__":
    results = test_improved_joint_mapping()