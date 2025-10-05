from quadruped_env import QuadrupedEnv
import numpy as np
import pybullet as p

def test_correct_joint_mapping():
    print(' 修正的关节映射测试...')
    
    env = QuadrupedEnv(render_mode=None)
    env.reset()
    
    print(f'joint_indices: {env.joint_indices}')
    
    mapping_results = {}
    
    for action_idx in range(12):
        print(f'\n--- 测试动作索引 {action_idx} ---')
        
        # 重置环境
        obs, _ = env.reset()
        
        # 获取初始关节状态
        initial_states = {}
        for i, joint_id in enumerate(env.joint_indices):
            joint_state = p.getJointState(env.robot_id, joint_id)
            initial_states[joint_id] = joint_state[0]
        
        # 施加单一动作
        action = np.zeros(12)
        action[action_idx] = 1.0
        
        # 执行50步
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        # 获取最终状态并计算变化
        changes = {}
        max_change = 0
        max_change_joint = None
        
        for i, joint_id in enumerate(env.joint_indices):
            joint_state = p.getJointState(env.robot_id, joint_id)
            final_angle = joint_state[0]
            change = abs(final_angle - initial_states[joint_id])
            changes[joint_id] = change
            
            if change > max_change:
                max_change = change
                max_change_joint = joint_id
        
        # 检查映射正确性
        expected_joint_id = env.joint_indices[action_idx]  # 这才是正确的期望关节
        
        print(f'  动作索引{action_idx} -> 期望关节{expected_joint_id}')
        print(f'  最大响应关节: {max_change_joint}, 变化量: {max_change:.4f}')
        
        # 验证映射
        mapping_correct = (expected_joint_id == max_change_joint and max_change > 0.01)
        
        if mapping_correct:
            print(f'   映射正确')
        else:
            print(f'   映射错误 - 期望关节{expected_joint_id}，实际响应关节{max_change_joint}')
            
        mapping_results[action_idx] = {
            'expected': expected_joint_id,
            'actual': max_change_joint,
            'change': max_change,
            'correct': mapping_correct
        }
    
    # 生成总结
    correct_count = sum(1 for r in mapping_results.values() if r['correct'])
    print(f'\n 总结: {correct_count}/12 个动作映射正确')
    
    env.close()
    return mapping_results

if __name__ == '__main__':
    results = test_correct_joint_mapping()
