#!/usr/bin/env python3
import numpy as np
from quadruped_env import QuadrupedEnv
import pybullet as p
import time

# 创建环境
env = QuadrupedEnv()

print('=== 测试机器人行为 ===')

# 重置环境
obs = env.reset()
print(f'初始观察空间维度: {obs[0].shape}')
print(f'初始位置: {obs[0][:3]}')

# 测试几个简单动作
for episode in range(2):
    print(f'\n--- Episode {episode+1} ---')
    obs = env.reset()
    
    for step in range(50):
        # 简单的周期性动作 - 模拟步态
        t = step * 0.1
        action = np.array([
            0.3 * np.sin(t),      # 前左腿髋关节
            -0.5 * np.cos(t),     # 前左腿膝关节
            0.3 * np.sin(t + np.pi),  # 前右腿髋关节
            -0.5 * np.cos(t + np.pi), # 前右腿膝关节
            0.3 * np.sin(t + np.pi),  # 后左腿髋关节
            -0.5 * np.cos(t + np.pi), # 后左腿膝关节
            0.3 * np.sin(t),      # 后右腿髋关节
            -0.5 * np.cos(t),     # 后右腿膝关节
            0, 0, 0, 0  # 其他关节保持静止
        ])
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if step % 10 == 0:
            pos = obs[0][:3]
            print(f'Step {step}: 位置={pos}, 奖励={reward:.3f}')
        
        if terminated or truncated:
            print(f'Episode结束在第{step}步')
            break
        
        time.sleep(0.01)  # 稍微放慢速度观察

env.close()
print('\n测试完成！')
