#!/usr/bin/env python3
import numpy as np
from quadruped_env import QuadrupedEnv

# 创建环境并测试
env = QuadrupedEnv()
obs = env.reset()

print('=== 机器人状态检查 ===')
print(f'观察空间类型: {type(obs)}')
if isinstance(obs, tuple):
    print(f'观察数据形状: {obs[0].shape}')
    print(f'初始位置: {obs[0][:3]}')
    actual_obs = obs[0]
else:
    print(f'观察数据形状: {obs.shape}')
    print(f'初始位置: {obs[:3]}')
    actual_obs = obs

# 执行几步简单动作
print('\n=== 执行简单动作 ===')
for i in range(5):
    # 简单动作：让所有关节轻微移动
    action = np.array([0.1 * np.sin(i * 0.5)] * 12)  # 12个关节都做轻微正弦运动
    
    obs, reward, terminated, truncated, _ = env.step(action)
    
    if isinstance(obs, tuple):
        pos = obs[0][:3]
    else:
        pos = obs[:3]
    
    print(f'Step {i}: 位置={pos}, 高度={pos[2]:.4f}, 奖励={reward:.3f}')
    
    if terminated:
        print(f'机器人在第{i}步终止')
        break

env.close()
