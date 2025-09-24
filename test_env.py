#!/usr/bin/env python3
"""
环境测试脚本
"""

from quadruped_env import QuadrupedEnv
import numpy as np

def test_environment():
    print("测试环境...")
    
    # 创建环境
    env = QuadrupedEnv()
    
    # 重置环境
    obs, info = env.reset()
    print(f"观测空间维度: {obs.shape}")
    print(f"动作空间: {env.action_space}")
    
    # 运行几步
    total_reward = 0
    for i in range(10):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        print(f"Step {i}: Reward = {reward:.3f}")
        
        if terminated or truncated:
            print("Episode terminated")
            obs, info = env.reset()
    
    print(f"Total reward: {total_reward:.3f}")
    env.close()
    print("环境测试完成!")

if __name__ == "__main__":
    test_environment()
