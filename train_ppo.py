#!/usr/bin/env python3
"""
轻量级PPO训练脚本
适合RTX 4070 GPU
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch
from quadruped_env import QuadrupedEnv

def main():
    print("=== 四足机器人PPO训练 ===")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("使用CPU训练")
        device = 'cpu'
    
    # 创建环境
    print("创建训练环境...")
    env = make_vec_env(lambda: QuadrupedEnv(render_mode=None), n_envs=8)
    
    # 轻量级PPO配置（适合RTX 4070）
    print("初始化PPO模型...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=512,         # 减小步数
        batch_size=64,       # 小批次
        n_epochs=4,          # 少epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )
    
    print("开始训练...")
    print("训练参数:")
    print(f"- 设备: {device}")
    print(f"- 总步数: 750000 (轻量级)")
    print(f"- 网络结构: MLP")
    
    # 开始训练
    model.learn(
        total_timesteps=750000,  # 适中的训练量
        progress_bar=True
    )
    
    # 保存模型
    print("保存模型...")
    model.save("ppo_quadruped_light")
    
    print("训练完成!")
    
    # 测试训练结果
    print("测试训练好的模型...")
    test_model(model)

def test_model(model):
    """测试训练好的模型"""
    env = QuadrupedEnv(render_mode=None)
    obs, _ = env.reset()
    
    total_reward = 0
    steps = 0
    
    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        steps += 1
        
        print(f"Step {i}: Reward = {reward:.3f}")
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.3f}, Steps: {steps}")
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
    
    env.close()

if __name__ == "__main__":
    main()
