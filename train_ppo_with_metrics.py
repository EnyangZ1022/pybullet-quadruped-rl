#!/usr/bin/env python3
"""
带完整指标监控的PPO训练脚本 - 性能优化版
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import os
from datetime import datetime
from quadruped_env import QuadrupedEnv
from quadruped_metrics import QuadrupedMetricsCallback

def print_final_metrics_summary():
    """训练结束后打印关键指标摘要"""
    print("\n" + "="*50)
    print("🎯 训练完成 - 关键性能指标摘要")
    print("="*50)
    
    print(f"📊 稳态性能指标:")
    print(f"   前进速度 (vx_mean): 待查看TensorBoard")     
    print(f"   横向漂移 (|vy|_mean): 待查看TensorBoard")   
    print(f"   身体稳定性 (roll_rms): 待查看TensorBoard")
    print(f"   身体稳定性 (pitch_rms): 待查看TensorBoard")
    print(f"   航向稳定性 (yaw_rate_rms): 待查看TensorBoard")
    
    print(f"\n🛡️  鲁棒性指标:")
    print(f"   跌倒频率: 待查看TensorBoard")     
    print(f"   每步平均奖励: 待查看TensorBoard")        
    
    print(f"\n📈 使用以下命令查看详细指标:")
    print(f"   tensorboard --logdir=runs/")
    print("="*50)

def main():
    # 版本和时间戳
    version = "ppo_metrics_v4c"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建运行目录
    run_name = f"{version}_{timestamp}"
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"=== 四足机器人PPO训练 (带指标监控) ===")
    print(f"运行名称: {run_name}")
    print(f"日志目录: {log_dir}")

    # GPU检查 - 与原始脚本保持一致
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("使用CPU训练")
        device = 'cpu'

    # 创建环境 - 保持原始的并行环境数量
    print("创建训练环境...")
    env = make_vec_env(lambda: QuadrupedEnv(render_mode=None), n_envs=8)

    # 创建指标监控回调 - 降低监控频率以提升性能
    metrics_callback = QuadrupedMetricsCallback(
        eval_freq=10000,  # 从2000改为10000，降低监控频率
        verbose=1
    )

    # PPO模型配置 - 完全保持原始脚本的配置
    print("初始化PPO模型...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=512,         # 与原始相同
        batch_size=64,       # 与原始相同
        n_epochs=4,          # 与原始相同
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,       # 确保GPU使用
        verbose=1,
        tensorboard_log=log_dir
    )

    print("开始训练...")
    print("训练参数:")
    print(f"- 设备: {device}")
    print(f"- 并行环境: 8")
    print(f"- 总步数: 750000")
    print(f"- 指标监控频率: 每10000步")

    # 训练 (带指标监控)
    model.learn(
        total_timesteps=750000,  # 与原始相同
        callback=metrics_callback,
        progress_bar=True
    )

    # 保存模型
    model_path = f"{log_dir}/{version}_model"
    model.save(model_path)

    # 训练结束打印总结
    print(f"训练完成!")
    print(f"模型保存: {model_path}.zip")
    print_final_metrics_summary()
    print(f"查看训练过程: tensorboard --logdir={log_dir}")
    print(f"生成对比图: python scripts/export_comprehensive_metrics.py {run_name}")




if __name__ == "__main__":
    main()