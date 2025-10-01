# Quadruped PPO (PyBullet + SB3)

这是一个基于 **PyBullet** 和 **Stable-Baselines3 (PPO)** 的四足机器人强化学习项目，  
适合在 RTX 4070 等消费级显卡上运行。

## ✨ 特性
- 自定义 Gymnasium 环境：`QuadrupedEnv`
- 改进奖励函数：前进/高度/姿态/平滑/方向等
- 轻量级 PPO 配置（适合中等算力 GPU）
- 一键训练与测试，支持 TensorBoard

## 📂 目录结构

```
├── quadruped_env.py # 核心环境
├── train_ppo.py # 训练脚本
├── requirements.txt # 依赖清单
├── LICENSE # 许可证
├── README.md # 项目主页
│
├── docs/ # 文档
│ ├── installation.md # 安装与配置
│ ├── reward.md # 奖励函数说明
│ ├── training_tips.md # 训练技巧
│ └── results.md # 实验结果
│
├── scripts/ # 实验 & 调试脚本
│ ├── check_height.py
│ ├── debug_env.py
│ ├── record_video.py
│ ├── test_env.py
│ ├── test_models.py
│ ├── test_robot_behavior.py
│ └── test_simple.py
│
├── archive/ # 历史版本
│ ├── quadruped_env_backup.py
│ └── quadruped_env_v2.py
│
└── models/ # 模型保存目录（gitignore 掉）
└── README.md
```


## 🚀 快速开始
```bash
git clone <你的仓库地址>
cd quadruped_rl
pip install -r requirements.txt
python train_ppo.py

🧪 查看训练过程
tensorboard --logdir=ppo_tensorboard/

📄 更多文档
docs/installation.md
docs/reward.md
docs/training_tips.md
docs/results.md

## English Summary
This is a lightweight quadruped reinforcement learning project based on **PyBullet** and **Stable-Baselines3 (PPO)**.  
It provides a custom Gymnasium environment, improved reward shaping, and a training pipeline suitable for RTX 4070 Laptop GPUs.
