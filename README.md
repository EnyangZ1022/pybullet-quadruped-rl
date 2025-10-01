# Quadruped PPO (PyBullet + SB3) - 专业版 

这是一个基于 **PyBullet** 和 **Stable-Baselines3 (PPO)** 的四足机器人强化学习项目，  
适合在 RTX 4070 等消费级显卡上运行，具备完整的指标监控和评估体系。

---

## ✨ 核心特性
- 🤖 自定义 Gymnasium 环境：`QuadrupedEnv`
- 📊 **完整指标监控系统**：前进速度、姿态稳定性、动作平滑度等
- 🔄 **系统化训练流程**：快速原型 → 完整训练 → 专业评估
- 📈 **自动化对比分析**：多版本性能对比图表
- ⚡ 针对 RTX 4070 优化配置

---

## 📂 项目结构 

```
quadruped_rl/
├── 🔧 核心文件
│   ├── quadruped_env.py              # 四足机器人环境
│   ├── quadruped_metrics.py          # 指标监控系统
│   ├── train_ppo.py                  # 快速训练脚本
│   └── train_ppo_with_metrics.py     # 完整监控训练
│
├── 📁 训练管理
│   ├── runs/                         # 训练结果（按版本组织）
│   │   ├── ppo_v1_20251002_1030/
│   │   ├── ppo_metrics_v3_20251002_011446/
│   │   └── ...
│   └── requirements.txt
│
├── 🧪 实验工具
│   ├── scripts/
│   │   ├── export_comprehensive_metrics.py   # 指标导出
│   │   ├── eval_protocol.py                  # 标准评估
│   │   ├── record_video.py
│   │   └── ... (其他调试脚本)
│
└── 📚 文档与结果
    ├── docs/
    │   ├── assets/              # 生成的对比图表
    │   ├── installation.md
    │   ├── reward.md
    │   └── results.md
    └── README.md
'''

🚀 系统化训练流程
阶段1: 快速原型验证 🏃‍♂️:
python train_ppo.py
小规模快速测试（无监控开销），用于验证奖励函数/动作接口是否有效。

阶段2: 参数优化 🔧
# 在 train_ppo.py 中调整 total_timesteps:
python train_ppo.py
效果良好后，增加训练步数进行充分训练。

阶段3: 完整监控训练 📊
python train_ppo_with_metrics.py
效果稳定后，使用完整监控系统保存详细数据。

阶段4: 专业评估与分析 📈
# 生成训练指标图表
python scripts/export_comprehensive_metrics.py ppo_metrics_v3_20251002_011446

# 查看训练过程
tensorboard --logdir=runs/ppo_metrics_v3_20251002_011446

🎯 核心监控指标

A. 运动性能指标

前进速度 (vx_mean): 目标方向移动效率

侧向偏移 (vy_abs_mean): 直线行走稳定性

高度控制 (height_error_rms): 身体高度维持精度

B. 稳定性指标

Roll/Pitch RMS: 姿态稳定性

动作平滑度: 控制信号连续性

稳定性得分: 速度与稳定性的综合评价

C. 训练质量指标

Episode 奖励趋势: 学习进程监控

Value/Policy Loss: 网络收敛状态

Entropy: 探索与利用平衡

📊 使用示例
'''
# 快速验证算法改进
python train_ppo.py

# 完整训练与监控
python train_ppo_with_metrics.py

# 单个训练 run 的详细分析
python scripts/export_comprehensive_metrics.py ppo_metrics_v3_20251002_011446

# 多个版本的性能对比
python scripts/export_comprehensive_metrics.py run1 run2 run3

# 实时监控训练过程
tensorboard --logdir=runs/latest_run/

# 查看生成的对比图表
ls docs/assets/*.png
'''

⚙️ 性能优化

RTX 4070 推荐配置

并行环境数: 8

批次大小: 64

训练步数: 512

期望性能: ~1800-2500 it/s

监控系统优化

监控频率: 每 10000 步

数据窗口: 50 个 episode

优先核心指标，减少 I/O 开销

📈 实验管理最佳实践

版本控制：使用时间戳自动命名，保留关键版本完整数据

对比分析：生成标准化的对比图表，建立性能基准线

结果复现：固定随机种子，记录关键超参数，保存环境配置

🔧 快速开始
'''
git clone https://github.com/EnyangZ1022/pybullet-quadruped-rl.git
cd quadruped_rl
pip install -r requirements.txt

# 快速测试
python train_ppo.py

# 完整流程
python train_ppo_with_metrics.py
python scripts/export_comprehensive_metrics.py [run_name]

'''

📝 更新日志

V3.0 (2025-10-02)

✅ 完整指标监控系统

✅ 自动化对比分析

✅ 系统化训练流程

✅ 性能优化配置

V2.0

✅ 改进奖励函数

✅ 基础 TensorBoard 集成

V1.0

✅ 基础 PPO 四足控制


🌍 English Summary

This is a professional quadruped reinforcement learning project with comprehensive metrics monitoring, automated comparison analysis, and systematic training workflow. Optimized for RTX 4070 GPUs with standardized evaluation protocols.
