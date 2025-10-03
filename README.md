# Quadruped PPO (PyBullet + SB3)

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
```

🚀 系统化训练流程
阶段1: 快速原型验证 🏃‍♂️:
python train_ppo.py
小规模快速测试（无监控开销），用于验证奖励函数/动作接口是否有效。

阶段2: 参数优化 🔧
```
# 在 train_ppo.py 中调整 total_timesteps:
python train_ppo.py
```
效果良好后，增加训练步数进行充分训练。

阶段3: 完整监控训练 📊
python train_ppo_with_metrics.py
效果稳定后，使用完整监控系统保存详细数据。

阶段4: 专业评估与分析 📈
# 生成训练指标图表
```
python scripts/export_comprehensive_metrics.py ppo_metrics_v3_20251002_011446
```

# 查看训练过程
```
tensorboard --logdir=runs/ppo_metrics_v3_20251002_011446
```

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
```bash
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
```

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
```
git clone https://github.com/EnyangZ1022/pybullet-quadruped-rl.git
cd quadruped_rl
pip install -r requirements.txt

# 快速测试
python train_ppo.py

# 完整流程
python train_ppo_with_metrics.py
python scripts/export_comprehensive_metrics.py [run_name]
```

📝 更新日志

## V3.0 (2025-10-03) - 专业指标监控系统

### 🎯 核心更新
✅ **完整指标监控系统** (`QuadrupedMetricsCallback`)
- 实时追踪 9 个关键性能指标（速度、稳定性、高度控制）
- 滑动窗口统计 + 性能优化的监控频率（每10K步）
- 自动记录到 TensorBoard，支持实时分析

✅ **自动化对比分析** (`export_comprehensive_metrics.py`)
- 生成标准化的训练对比图表（6类核心指标）
- 支持多版本性能基准对比
- 自动保存到 `docs/assets/` 目录

✅ **系统化训练流程**
- 阶段1：快速原型验证 (`train_ppo.py`)
- 阶段2：完整监控训练 (`train_ppo_with_metrics.py`) 
- 阶段3：标准评估协议 (`eval_protocol.py`)
- 阶段4：自动化分析与可视化

✅ **性能优化配置**
- RTX 4070 优化：达到 1,366 it/s 实际训练速度
- 8 并行环境 + GPU 加速 + 内存优化
- 智能监控频率平衡性能与数据质量

### 🔧 技术细节
- **机器人参数校准**: Vision60 (21.1cm), Minitaur (4.9cm), Spirit40 (17.5cm)
- **Episode 配置**: 最大1000步，累积奖励范围 500-1500（正常）
- **数据管理**: 时间戳命名 + runs/ 结构化存储
- **视频系统**: 自动模型检测 + 智能路径解析

### 📊 实际训练成果
✅ 训练配置: 750K步，8环境并行，RTX 4070
✅ 性能指标: ep_rew_mean=1290, ep_len_mean=1000步
✅ 训练效率: 753,664步/9:12 = 1,366 it/s
✅ 存储优化: ~348KB per模型，结构化管理
✅ 监控体系: 9项指标实时追踪，自动图表生成

### 🔍 训练数据解析
- **单步奖励**: -1.2 ~ 1.4（环境即时反馈）
- **累积奖励**: 1290（1000步Episode总和）
- **训练验证**: 自动200步测试确保模型质量

## V2.0
✅ 改进奖励函数
✅ 基础 TensorBoard 集成

## V1.0  
✅ 基础 PPO 四足控制
🌍 English Summary

This is a professional quadruped reinforcement learning project with comprehensive metrics monitoring, automated comparison analysis, and systematic training workflow. Optimized for RTX 4070 GPUs with standardized evaluation protocols.
