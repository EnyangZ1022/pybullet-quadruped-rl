# Installation
# 📦 安装与环境配置

本项目基于 **PyBullet** 和 **Stable-Baselines3 (PPO)**，支持在消费级 GPU（如 RTX 4070）上运行。  
推荐使用 **Python 3.9 ~ 3.11**，并在虚拟环境中安装依赖。

---

1.创建虚拟环境

建议使用 `venv` 或 `conda`，例如使用 venv：

'''
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
'''

2.安装依赖

克隆仓库并安装依赖：
'''
git clone https://github.com/EnyangZ1022/pybullet-quadruped-rl.git
cd quadruped_rl
pip install -r requirements.txt
'''

3.检查安装是否成功

运行以下命令确认依赖正常：
'''
python - <<'PY'
import torch, pybullet
print("CUDA available:", torch.cuda.is_available())
print("PyBullet version:", pybullet.__version__)

'''

4.运行测试环境
'''
python scripts/test_env.py
'''

5.常见问题

Q1: GPU 没有启用

请确保安装了与 CUDA 版本匹配的 PyTorch。
参考 PyTorch 官网

Q2: ImportError: No module named pybullet

重新安装依赖：pip install --upgrade pybullet

Q3: 无法渲染 GUI

远程服务器环境请设置 render_mode=None，使用 DIRECT 模式。

6.下一步

阅读 奖励函数设计
启动训练脚本：python train_ppo.py
