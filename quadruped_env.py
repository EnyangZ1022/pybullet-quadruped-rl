# scripts/export_reward_curve.py
import os, glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

LOGDIR = "./ppo_tensorboard"         # 你的 SB3 日志目录
OUT_PNG = "docs/assets/reward_curve_v3.png"  # 导出图片名
MAX_POINTS = 1000                     # 最多保留多少个点（自动抽点）
EMA_ALPHA = 0.2                       # 指数滑动平均系数（0=不平滑，建议0.1~0.3）
PLOT_PER_STEP_REWARD = True           # True=画“每步平均奖励”；False=画“每回合总奖励”

TAG_REWARD = "rollout/ep_rew_mean"
TAG_EPLEN  = "rollout/ep_len_mean"
TAG_XAXIS  = "time/total_timesteps"   # 用这个作横轴，避免“只有一万step”的问题

def find_event_files(logdir):
    return glob.glob(os.path.join(logdir, "**", "events.*"), recursive=True)

def load_scalar(event_file, tag):
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    xs = np.array([e.step for e in events], dtype=np.int64)   # TB 的内部 step（不一定是 total steps）
    ys = np.array([e.value for e in events], dtype=np.float64)
    return xs, ys

def concat_series(files, tag):
    xs_all, ys_all = [], []
    for f in sorted(files):
        xs, ys = load_scalar(f, tag)
        if xs is None: 
            continue
        xs_all.append(xs)
        ys_all.append(ys)
    if not ys_all:
        return None, None
    # 简单做法：按文件顺序拼接（SB3 通常每次运行一个events文件，拼接近似时间顺序）
    xs_cat = np.concatenate(xs_all)
    ys_cat = np.concatenate(ys_all)
    # 以 x 排序并去重（同 x 取最后一个）
    order = np.argsort(xs_cat)
    xs_cat, ys_cat = xs_cat[order], ys_cat[order]
    _, idx = np.unique(xs_cat, return_index=True)
    xs_cat, ys_cat = xs_cat[idx], ys_cat[idx]
    return xs_cat, ys_cat

def ema(y, alpha):
    if alpha <= 0: 
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]
    return out

def downsample(x, y, max_points=1000):
    n = len(x)
    if n <= max_points:
        return x, y
    stride = int(np.ceil(n / max_points))
    return x[::stride], y[::stride]

def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    files = find_event_files(LOGDIR)
    if not files:
        print("No TensorBoard event files found in", LOGDIR)
        return

    # 横轴：用 total_timesteps（更符合直觉）
    x_step, x_ttotal = concat_series(files, TAG_XAXIS)
    if x_ttotal is None:
        print(f"Tag '{TAG_XAXIS}' not found. Fallback to TB step.")
        # 若取不到 total_timesteps，就回退用 TB 内部 step
        x_step, x_ttotal = concat_series(files, TAG_REWARD)

    # 纵轴：奖励
    _, y_rew = concat_series(files, TAG_REWARD)
    if y_rew is None:
        print(f"Tag '{TAG_REWARD}' not found.")
        return

    # 可选：每步平均奖励 = ep_rew_mean / ep_len_mean
    if PLOT_PER_STEP_REWARD:
        _, y_len = concat_series(files, TAG_EPLEN)
        if y_len is not None and len(y_len) >= 1:
            y = y_rew / np.maximum(y_len, 1e-6)
            y_label = "Avg Reward per Step"
        else:
            print(f"Tag '{TAG_EPLEN}' not found, fallback to ep_rew_mean.")
            y = y_rew
            y_label = "Episode Reward (sum)"
    else:
        y = y_rew
        y_label = "Episode Reward (sum)"

    # 对齐（防止一个更长一个更短）
    L = min(len(x_ttotal), len(y))
    x, y = x_ttotal[:L], y[:L]

    # 平滑 + 抽点
    y_smooth = ema(y, EMA_ALPHA)
    x_ds, y_ds = downsample(x, y_smooth, MAX_POINTS)

    # 画图
    plt.figure()
    plt.plot(x_ds, y_ds, label=y_label)
    plt.xlabel("Total Timesteps")
    plt.ylabel(y_label)
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print("Saved:", OUT_PNG, "Points:", len(x_ds))

if __name__ == "__main__":
    main()