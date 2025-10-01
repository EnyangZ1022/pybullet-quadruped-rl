# scripts/export_reward_curve.py
import os, glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

LOGDIR = "./ppo_tensorboard"   # 你的日志目录
TAG = "rollout/ep_rew_mean"    # 或 "train/value_loss" 等

def find_event_files(logdir):
    return glob.glob(os.path.join(logdir, "**", "events.*"), recursive=True)

def load_scalar(event_file, tag):
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    xs = [e.step for e in events]
    ys = [e.value for e in events]
    return np.array(xs), np.array(ys)

def main():
    xs_all, ys_all = [], []
    for f in sorted(find_event_files(LOGDIR)):
        loaded = load_scalar(f, TAG)
        if loaded is None:
            continue
        xs, ys = loaded
        xs_all.append(xs)
        ys_all.append(ys)

    if not ys_all:
        print("No data found for tag:", TAG)
        return

    # 对齐并取平均（若多并行环境/多文件）
    min_len = min(len(y) for y in ys_all)
    ys_stack = np.stack([y[:min_len] for y in ys_all], axis=0)
    xs_ref   = xs_all[0][:min_len]
    ys_mean  = ys_stack.mean(axis=0)

    plt.figure()
    plt.plot(xs_ref, ys_mean, label=TAG)
    plt.xlabel("Steps")
    plt.ylabel(TAG)
    plt.title("Training Curve")
    plt.legend()
    os.makedirs("docs/assets", exist_ok=True)
    out = "docs/assets/reward_curve.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print("Saved:", out)

if __name__ == "__main__":
    main()