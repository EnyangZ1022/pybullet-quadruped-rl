#!/usr/bin/env python3
"""
完整的指标导出脚本 - 支持单run和多run对比
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import sys


def load_tensorboard_data(run_dir, tag):
    """从tensorboard日志加载数据"""
    event_files = list(Path(run_dir).rglob("events.out.tfevents.*"))
    
    if not event_files:
        return None, None
    
    all_data = []
    for event_file in event_files:
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        
        if tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            for event in events:
                all_data.append((event.step, event.value))
    
    if not all_data:
        return None, None
        
    # 排序并平滑
    all_data.sort(key=lambda x: x[0])
    steps, values = zip(*all_data)
    
    return np.array(steps), smooth_data(np.array(values))


def smooth_data(values, alpha=0.1):
    """指数移动平均平滑"""
    if len(values) == 0:
        return values
        
    smoothed = [values[0]]
    for val in values[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)


def export_single_run_metrics(run_name):
    """导出单个run的指标"""
    run_dir = f"runs/{run_name}"
    output_dir = Path("docs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要导出的指标
    metrics = {
        'Episode Reward': 'rollout/ep_rew_mean',
        'Forward Velocity': 'quadruped/vx_mean',
        'Roll RMS': 'quadruped/roll_rms',
        'Pitch RMS': 'quadruped/pitch_rms',
        'Height Stability': 'quadruped/height_error_rms',
        'Stability Score': 'quadruped/stability_score'
    }
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Metrics - {run_name}', fontsize=16)
    axes = axes.flatten()
    
    for i, (title, tag) in enumerate(metrics.items()):
        steps, values = load_tensorboard_data(run_dir, tag)
        
        if steps is not None:
            axes[i].plot(steps, values, 'b-', alpha=0.8, linewidth=2)
            axes[i].set_title(title)
            axes[i].set_xlabel('Steps')
            axes[i].set_ylabel(title)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f"{run_name}_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 单run指标图保存: {output_path}")


def export_comparison_metrics(run_names):
    """导出多run对比指标"""
    output_dir = Path("docs/assets")
    
    # 关键对比指标
    key_metrics = {
        'episode_reward_comp.png': ('rollout/ep_rew_mean', 'Episode Reward'),
        'vx_mean_comp.png': ('quadruped/vx_mean', 'Forward Velocity (m/s)'),
        'roll_rms_comp.png': ('quadruped/roll_rms', 'Roll RMS (rad)'),
        'stability_score_comp.png': ('quadruped/stability_score', 'Stability Score')
    }
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for filename, (tag, ylabel) in key_metrics.items():
        plt.figure(figsize=(12, 8))
        
        for i, run_name in enumerate(run_names):
            run_dir = f"runs/{run_name}"
            steps, values = load_tensorboard_data(run_dir, tag)
            
            if steps is not None:
                color = colors[i % len(colors)]
                plt.plot(steps, values, label=run_name, 
                        color=color, alpha=0.8, linewidth=2)
        
        plt.xlabel('Training Steps')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 对比图保存: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  单run: python export_comprehensive_metrics.py run_name")
        print("  多run对比: python export_comprehensive_metrics.py run1 run2 run3")
        print("示例:")
        print("  python export_comprehensive_metrics.py ppo_metrics_v1_20251002_1030")
        return
    
    run_names = sys.argv[1:]
    
    print(f"=== 导出 {len(run_names)} 个run的指标 ===")
    
    # 单run详细指标
    for run_name in run_names:
        if os.path.exists(f"runs/{run_name}"):
            export_single_run_metrics(run_name)
        else:
            print(f"⚠️  警告: runs/{run_name} 不存在")
    
    # 多run对比
    if len(run_names) > 1:
        valid_runs = [r for r in run_names if os.path.exists(f"runs/{r}")]
        if len(valid_runs) > 1:
            export_comparison_metrics(valid_runs)
    
    print("=== 导出完成 ===")
    print("查看 docs/assets/ 目录获取生成的图表")


if __name__ == "__main__":
    main()