#!/usr/bin/env python3
"""
从TensorBoard日志中读取训练指标
用法: python scripts/read_tensorboard_metrics.py runs/ppo_metrics_v4b_20251003_195933
"""

import os
import sys
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_latest_run():
    """找到最新的训练run"""
    runs = glob.glob('runs/ppo_metrics_v4b_*')
    if runs:
        return max(runs, key=os.path.getctime)
    return None

def read_final_metrics(log_dir):
    """从TensorBoard日志中读取最终指标"""
    try:
        # 找到PPO_1目录
        ppo_dir = os.path.join(log_dir, 'PPO_1')
        if not os.path.exists(ppo_dir):
            print(f'❌ 未找到TensorBoard日志目录: {ppo_dir}')
            return None
            
        # 创建事件累加器
        ea = EventAccumulator(ppo_dir)
        ea.Reload()
        
        # 获取可用的标量
        scalars = ea.Tags()['scalars']
        print(f'🔍 发现 {len(scalars)} 个可用指标')
        
        # 读取关键指标的最终值
        metrics = {}
        
        # 基础训练指标
        basic_metrics = {
            'rollout/ep_rew_mean': 'ep_rew_mean',
            'rollout/ep_len_mean': 'ep_len_mean',
            'time/total_timesteps': 'total_timesteps'
        }
        
        for full_name, short_name in basic_metrics.items():
            if full_name in scalars:
                values = ea.Scalars(full_name)
                if values:
                    metrics[short_name] = values[-1].value
                    
        # 自定义quadruped指标
        quadruped_metrics = ['vx_mean', 'vy_abs_mean', 'roll_rms', 'pitch_rms', 'height_error_rms']
        for metric in quadruped_metrics:
            full_name = f'quadruped/{metric}'
            if full_name in scalars:
                values = ea.Scalars(full_name)
                if values:
                    metrics[metric] = values[-1].value
                    
        return metrics
        
    except Exception as e:
        print(f'❌ 读取指标时出错: {e}')
        return None

def print_metrics_summary(metrics):
    """打印指标摘要"""
    print("\n" + "="*50)
    print("🎯 训练完成 - 关键性能指标摘要")
    print("="*50)
    
    # 训练基本信息
    if 'total_timesteps' in metrics:
        print(f"📈 训练信息:")
        print(f"   总训练步数: {int(metrics['total_timesteps']):,} steps")
        
        if 'ep_rew_mean' in metrics and 'ep_len_mean' in metrics:
            reward_per_step = metrics['ep_rew_mean'] / metrics['ep_len_mean']
            print(f"   平均episode奖励: {metrics['ep_rew_mean']:.1f}")
            print(f"   平均episode长度: {int(metrics['ep_len_mean'])}")
            print(f"   每步平均奖励: {reward_per_step:.3f}")
    
    # 运动性能指标
    print(f"\n📊 稳态性能指标:")
    
    if 'vx_mean' in metrics:
        vx = metrics['vx_mean']
        status = '✅ 达标' if vx >= 0.6 else '❌ 未达标'
        print(f"   前进速度 (vx_mean): {vx:.3f} m/s [{status} ≥0.6]")
    else:
        print(f"   前进速度 (vx_mean): 数据未找到")
        
    if 'vy_abs_mean' in metrics:
        vy = metrics['vy_abs_mean']
        status = '✅ 达标' if vy < 0.05 else '❌ 未达标'
        print(f"   横向漂移 (|vy|_mean): {vy:.3f} m/s [{status} <0.05]")
    else:
        print(f"   横向漂移 (|vy|_mean): 数据未找到")
    
    # 稳定性指标
    print(f"\n🛡️  身体稳定性指标:")
    
    if 'roll_rms' in metrics:
        roll = metrics['roll_rms']
        status = '✅ 达标' if roll < 0.1 else '❌ 未达标'
        print(f"   Roll稳定性 (roll_rms): {roll:.3f} rad [{status} <0.10]")
    else:
        print(f"   Roll稳定性 (roll_rms): 数据未找到")
        
    if 'pitch_rms' in metrics:
        pitch = metrics['pitch_rms']
        status = '✅ 达标' if pitch < 0.1 else '❌ 未达标'
        print(f"   Pitch稳定性 (pitch_rms): {pitch:.3f} rad [{status} <0.10]")
    else:
        print(f"   Pitch稳定性 (pitch_rms): 数据未找到")
        
    if 'height_error_rms' in metrics:
        height_err = metrics['height_error_rms']
        print(f"   高度控制精度: {height_err:.3f} m")
    else:
        print(f"   高度控制精度: 数据未找到")
    
    print("\n💡 使用以下命令查看详细训练过程:")
    print(f"   tensorboard --logdir=runs/")
    print("="*50)

def main():
    if len(sys.argv) == 1:
        # 没有参数，自动找最新的run
        log_dir = find_latest_run()
        if not log_dir:
            print("❌ 未找到训练结果，请指定log_dir")
            print("用法: python scripts/read_tensorboard_metrics.py runs/ppo_metrics_v4b_xxx")
            sys.exit(1)
        print(f"🔍 自动使用最新训练结果: {log_dir}")
    else:
        log_dir = sys.argv[1]
    
    if not os.path.exists(log_dir):
        print(f"❌ 目录不存在: {log_dir}")
        sys.exit(1)
    
    print(f"📂 读取训练日志: {log_dir}")
    metrics = read_final_metrics(log_dir)
    
    if metrics:
        print_metrics_summary(metrics)
    else:
        print("❌ 无法读取指标数据")
        print("请确保:")
        print("1. TensorBoard日志文件存在")
        print("2. 已安装tensorboard包")
        print("3. 训练已完成并保存了日志")

if __name__ == '__main__':
    main()