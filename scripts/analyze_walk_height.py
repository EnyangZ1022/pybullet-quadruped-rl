#!/usr/bin/env python3
"""
分析机器人高度表现
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from quadruped_env import QuadrupedEnv
from stable_baselines3 import PPO

def analyze_height_performance(model_path="ppo_quadruped_light.zip"):
    """分析机器人的高度表现"""
    print("=== 机器人高度性能分析 ===")
    print(f"加载模型: {model_path}")
    
    try:
        model = PPO.load(model_path)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    env = QuadrupedEnv(render_mode=None)
    print("✅ 环境创建成功!")
    
    # 分析多个episode
    all_heights = []
    
    for episode in range(3):
        print(f"\n📊 分析Episode {episode + 1}/3...")
        
        heights = []
        obs, _ = env.reset()
        
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 收集高度数据
            height = info['height']
            heights.append(height)
            
            # 每200步输出一次实时高度
            if step % 200 == 0:
                print(f"  Step {step}: Height = {height:.3f}m")
            
            if terminated or truncated:
                print(f"  Episode在第{step}步结束")
                break
        
        # Episode统计
        heights = np.array(heights)
        all_heights.extend(heights)
        
        print(f"  Episode {episode + 1} 高度统计:")
        print(f"    平均高度: {np.mean(heights):.3f}m")
        print(f"    高度范围: {np.min(heights):.3f} - {np.max(heights):.3f}m")
        print(f"    标准差: {np.std(heights):.3f}m")
    
    # 总体分析
    all_heights = np.array(all_heights)
    target_height = 0.21
    rms_error = np.sqrt(np.mean((all_heights - target_height)**2))
    
    print(f"\n" + "="*50)
    print("📈 总体高度性能分析")
    print("="*50)
    print(f"🎯 目标高度: {target_height}m")
    print(f"📊 实际表现:")
    print(f"   平均高度: {np.mean(all_heights):.3f}m")
    print(f"   高度范围: {np.min(all_heights):.3f} - {np.max(all_heights):.3f}m")
    print(f"   标准差: {np.std(all_heights):.3f}m")
    print(f"   RMS误差: {rms_error:.3f}m")
    print(f"   与目标差异: {np.mean(all_heights) - target_height:+.3f}m")
    
    # 高度分布分析
    below_target = np.sum(all_heights < target_height) / len(all_heights) * 100
    above_target = np.sum(all_heights > target_height) / len(all_heights) * 100
    
    print(f"\n📊 高度分布:")
    print(f"   低于目标: {below_target:.1f}%")
    print(f"   高于目标: {above_target:.1f}%")
    
    env.close()
    print(f"\n✅ 分析完成!")

def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "ppo_quadruped_light.zip"
    
    analyze_height_performance(model_path)

if __name__ == "__main__":
    main()