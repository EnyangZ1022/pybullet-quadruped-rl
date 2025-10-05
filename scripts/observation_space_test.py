import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quadruped_env import QuadrupedEnv
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

def analyze_observation_space(n_samples=1000):
    """观察空间有效性诊断"""
    print('🔍 观察空间有效性诊断开始...')
    
    env = QuadrupedEnv(render_mode=None)
    
    # 收集观察数据
    observations = []
    velocities = []
    positions = []
    orientations = []
    
    print(f'📊 收集{n_samples}个观察样本...')
    
    obs, _ = env.reset()
    observations.append(obs)
    
    for i in range(n_samples):
        # 随机动作采样
        action = np.random.uniform(-1, 1, 12)
        obs, reward, terminated, truncated, info = env.step(action)
        
        observations.append(obs)
        
        # 记录关键信息用于一致性检查
        if 'velocity' in info:
            velocities.append(info['velocity'])
        if 'height' in info:
            positions.append([obs[0], obs[1], obs[2]])  # 从观察中提取位置
        if 'orientation' in info:
            orientations.append(info['orientation'])
            
        if terminated or truncated:
            obs, _ = env.reset()
            
        if i % 200 == 0:
            print(f'  进度: {i}/{n_samples}')
    
    observations = np.array(observations)
    
    print(f'✅ 收集完成，观察维度: {observations.shape}')
    
    # 分析结果
    analysis_results = {}
    
    # 1. 维度验证
    print('\n📋 1. 观察空间维度验证')
    expected_dims = 37  # 3位置 + 4姿态 + 3线速度 + 3角速度 + 12关节位置 + 12关节速度
    actual_dims = observations.shape[1]
    print(f'  期望维度: {expected_dims}')
    print(f'  实际维度: {actual_dims}')
    print(f'  维度检查: {"✅ 正确" if actual_dims == expected_dims else "❌ 错误"}')
    
    analysis_results['dimension_check'] = {
        'expected': expected_dims,
        'actual': actual_dims,
        'correct': actual_dims == expected_dims
    }
    
    # 2. 数据范围合理性检查
    print('\n📊 2. 数据范围合理性检查')
    
    # 定义观察维度名称
    obs_names = [
        # 位置 (0-2)
        'pos_x', 'pos_y', 'pos_z',
        # 姿态四元数 (3-6) 
        'orn_x', 'orn_y', 'orn_z', 'orn_w',
        # 线速度 (7-9)
        'vel_x', 'vel_y', 'vel_z', 
        # 角速度 (10-12)
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z'
    ] + [f'joint_pos_{i}' for i in range(12)] + [f'joint_vel_{i}' for i in range(12)]
    
    # 定义合理范围
    reasonable_ranges = {
        'pos': (-50.0, 50.0),       # 位置范围：连续运动可能到更远距离
        'orn': (-1.0, 1.0),        # 四元数范围
        'vel': (-10.0, 10.0),       # 线速度：为高速奔跑预留空间
        'ang_vel': (-30.0, 30.0),   # 角速度：快速转动可能很大
        'joint_pos': (-5.0, 5.0),   # 关节位置：URDF限制确实在4附近，稍微放宽
        'joint_vel': (-100.0, 100.0) # 关节速度：PyBullet中关节速度可能很大
    }
    
    range_violations = []
    
    for dim in range(observations.shape[1]):
        obs_data = observations[:, dim]
        obs_name = obs_names[dim] if dim < len(obs_names) else f'dim_{dim}'
        
        # 确定合理范围类型
        if dim < 3:  # 位置
            min_reasonable, max_reasonable = reasonable_ranges['pos']
        elif dim < 7:  # 姿态
            min_reasonable, max_reasonable = reasonable_ranges['orn'] 
        elif dim < 10:  # 线速度
            min_reasonable, max_reasonable = reasonable_ranges['vel']
        elif dim < 13:  # 角速度
            min_reasonable, max_reasonable = reasonable_ranges['ang_vel']
        elif dim < 25:  # 关节位置
            min_reasonable, max_reasonable = reasonable_ranges['joint_pos']
        else:  # 关节速度
            min_reasonable, max_reasonable = reasonable_ranges['joint_vel']
            
        # 统计信息
        obs_min, obs_max = float(np.min(obs_data)), float(np.max(obs_data))
        obs_mean, obs_std = float(np.mean(obs_data)), float(np.std(obs_data))
        
        # 检查范围违规
        range_ok = min_reasonable <= obs_min and obs_max <= max_reasonable
        
        if not range_ok:
            range_violations.append({
                'dim': dim,
                'name': obs_name,
                'actual_range': [obs_min, obs_max],
                'reasonable_range': [min_reasonable, max_reasonable]
            })
            
        # 打印关键维度
        if dim < 13:  # 只显示前13个关键维度
            status = "✅" if range_ok else "❌"
            print(f'  {obs_name:12s}: 范围[{obs_min:7.3f}, {obs_max:7.3f}], 均值={obs_mean:6.3f}, 标准差={obs_std:6.3f} {status}')
    
    if range_violations:
        print(f'\n❌ 发现{len(range_violations)}个范围违规:')
        for violation in range_violations:
            print(f'  {violation["name"]}: 实际{violation["actual_range"]} vs 合理{violation["reasonable_range"]}')
    else:
        print('\n✅ 所有观察维度范围正常')
        
    analysis_results['range_check'] = {
        'violations': range_violations,
        'total_violations': len(range_violations),
        'range_ok': len(range_violations) == 0
    }
    
    # 3. NaN/Inf检测
    print('\n🔍 3. NaN/Inf异常值检测')
    
    nan_count = int(np.sum(np.isnan(observations)))
    inf_count = int(np.sum(np.isinf(observations)))
    
    print(f'  NaN值数量: {nan_count}')
    print(f'  Inf值数量: {inf_count}')
    print(f'  异常值检查: {"✅ 无异常" if (nan_count + inf_count) == 0 else "❌ 存在异常值"}')
    
    analysis_results['anomaly_check'] = {
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'anomaly_free': (nan_count + inf_count) == 0
    }
    
    # 4. 四元数归一化检查
    print('\n🎯 4. 四元数归一化检查')
    
    if observations.shape[1] >= 7:
        quaternions = observations[:, 3:7]  # orn_x, orn_y, orn_z, orn_w
        quat_norms = np.linalg.norm(quaternions, axis=1)
        norm_errors = np.abs(quat_norms - 1.0)
        
        max_norm_error = np.max(norm_errors)
        avg_norm_error = np.mean(norm_errors)
        
        quat_ok = max_norm_error < 0.01  # 允许1%的误差
        
        print(f'  四元数范数最大误差: {max_norm_error:.6f}')
        print(f'  四元数范数平均误差: {avg_norm_error:.6f}')
        print(f'  归一化检查: {"✅ 正常" if quat_ok else "❌ 归一化异常"}')
        
        analysis_results['quaternion_check'] = {
            'max_norm_error': float(max_norm_error),
            'avg_norm_error': float(avg_norm_error),
            'normalized_ok': quat_ok
        }
    
    # 5. 速度一致性检查（如果有足够数据）
    print('\n⚡ 5. 速度-位置一致性检查')
    
    if len(positions) > 10 and len(velocities) > 10:
        # 简化的一致性检查：计算位置变化率与速度的相关性
        positions = np.array(positions[:min(len(positions), len(velocities))])
        velocities = np.array(velocities[:len(positions)])
        
        if len(positions) > 1:
            # 计算位置变化率（数值微分）
            dt = 1/240.0  # 时间步长
            pos_derivatives = np.diff(positions, axis=0) / dt
            
            # 与报告的速度比较
            if len(pos_derivatives) > 0:
                vel_diff = np.mean(np.abs(pos_derivatives - velocities[1:len(pos_derivatives)+1]))
                consistency_ok = vel_diff < 0.1  # 允许0.1m/s的误差
                
                print(f'  位置微分与速度平均差异: {vel_diff:.4f} m/s')
                print(f'  一致性检查: {"✅ 一致" if consistency_ok else "❌ 不一致"}')
                
                analysis_results['velocity_consistency'] = {
                    'avg_difference': float(vel_diff),
                    'consistent': consistency_ok
                }
            else:
                print('  ⚠️ 数据不足，无法检查速度一致性')
                analysis_results['velocity_consistency'] = {'status': 'insufficient_data'}
    else:
        print('  ⚠️ 数据不足，无法检查速度一致性')
        analysis_results['velocity_consistency'] = {'status': 'insufficient_data'}
    
    # 6. 生成总结报告
    print('\n' + '='*60)
    print('📊 观察空间诊断总结')
    print('='*60)
    
    all_checks_passed = True
    
    if analysis_results['dimension_check']['correct']:
        print('✅ 维度验证: 通过')
    else:
        print('❌ 维度验证: 失败')
        all_checks_passed = False
        
    if analysis_results['range_check']['range_ok']:
        print('✅ 数据范围: 正常')
    else:
        print(f'❌ 数据范围: {analysis_results["range_check"]["total_violations"]}个违规')
        all_checks_passed = False
        
    if analysis_results['anomaly_check']['anomaly_free']:
        print('✅ 异常值检测: 无异常')
    else:
        print('❌ 异常值检测: 存在NaN/Inf')
        all_checks_passed = False
        
    if 'quaternion_check' in analysis_results and analysis_results['quaternion_check']['normalized_ok']:
        print('✅ 四元数归一化: 正常')
    elif 'quaternion_check' in analysis_results:
        print('❌ 四元数归一化: 异常')
        all_checks_passed = False
        
    print(f'\n🎯 整体评估: {"✅ 观察空间健康" if all_checks_passed else "❌ 存在问题，需要修复"}')
    
    if all_checks_passed:
        print('✅ 可以安全进入Phase 1奖励优化')
    else:
        print('⚠️ 建议先修复观察空间问题再进行奖励优化')
    
    env.close()
    
    return analysis_results

if __name__ == "__main__":
    results = analyze_observation_space(n_samples=1000)
    
    # 保存结果
    with open('observation_space_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n💾 分析结果已保存到: observation_space_analysis.json')