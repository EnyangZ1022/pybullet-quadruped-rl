#!/usr/bin/env python3
"""
测量所有四足机器人模型在正确站立姿态下的高度
"""
import pybullet as p
import pybullet_data
import numpy as np

# 不同机器人的理想站立关节角度
robot_joint_configs = {
    'quadruped/minitaur.urdf': [0.0] * 8,  # Minitaur使用不同的关节配置
    'quadruped/vision60.urdf': [
        0.0,  0.9, -1.8,  # 前左腿
        0.0,  0.9, -1.8,  # 前右腿  
        0.0,  0.9, -1.8,  # 后左腿
        0.0,  0.9, -1.8,  # 后右腿
    ],
    'quadruped/spirit40.urdf': [
        0.0,  0.9, -1.8,  # 与vision60类似
        0.0,  0.9, -1.8,   
        0.0,  0.9, -1.8,   
        0.0,  0.9, -1.8,   
    ]
}

def test_robot_height(model_name, joint_angles):
    """测试单个机器人的站立高度"""
    print(f"\n=== 测试 {model_name} ===")
    
    try:
        # 重置仿真
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # 加载地面和机器人
        p.loadURDF('plane.urdf')
        robot_id = p.loadURDF(model_name, [0, 0, 0.5])

        # 获取可控关节
        joint_indices = []
        num_joints = p.getNumJoints(robot_id)
        print(f"总关节数: {num_joints}")
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:
                joint_indices.append(i)

        # 限制到主要关节
        joint_indices = joint_indices[:len(joint_angles)]
        print(f"使用关节数: {len(joint_indices)}")

        # 设置站立姿态
        for i, joint_id in enumerate(joint_indices):
            if i < len(joint_angles):
                p.resetJointState(robot_id, joint_id, joint_angles[i])

        # 让机器人稳定下来
        print("让机器人稳定到站立姿态...")
        heights = []
        
        for i in range(600):  # 增加稳定时间
            p.stepSimulation()
            
            # 记录最后100步的高度
            if i >= 500:
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                heights.append(pos[2])
            
            # 每100步输出一次高度
            if i % 100 == 0:
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                print(f"第{i}步: 高度 = {pos[2]:.4f}m")

        # 计算平均稳定高度
        avg_height = np.mean(heights)
        std_height = np.std(heights)
        
        print(f"\n--- 结果 ---")
        print(f"平均稳定高度: {avg_height:.4f}m ± {std_height:.4f}m")
        print(f"高度范围: {min(heights):.4f}m - {max(heights):.4f}m")
        
        return avg_height, std_height

    except Exception as e:
        print(f"测试失败: {e}")
        return None, None

def main():
    # 连接PyBullet
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    print("=== 四足机器人站立高度测试 ===")
    
    results = {}
    
    for model_name in robot_joint_configs.keys():
        joint_angles = robot_joint_configs[model_name]
        avg_height, std_height = test_robot_height(model_name, joint_angles)
        
        if avg_height is not None:
            results[model_name] = {
                'height': avg_height,
                'std': std_height,
                'recommended_target': round(avg_height * 0.95, 3)  # 稍微低一点作为目标
            }

    p.disconnect()

    # 输出汇总结果
    print("\n" + "="*50)
    print("=== 汇总结果 ===")
    print("="*50)
    
    for model_name, data in results.items():
        print(f"\n{model_name}:")
        print(f"  站立高度: {data['height']:.4f}m ± {data['std']:.4f}m")
        print(f"  建议目标高度: {data['recommended_target']}m")
    
    # 针对vision60的特别建议
    if 'quadruped/vision60.urdf' in results:
        vision60_target = results['quadruped/vision60.urdf']['recommended_target']
        print(f"\n🎯 针对您的项目 (vision60):")
        print(f"   建议在 quadruped_metrics.py 中设置:")
        print(f"   target_height = {vision60_target}  # vision60站立目标高度")

if __name__ == "__main__":
    main()