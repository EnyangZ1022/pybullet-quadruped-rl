# 修复关节映射问题
from quadruped_env import QuadrupedEnv
import pybullet as p
import numpy as np

def analyze_joint_structure():
    env = QuadrupedEnv(render_mode=None) 
    env.reset()
    
    print(' 分析Vision60关节结构...')
    
    # 获取所有可动关节
    movable_joints = []
    num_joints = p.getNumJoints(env.robot_id)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(env.robot_id, i)
        joint_name = joint_info[1].decode('utf-8') 
        joint_type = joint_info[2]
        joint_lower = joint_info[8]
        joint_upper = joint_info[9]
        
        # 只选择旋转关节（类型0）且有有效限位的关节
        if joint_type == 0 and joint_lower != joint_upper:
            movable_joints.append({
                'id': i,
                'name': joint_name,
                'lower': joint_lower,
                'upper': joint_upper,
                'range': joint_upper - joint_lower
            })
            
    print(f'找到{len(movable_joints)}个可动关节:')
    for joint in movable_joints:
        print(f'  关节{joint[
