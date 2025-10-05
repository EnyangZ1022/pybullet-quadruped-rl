from quadruped_env import QuadrupedEnv
import pybullet as p

env = QuadrupedEnv(render_mode=None)
env.reset()

print(' 检查关节配置:')
print(f'joint_indices长度: {len(env.joint_indices)}')
print(f'joint_indices内容: {env.joint_indices}')

print('\n 所有关节信息:')
num_joints = p.getNumJoints(env.robot_id)
for i in range(num_joints):
    joint_info = p.getJointInfo(env.robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    print(f'  关节{i}: {joint_name}, 类型{joint_type}')

env.close()
