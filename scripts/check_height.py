import pybullet as p
import pybullet_data

# 连接PyBullet
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

models_to_test = [
    'quadruped/minitaur.urdf',
    'quadruped/vision60.urdf', 
    'quadruped/spirit40.urdf'
]

for model_name in models_to_test:
    try:
        # 重置仿真
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # 加载地面和机器人
        p.loadURDF('plane.urdf')
        robot_id = p.loadURDF(model_name, [0, 0, 0.5])
        
        # 让机器人稳定下来
        for i in range(300):
            p.stepSimulation()
        
        # 获取最终位置
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        height = position[2]
        
        print(f'{model_name}: 稳定高度 = {height:.4f}米')
        
    except Exception as e:
        print(f'{model_name}: 加载失败')

p.disconnect()
