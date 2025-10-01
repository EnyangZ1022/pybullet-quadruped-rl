import pybullet as p
import pybullet_data

# 测试不同机器人模型
models = [
    'quadruped/minitaur.urdf',
    'quadruped/vision60.urdf', 
    'quadruped/spirit40.urdf',
    'quadruped/quadruped.urdf'
]

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

for model in models:
    try:
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF('plane.urdf')
        
        robot_id = p.loadURDF(model, [0, 0, 0.5])
        
        # 稳定几步
        for _ in range(100):
            p.stepSimulation()
        
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        print(f'{model}: 高度 = {pos[2]:.4f}m')
        
    except Exception as e:
        print(f'{model}: 加载失败 - {e}')

p.disconnect()
EOF && python test_models.py
