

import math

#!/usr/bin/env python3
"""
轻量级四足机器人PyBullet环境
适合RTX 4070进行PPO强化学习训练
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import time

#!/usr/bin/env python3
"""
改进版四足机器人环境 - 解决步态和控制问题
"""

class QuadrupedEnv(gym.Env):
    """改进的四足机器人环境"""
    
    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
        
        # PyBullet 设置
        self.physics_client = None
        self.time_step = 1./240.
        
        # 动作和观察空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
        )
        
        # 机器人相关
        self.robot_id = None
        self.initial_pos = [0, 0, 0.5]
        self.initial_orn = [0, 0, 0, 1]
        
        # 控制相关
        self.max_force = 20.0
        self.joint_indices = []
        self.previous_action = None  # 用于动作平滑
        
        # 理想的关节角度 (站立姿态)
        self.default_joint_angles = [
            0.0,  0.9, -1.8,  # 前左腿: 髋关节，上腿，下腿
            0.0,  0.9, -1.8,  # 前右腿
            0.0,  0.9, -1.8,  # 后左腿  
            0.0,  0.9, -1.8,  # 后右腿
        ]
        
        # 目标设置
        self.target_velocity = 1.0
        self.previous_pos = None
        self.previous_yaw = 0.0
        
        # episode相关
        self.step_count = 0
        self.max_steps = 1000
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        if self.physics_client is None:
            if self.render_mode == 'human':
                self.physics_client = p.connect(p.GUI)
                # 优化相机视角
                p.resetDebugVisualizerCamera(
                    cameraDistance=3.0,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=[0, 0, 0.5]
                )
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        
        # 加载地面
        p.loadURDF("plane.urdf")
        
        # 使用vision60机器人 - 更稳定的高度
        self.robot_id = p.loadURDF("quadruped/vision60.urdf", 
                                 self.initial_pos, 
                                 self.initial_orn)
        
        # 获取关节信息
        self.joint_indices = []
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:
                self.joint_indices.append(i)
        
        # 限制到12个主要关节
        self.joint_indices = self.joint_indices[:12]
        
        # 设置初始关节角度 - 让机器人呈现正确的站立姿态
        for i, joint_id in enumerate(self.joint_indices):
            if i < len(self.default_joint_angles):
                p.resetJointState(self.robot_id, joint_id, self.default_joint_angles[i])
            else:
                p.resetJointState(self.robot_id, joint_id, 0.0)
        
        # 让机器人稳定下来
        for _ in range(200):
            p.stepSimulation()
        
        self.step_count = 0
        self.previous_action = np.zeros(12)
        self.previous_pos = np.array(self.initial_pos)
        
        # 获取初始朝向
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        self.previous_yaw = euler[2]
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        """获取观察值"""
        # 机器人状态
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # 关节状态
        joint_positions = []
        joint_velocities = []
        for joint_id in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_id)
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
        
        # 构建观察向量 (与原环境保持一致的37维)
        obs = []
        obs.extend(pos)           # 3维位置
        obs.extend(orn)           # 4维姿态
        obs.extend(vel)           # 3维线速度
        obs.extend(ang_vel)       # 3维角速度
        obs.extend(joint_positions)  # 12维关节位置
        obs.extend(joint_velocities) # 12维关节速度
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self):
        """计算奖励 - 全面改进"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # 1. 前进奖励 - 鼓励X轴正向移动
        forward_velocity = vel[0]
        forward_reward = min(forward_velocity * 3.0, 3.0)
        
        # 2. 高度奖励 - 保持合适高度
        height = pos[2]
        if 0.3 < height < 0.8:
            height_reward = 2.0
        elif height > 0.15:
            height_reward = 1.0
        else:
            height_reward = -10.0  # 严重惩罚过低
        
        # 3. 姿态稳定奖励
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        
        # 惩罚过度倾斜
        orientation_reward = -5.0 * (abs(roll) + abs(pitch))
        
        # 朝向一致性奖励 - 保持朝前
        yaw_diff = abs(yaw - self.previous_yaw)
        if yaw_diff > math.pi:
            yaw_diff = 2 * math.pi - yaw_diff
        direction_reward = -2.0 * yaw_diff
        
        # 4. 关节角度奖励 - 鼓励合理的腿部姿态
        joint_positions = []
        for joint_id in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_id)
            joint_positions.append(joint_state[0])
        
        joint_reward = 0.0
        for i, (actual, desired) in enumerate(zip(joint_positions[:len(self.default_joint_angles)], 
                                                 self.default_joint_angles)):
            joint_diff = abs(actual - desired)
            if joint_diff < 0.5:  # 接近理想角度
                joint_reward += 0.1
            else:
                joint_reward -= 0.1 * joint_diff
        
        # 5. 动作平滑奖励 - 减少抖动
        if self.previous_action is not None:
            action_diff = np.sum(np.abs(self.current_action - self.previous_action))
            smoothness_reward = -0.1 * action_diff
        else:
            smoothness_reward = 0.0
        
        # 6. 稳定性奖励 - 惩罚过度摇摆
        stability_penalty = -0.1 * (abs(ang_vel[0]) + abs(ang_vel[1]) + abs(ang_vel[2]))
        
        # 7. 侧向移动惩罚 - 避免无目的转向
        lateral_penalty = -0.5 * abs(vel[1])
        
        # 8. 存活奖励
        alive_reward = 0.2
        
        total_reward = (forward_reward + height_reward + orientation_reward + 
                       direction_reward + joint_reward + smoothness_reward +
                       stability_penalty + lateral_penalty + alive_reward)
        
        return total_reward, {
            'forward': forward_reward,
            'height': height_reward,
            'orientation': orientation_reward,
            'direction': direction_reward,
            'joint': joint_reward,
            'smoothness': smoothness_reward,
            'stability': stability_penalty,
            'lateral': lateral_penalty
        }
    
    def step(self, action):
        """执行一步 - 改进的动作处理"""
        # 动作平滑化 - 减少抖动
        if self.previous_action is not None:
            smooth_factor = 0.3  # 平滑系数
            action = smooth_factor * action + (1 - smooth_factor) * self.previous_action
        
        self.current_action = action
        
        # 应用动作到关节 - 使用PD控制
        for i, joint_id in enumerate(self.joint_indices):
            if i < len(action):
                # 限制动作范围并添加到默认角度
                target_angle = self.default_joint_angles[i % len(self.default_joint_angles)]
                target_angle += action[i] * 0.5  # 限制变化幅度
                
                p.setJointMotorControl2(
                    self.robot_id, joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=self.max_force,
                    positionGain=0.1,  # P控制增益
                    velocityGain=0.01  # D控制增益
                )
        
        # 执行物理仿真
        p.stepSimulation()
        
        # 获取观察和奖励
        observation = self._get_observation()
        reward, reward_info = self._calculate_reward()
        
        # 检查终止条件
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        
        terminated = False
        if pos[2] < 0.05:  # 高度过低
            terminated = True
        
        # 检查翻倒 - 更宽松的条件
        euler = p.getEulerFromQuaternion(orn)
        if abs(euler[0]) > 1.2 or abs(euler[1]) > 1.2:
            terminated = True
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        # 更新状态
        self.previous_action = self.current_action.copy()
        self.previous_pos = np.array(pos)
        self.previous_yaw = euler[2]
        
        return observation, reward, terminated, truncated, reward_info
    
    def close(self):
        """关闭环境"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
