

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
            0.0,  0.9, 1.5,  # 前左腿: 髋关节，上腿，下腿
            0.0,  0.9, 1.5,  # 前右腿
            0.0,  0.9, 1.5,  # 后左腿  
            0.0,  0.9, 1.5,  # 后右腿
        ]
        
        # 目标设置
        self.target_velocity = 1.0
        self.previous_pos = None
        self.previous_yaw = 0.0
        
        # episode相关
        self.step_count = 0
        self.max_steps = 1000

        # V4a: 关节限位信息
        self.joint_limits = {}
        self.joint_action_scales = {}
        self._setup_joint_limits()
    
    def _setup_joint_limits(self):
        """V4a: 设置关节限位和缩放参数"""
        # Vision60关节限位 (来自check_height.py的实测数据)
        vision60_limits = {
            0: {'lower': -0.430, 'upper': 0.430, 'range': 0.860},
            1: {'lower': -3.142, 'upper': 3.142, 'range': 6.283},
            2: {'lower': 0.000, 'upper': 3.142, 'range': 3.142},
            4: {'lower': -0.430, 'upper': 0.430, 'range': 0.860},
            5: {'lower': -3.142, 'upper': 3.142, 'range': 6.283},
            6: {'lower': 0.000, 'upper': 3.142, 'range': 3.142},
            8: {'lower': -0.430, 'upper': 0.430, 'range': 0.860},
            9: {'lower': -3.142, 'upper': 3.142, 'range': 6.283},
            10: {'lower': 0.000, 'upper': 3.142, 'range': 3.142},
            12: {'lower': -0.430, 'upper': 0.430, 'range': 0.860},
            13: {'lower': -3.142, 'upper': 3.142, 'range': 6.283},
            14: {'lower': 0.000, 'upper': 3.142, 'range': 3.142},
        }
    
        self.joint_limits = vision60_limits
        
        # 根据关节范围设置独立的动作缩放
        for joint_id, limits in self.joint_limits.items():
            # 使用关节范围的30%作为单步动作的最大幅度
            self.joint_action_scales[joint_id] = limits['range'] * 0.3

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

    def _calculate_symmetry_reward(self, ema_alpha=0.3, w=0.15, eps=1e-6, min_scale=0.05):
        """
        基于关节速度的对角线协调性奖励 - 适配V4d版本
        ema_alpha: 滑动平均参数 (0.2-0.3适合实时控制)
        w: 奖励最大值 (可稍微提高以增强信号)
        min_scale: 最小归一化尺度，避免在静止时奖励爆炸
        """
        
        # 获取关节速度 
        v = []
        for joint_id in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_id)
            v.append(joint_state[1])  # 关节速度
        v = np.array(v, dtype=np.float32)  # 明确指定数据类型
    
        # 初始化EMA状态 - 修复数据类型问题
        if not hasattr(self, "_v_ema"):
            self._v_ema = np.array(v.copy(), dtype=np.float32)  # 确保numpy数组
            self._scale_ema = 0.1  # 初始尺度估计
        
        # 确保_v_ema是numpy数组（防御性编程）
        if not isinstance(self._v_ema, np.ndarray):
            self._v_ema = np.array(self._v_ema, dtype=np.float32)
        
        # 低通滤波 (EMA) - 现在数据类型一致
        self._v_ema = ema_alpha * v + (1 - ema_alpha) * self._v_ema
        v_s = self._v_ema
        
        # 其余代码保持不变...
        FL = v_s[0:3]   
        FR = v_s[3:6]   
        RL = v_s[6:9]   
        RR = v_s[9:12]  
        
        # 方向校正
        sign = np.array([+1, +1, +1], dtype=np.float32)
        
        FLc = sign * FL
        FRc = sign * FR  
        RLc = sign * RL
        RRc = sign * RR
        
        # 关键改进1: 检查对角线速度方向一致性
        direction_penalty = 0.0
        for i in range(3):  # hip, thigh, calf
            # 对角线1 (FL-RR): 如果速度方向相反，施加惩罚
            if FLc[i] * RRc[i] < -eps:  # 方向相反
                direction_penalty += 0.1 * (abs(FLc[i]) + abs(RRc[i]))
            
            # 对角线2 (FR-RL): 如果速度方向相反，施加惩罚
            if FRc[i] * RLc[i] < -eps:  # 方向相反  
                direction_penalty += 0.1 * (abs(FRc[i]) + abs(RLc[i]))
        
        # 对角线差值计算
        err_d1 = np.abs(FLc - RRc)  # 对角线1误差
        err_d2 = np.abs(FRc - RLc)  # 对角线2误差
        
        # 关键改进2: 更鲁棒的自适应归一化
        current_scale = np.sqrt(np.mean(v_s**2)) + eps
        # 对尺度也进行EMA平滑，避免突变
        self._scale_ema = ema_alpha * current_scale + (1 - ema_alpha) * self._scale_ema
        scale = max(self._scale_ema, min_scale)  # 确保最小尺度
        
        # 关键改进3: 组合误差计算，考虑方向惩罚
        normalized_err_d1 = np.clip(err_d1 / scale, 0, 5)  # 防止异常值
        normalized_err_d2 = np.clip(err_d2 / scale, 0, 5)
        
        cost = (normalized_err_d1.mean() + normalized_err_d2.mean()) + direction_penalty / scale
        
        # 高斯核奖励映射
        symmetry_reward = w * np.exp(-cost)
        
        return float(symmetry_reward)

    def _calculate_reward(self):
        """计算奖励 - 全面改进"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # 1. 加大前进奖励 - 更多鼓励X轴正向移动
        forward_velocity = vel[0]
        forward_reward = min(forward_velocity * 5.0, 5.0)
        
        # 2. 高度奖励 - 保持合适高度
        height = pos[2]
        if 0.18 < height < 0.25:      # 基于实际站立高度调整
            height_reward = 2.0
        elif height > 0.12:           # 降低最低要求
            height_reward = 1.0
        else:
            height_reward = -10.0
        
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
        
        #V4b: 增加膝关节姿态奖励 - 防止小腿收缩作弊策略
        knee_extension_reward = 0.0
        # 膝关节索引：每条腿的第3个关节 (calf joints)
        knee_indices = [2, 5, 8, 11]  # 基于joint_indices顺序
        for knee_idx in knee_indices:
            if knee_idx < len(joint_positions):
                knee_angle = joint_positions[knee_idx]
                # 鼓励膝关节在合理范围内伸展(基于URDF: -2.618 to 2.618)
                # 理想角度接近0(中性位置)，避免过度屈曲
                if abs(knee_angle) < 0.8:  # 在±0.8弧度内为良好姿态
                    knee_extension_reward += 0.2
                elif abs(knee_angle) < 1.5:  # 适中姿态
                    knee_extension_reward += 0.1
                else:  # 过度屈曲惩罚
                    knee_extension_reward -= 0.1 * abs(knee_angle)

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

        # 10.v4c加入对称性奖励 - 鼓励左右腿协调运动
        symmetry_reward = self._calculate_symmetry_reward()
        
        
        total_reward = (forward_reward + height_reward + orientation_reward + 
                       direction_reward + joint_reward + knee_extension_reward + smoothness_reward +
                       stability_penalty + lateral_penalty + alive_reward + symmetry_reward)
        
        self.previous_positions = joint_positions.copy()

        return total_reward, {
            'forward': forward_reward,
            'height': height_reward,
            'orientation': orientation_reward,
            'direction': direction_reward,
            'joint': joint_reward,
            'smoothness': smoothness_reward,
            'stability': stability_penalty,
            'lateral': lateral_penalty,
            'knee_extension': knee_extension_reward,
            'symmetry': symmetry_reward,
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
                # 计算目标角度
                base_angle = self.default_joint_angles[i % len(self.default_joint_angles)]
                
                # V4a: 使用关节特定的动作缩放
                if joint_id in self.joint_action_scales:
                    action_scale = self.joint_action_scales[joint_id]
                else:
                    action_scale = 0.5  # 默认缩放（兼容性）
                
                target_angle = base_angle + action[i] * action_scale
                
                # V4a: 根据URDF限位裁剪目标角度
                if joint_id in self.joint_limits:
                    limits = self.joint_limits[joint_id]
                    target_angle = np.clip(target_angle, limits['lower'], limits['upper'])
                
                #if joint_id == 2:  # 调试默认角度用膝关节
                #    print(f"关节{joint_id}: base={base_angle:.3f}, action={action[i]:.3f}, scale={action_scale:.3f}, target={target_angle:.3f}")

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
        
        # 获取额外状态信息用于指标监控
        vel, ang_vel = p.getBaseVelocity(self.robot_id)

        # 构建info字典
        info = {
            'velocity': vel,           # (vx, vy, vz) 
            'angular_velocity': ang_vel,  # 新增：(wx, wy, wz)
            'height': pos[2],         # 高度
            'orientation': euler,     # (roll, pitch, yaw)
            'terminated': terminated,     # 新增：是否终止
            'fall_termination': terminated and pos[2] < 0.05,  # 新增：是否因跌倒终止
            'action': self.current_action.copy(),
            'reward_info': reward_info  # 原有的奖励分解
        }

        return observation, reward, terminated, truncated, info
    
    def close(self):
        """关闭环境"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
