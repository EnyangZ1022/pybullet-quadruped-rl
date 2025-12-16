

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
            # 根据关节类型设置不同缩放
            if limits['range'] > 5.0:  # 大范围关节（thigh joints：1,5,9,13）
                self.joint_action_scales[joint_id] = limits['range'] * 0.4
            elif limits['range'] > 2.0:  # 中等范围关节（calf joints：2,6,10,14）  
                self.joint_action_scales[joint_id] = limits['range'] * 0.8  # 更大缩放
            else:  # 小范围关节（hip joints：0,4,8,12）
                self.joint_action_scales[joint_id] = limits['range'] * 0.7

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
        self.joint_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        
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
        """Phase 1：零速度跟踪 + 稳定性优先 (基于系统诊断优化)"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        # —— 辅助函数：死区 & 分段奖励 ——
        def deadband_abs(x, tol):
            """死区函数: |x| <= tol -> 0, else |x|-tol"""
            ax = abs(x)
            return 0.0 if ax <= tol else (ax - tol)

        def band_reward(x, lo, hi, full=1.0, mid=0.5):
            """分段奖励: x在[lo,hi]给full分；边缘区域给mid分；出界惩罚"""
            if lo <= x <= hi:
                return full
            # 边缘容忍区间 (5%宽度)
            width = hi - lo
            delta = 0.05 * max(width, 1e-6)
            if lo - delta <= x < lo:
                return mid * (x - (lo - delta)) / delta
            if hi < x <= hi + delta:
                return mid * ((hi + delta) - x) / delta
            return -0.2  # 轻微出界惩罚

        # === 1) 零速度跟踪 (带死区容忍) ===
        # 基于观察空间诊断：实际速度范围 vel_x:[-1.3,0.3], vel_y:[-1.2,0.6]
        # 允许微小抖动避免过度敏感
        vx_penalty = -deadband_abs(vel[0], 0.02) * 6.0    # 前进/后退惩罚
        vy_penalty = -deadband_abs(vel[1], 0.02) * 10.0   # 侧漂惩罚更重
        yaw_rate_penalty = -deadband_abs(ang_vel[2], 0.08) * 5.0  # 偏航惩罚

        # === 2) 稳定性奖励 (大幅强化 - DeepSeek策略) ===
        # 基于观察空间诊断：roll/pitch范围正常，四元数归一化良好
        roll_abs, pitch_abs = abs(roll), abs(pitch)
        
        # 姿态分级奖励 (更精细的稳定性评估)
        if roll_abs < 0.05 and pitch_abs < 0.05:           # 优秀稳定 
            stability_base = 3.0
        elif roll_abs < 0.08 and pitch_abs < 0.08:         # 良好稳定
            stability_base = 2.0  
        elif roll_abs < 0.12 and pitch_abs < 0.12:         # 可接受
            stability_base = 1.0
        else:                                               # 不稳定
            stability_base = -1.0
        
        # 二次惩罚大角度偏移
        stability_quadratic = -4.0 * (roll_abs**2 + pitch_abs**2)
        stability_reward = stability_base + stability_quadratic

        # === 3) 高度稳定 (基于观察空间诊断结果) ===
        # 诊断显示：pos_z范围[0.168,0.433]，均值0.327
        height = pos[2]
        # 调整到诊断验证的实际范围
        height_reward = band_reward(height, 0.28, 0.36, full=2.0, mid=1.0)
        
        # 危险高度强惩罚
        if height < 0.18:  # 基于诊断最低观测值调整
            height_penalty = -15.0 * (0.18 - height)
        else:
            height_penalty = 0.0

        # === 4) 膝关节奖励 (修复索引错误) ===
        # 修正：joint_indices只有12个元素，索引0-11
        # 根据Vision60结构：每条腿3个关节 [hip, thigh, calf]
        knee_reward = 0.0
        knee_indices_in_joint_list = [2, 5, 8, 11]  # 修正为正确的索引范围
        
        for knee_idx in knee_indices_in_joint_list:
            if knee_idx < len(self.joint_indices):  # 安全检查
                joint_state = p.getJointState(self.robot_id, self.joint_indices[knee_idx])
                knee_angle = joint_state[0]
                
                # 鼓励适度屈曲的自然站立姿态
                if -1.8 <= knee_angle <= -1.2:          # 理想屈曲范围
                    knee_reward += 0.3
                elif -2.2 <= knee_angle <= -0.8:        # 可接受范围  
                    knee_reward += 0.15
                else:                                    # 过度屈曲/伸展惩罚
                    knee_reward -= 0.1 * abs(knee_angle + 1.5)

        # === 5) 关节速度平滑性 (基于观察空间诊断) ===
        # 诊断显示：关节速度可能很大，但静止时应趋于零
        joint_velocities = []
        for joint_id in self.joint_indices:
            joint_vel = p.getJointState(self.robot_id, joint_id)[1] 
            joint_velocities.append(joint_vel)
        
        # 鼓励低关节速度 (静止状态)
        smoothness_reward = -0.05 * np.mean(np.abs(joint_velocities))

        # === 6) 存活奖励 ===
        survival_reward = 0.4

        # === 组合权重 (Phase 1: 稳定性优先) ===
        # 严格按照DeepSeek建议：stability_penalty权重2.0-3.0，forward_reward权重0.05-0.1
        w_velocity = 0.05      # 大幅降低前进权重 (DeepSeek建议)
        w_stability = 3.0      # 大幅提升稳定性权重 (DeepSeek建议)
        w_height = 0.15        # 高度控制权重  
        w_knee = 0.08          # 膝关节控制权重
        w_smooth = 0.02        # 平滑性权重

        # 分项计算
        velocity_term = w_velocity * (vx_penalty + vy_penalty + yaw_rate_penalty)
        stability_term = w_stability * stability_reward  
        height_term = w_height * (height_reward + height_penalty)
        knee_term = w_knee * knee_reward
        smooth_term = w_smooth * smoothness_reward

        # 最终奖励
        total_reward = (velocity_term + stability_term + height_term + 
                    knee_term + smooth_term + survival_reward)

        # 调试信息字典
        reward_info = {
            'total': total_reward,
            'velocity_penalty': vx_penalty + vy_penalty + yaw_rate_penalty,
            'stability': stability_reward,
            'height': height_reward + height_penalty, 
            'knee': knee_reward,
            'smoothness': smoothness_reward,
            'survival': survival_reward,
            # 状态监控
            'vel_x': vel[0], 'vel_y': vel[1], 'yaw_rate': ang_vel[2],
            'roll': roll_abs, 'pitch': pitch_abs, 'height_val': height,
            # 权重监控 (便于Phase 2调整)
            'w_vel': w_velocity, 'w_stab': w_stability, 'w_height': w_height
        }

        return total_reward, reward_info
    
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
                    positionGain=0.5,  # P控制增益
                    velocityGain=0.05  # D控制增益
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
