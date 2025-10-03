#!/usr/bin/env python3
"""
四足机器人PPO训练指标监控系统 - 性能优化版
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque, defaultdict
import gc  # 添加垃圾回收


class QuadrupedMetricsCallback(BaseCallback):
    """四足机器人专用指标回调 - 性能优化版本"""
    
    def __init__(self, eval_freq=10000, verbose=0):  # 默认降低频率
        super().__init__(verbose)
        self.eval_freq = eval_freq
        
        # 减小窗口大小以节省内存
        self.window_size = 50  # 从100减少到50
        self.episode_metrics = defaultdict(lambda: deque(maxlen=self.window_size))
        
        # 当前episode数据收集
        self.current_episode = {
            'velocities': [],
            'heights': [],
            'orientations': [],
            'rewards': []
        }
        
        # 性能计数器
        self.episodes_processed = 0
        
    def _on_step(self) -> bool:
        # 获取环境信息
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]  # 只处理第一个环境，减少计算量
            reward = self.locals.get('rewards', [0])[0]
            
            # 只收集关键数据，减少内存使用
            if 'velocity' in info:
                # 只保留前两个维度 (vx, vy)
                self.current_episode['velocities'].append(info['velocity'][:2])
            if 'height' in info:
                self.current_episode['heights'].append(info['height'])
            if 'orientation' in info:
                # 只保留 roll, pitch
                self.current_episode['orientations'].append(info['orientation'][:2])
            
            self.current_episode['rewards'].append(reward)
            
            # 检查episode结束
            if self.locals.get('dones', [False])[0]:
                self._process_episode_end()
                
        # 定期记录指标 (频率已降低)
        if self.n_calls % self.eval_freq == 0:
            self._log_metrics()
            # 定期清理内存
            if self.episodes_processed % 100 == 0:
                gc.collect()
            
        return True
    
    def _process_episode_end(self):
        """处理episode结束时的指标计算 - 优化版"""
        if not self.current_episode['velocities']:
            self._reset_episode()
            return
            
        # 转换为numpy数组进行批量计算
        velocities = np.array(self.current_episode['velocities'])  # (n, 2)
        heights = np.array(self.current_episode['heights'])
        orientations = np.array(self.current_episode['orientations'])  # (n, 2)
        
        # 新增：跌倒统计 & yaw rate统计 & 安全访问
        if 'falls' in self.current_episode and self.current_episode['falls']:
            falls = sum(self.current_episode['falls'])
            self.episode_metrics['fall_count'].append(falls)

        if 'angular_velocities' in self.current_episode and self.current_episode['angular_velocities']:
            ang_velocities = np.array(self.current_episode['angular_velocities'])
            yaw_rates = ang_velocities[:, 2]
            self.episode_metrics['yaw_rate_rms'].append(float(np.sqrt(np.mean(yaw_rates**2))))
        
        # 批量计算指标
        vx, vy = velocities[:, 0], velocities[:, 1]
        roll, pitch = orientations[:, 0], orientations[:, 1]
        

        # 更新指标 (使用更高效的计算)
        self.episode_metrics['vx_mean'].append(float(np.mean(vx)))
        self.episode_metrics['vy_abs_mean'].append(float(np.mean(np.abs(vy))))
        self.episode_metrics['height_mean'].append(float(np.mean(heights)))

         # 高度误差RMS - 基于vision60实际站立高度
        target_height = 0.21  # vision60站立目标高度 (来自测试结果)
        height_error_rms = np.sqrt(np.mean((heights - target_height)**2))
        self.episode_metrics['height_error_rms'].append(float(height_error_rms))

        self.episode_metrics['roll_rms'].append(float(np.sqrt(np.mean(roll**2))))
        self.episode_metrics['pitch_rms'].append(float(np.sqrt(np.mean(pitch**2))))
        
        # Episode基本信息
        episode_reward = sum(self.current_episode['rewards'])
        episode_length = len(self.current_episode['rewards'])
        
        self.episode_metrics['episode_reward'].append(episode_reward)
        self.episode_metrics['episode_length'].append(episode_length)
        
        self.episodes_processed += 1
        self._reset_episode()
    
    def _reset_episode(self):
        """重置episode数据"""
        self.current_episode = {
            'velocities': [],
            'heights': [], 
            'orientations': [],
            'angular_velocities': [],  # 添加这行
            'falls': [],               # 添加这行
            'rewards': []
        }
    
    def _log_metrics(self):
        """记录指标到tensorboard - 优化版"""
        if not self.episode_metrics:
            return
            
        # 只记录关键指标，减少I/O
        key_metrics = ['vx_mean', 'roll_rms', 'pitch_rms', 'height_error_rms', 'episode_reward']
        
        for metric_name in key_metrics:
            values = self.episode_metrics.get(metric_name, [])
            if values:
                mean_val = float(np.mean(values))
                self.logger.record(f"quadruped/{metric_name}", mean_val)
        
        # 计算稳定性得分
        vx_values = self.episode_metrics.get('vx_mean', [])
        roll_values = self.episode_metrics.get('roll_rms', [])
        
        if vx_values and roll_values:
            vx_mean = np.mean(vx_values)
            roll_rms = np.mean(roll_values)
            stability_score = vx_mean / (1 + roll_rms)
            self.logger.record("quadruped/stability_score", float(stability_score))
        
        if self.verbose:
            print(f"[Step {self.n_calls}] 已处理 {self.episodes_processed} 个episodes")