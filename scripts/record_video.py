#!/usr/bin/env python3
"""
录制训练好的机器人视频
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import imageio
from stable_baselines3 import PPO
from quadruped_env import QuadrupedEnv
import pybullet as p

class VideoRecordingEnv:
    """带视频录制功能的环境包装器"""
    
    def __init__(self, base_env, video_folder="./videos/", video_length=500):
        self.base_env = base_env
        self.video_folder = video_folder
        self.video_length = video_length
        self.current_step = 0
        self.images = []
        
        # 创建视频文件夹
        os.makedirs(video_folder, exist_ok=True)
        
    def reset(self):
        self.current_step = 0
        self.images = []
        obs = self.base_env.reset()
        
        # 拍摄第一帧
        self._capture_frame()
        return obs
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # 拍摄当前帧
        self._capture_frame()
        self.current_step += 1
        
        # 如果episode结束或达到视频长度，保存视频
        if terminated or truncated or self.current_step >= self.video_length:
            self._save_video()
            
        return obs, reward, terminated, truncated, info
        
    def _capture_frame(self):
        """捕获当前帧"""
        # 设置相机视角
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.5],
            distance=3.0,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=16/9,
            nearVal=0.1,
            farVal=100.0
        )
        
        # 渲染图像
        width, height = 640, 480
        _, _, rgb_array, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix
        )
        
        # 转换为RGB格式
        rgb_array = np.array(rgb_array).reshape(height, width, 4)
        rgb_array = rgb_array[:, :, :3]  # 去掉alpha通道
        
        self.images.append(rgb_array)
        
    def _save_video(self):
        """保存视频文件"""
        if len(self.images) > 0:
            video_path = os.path.join(
                self.video_folder, 
                f"quadruped_episode_{int(time.time())}.mp4"
            )
            
            print(f"保存视频到: {video_path}")
            imageio.mimsave(video_path, self.images, fps=30)
            print(f"视频已保存，共 {len(self.images)} 帧")
            
    def close(self):
        self.base_env.close()

def record_trained_agent():
    """录制训练好的智能体"""
    print("=== 录制训练好的机器人视频 ===")
    
    # 加载训练好的模型
    try:
        model = PPO.load("ppo_quadruped_light.zip")
        print("模型加载成功!")
    except:
        print("模型未找到，请先训练模型")
        return
    
    # 创建环境（无GUI但可以录制）
    base_env = QuadrupedEnv(render_mode=None)
    env = VideoRecordingEnv(base_env, video_folder="./videos/")
    
    print("开始录制...")
    
    # 录制几个episode
    for episode in range(3):
        print(f"\n录制第 {episode+1} 个episode...")
        
        # 设置不同随机种子
        import numpy as np
        np.random.seed(100 + episode * 10)

        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            # 使用训练好的策略
            if episode == 0:
                action, _ = model.predict(obs, deterministic=True)   # 最佳表现
            else:
                action, _ = model.predict(obs, deterministic=False)  # 不同变化
            obs, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            
            if step % 50 == 0:
                print(f"  Step {step}: Reward = {reward:.3f}")
            
            if terminated or truncated:
                env._save_video()  # 强制保存
                print(f"  Episode结束: 总奖励 = {total_reward:.3f}")
                break
        
    env.close()
    print("\n录制完成! 视频保存在 ./videos/ 文件夹中")

if __name__ == "__main__":
    record_trained_agent()
