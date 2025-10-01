#!/usr/bin/env python3
"""
录制训练好的机器人视频 - 支持新的runs目录结构
"""
import os
import sys
# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import imageio
from stable_baselines3 import PPO
from quadruped_env import QuadrupedEnv
import pybullet as p

import glob

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

            print(f"💾 保存视频到: {video_path}")
            try:
                imageio.mimsave(video_path, self.images, fps=30)
                print(f"✅ 视频已保存，共 {len(self.images)} 帧")
            except Exception as e:
                print(f"❌ 视频保存失败: {e}")

    def close(self):
        self.base_env.close()

def find_latest_model():
    """查找最新的训练模型"""
    # 查找runs目录中的模型
    model_patterns = [
        "runs/*/ppo_metrics_*_model.zip",
        "runs/*/*.zip",
        "ppo_quadruped_light.zip",  # 兼容旧版本
        "ppo_quadruped_light"       # 兼容旧版本
    ]
    
    for pattern in model_patterns:
        models = glob.glob(pattern)
        if models:
            # 按修改时间排序，返回最新的
            latest_model = max(models, key=os.path.getmtime)
            return latest_model
    
    return None

def record_trained_agent(model_path=None, output_dir="videos"):
    """录制训练好的智能体"""
    print("=== 录制训练好的机器人视频 ===")

    # 查找模型
    if model_path is None:
        model_path = find_latest_model()
    
    if model_path is None:
        print("❌ 未找到训练好的模型!")
        print("可用选项:")
        print("1. 先运行训练: python train_ppo_with_metrics.py")
        print("2. 或指定模型路径: python scripts/record_video.py <model_path>")
        return
    
    print(f"📦 加载模型: {model_path}")
    
    # 加载模型
    try:
        model = PPO.load(model_path)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 创建视频输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建环境
    base_env = QuadrupedEnv(render_mode=None)
    env = VideoRecordingEnv(base_env, video_folder=f"./{output_dir}/")

    print("🎬 开始录制...")

    # 录制几个episode
    for episode in range(3):
        print(f"\n📹 录制第 {episode+1} 个episode...")

        obs, _ = env.reset()
        total_reward = 0

        for step in range(200):  # 增加录制长度
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            if step % 50 == 0:
                print(f"  Step {step}: Reward = {reward:.3f}")

            if terminated or truncated:
                break

        print(f"  ✅ Episode结束: 总奖励 = {total_reward:.3f}")

    env.close()
    print(f"\n🎉 录制完成! 视频保存在 ./{output_dir}/ 文件夹中")

def main():
    """主函数 - 支持命令行参数"""
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"使用指定模型: {model_path}")
    else:
        model_path = None
        print("自动查找最新模型...")
    
    record_trained_agent(model_path)

if __name__ == "__main__":
    main()