import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quadruped_env import QuadrupedEnv
import numpy as np
import time

env = QuadrupedEnv(render_mode=None)
obs, _ = env.reset()

print('🔍 测试静态站立稳定性...')
for i in range(500):  # 约5秒
    obs, reward, terminated, truncated, info = env.step(np.zeros(12))
    if terminated:
        print(f'❌ 机器人在{i*env.time_step:.2f}秒时倒下')
        break
    if i % 100 == 0:
        print(f"  {i*env.time_step:.1f}s: 高度={info['height']:.3f}m")

if not terminated:
    print('✅ 静态站立稳定，可以继续优化')
else:
    print('❌ 默认姿态不稳定，需要先修复基础问题')

env.close()