from quadruped_env import QuadrupedEnv

env = QuadrupedEnv()
obs, _ = env.reset()

print("=== Initial State ===")
print("Observation shape:", obs.shape)
print("Initial position:", obs[:3])
print("Initial orientation:", obs[3:6]) 
print("Initial velocity:", obs[6:9])

# Test one step
action = env.action_space.sample()
print("\n=== Execute Action ===")
print("Action:", action)

obs, reward, terminated, truncated, _ = env.step(action)

print("\n=== After One Step ===")
print("Position:", obs[:3])
print("Height (z):", obs[2])
print("Orientation:", obs[3:6])
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

# Check termination reasons
if obs[2] < 0.2:
    print("Termination reason: Height too low")
if abs(obs[3]) > 1.57 or abs(obs[4]) > 1.57:
    print("Termination reason: Over-rotated")

env.close()
