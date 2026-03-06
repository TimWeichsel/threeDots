from threedots.env import MyEnv

env = MyEnv(obstacle_num=4)
obs, info = env.reset(seed=42)

print("=== Initial Board ===")
env.render()
print(f"\nCurrent Player: {info['current_player']}")
print(f"Valid Actions: {len(info['valid_actions'])} available")
print(f"Scores: {info['current_score']}")

print("\n=== Playing 3 moves ===")
obs, reward, terminated, truncated, info = env.step(0)
print(f"Player 1 played 0 → reward={reward}")
env.render()

obs, reward, terminated, truncated, info = env.step(1)
print(f"\nPlayer -1 played 1 → reward={reward}")
env.render()

obs, reward, terminated, truncated, info = env.step(6)
print(f"\nPlayer 1 played 6 → reward={reward}")
env.render()

print("\n=== Testing horizontal 3-in-a-row ===")
env2 = MyEnv(obstacle_num=0)
env2.reset(seed=1)
env2.step(0)   # Player 1
env2.step(6)   # Player -1
env2.step(1)   # Player 1
env2.step(7)   # Player -1
obs, reward, terminated, truncated, info = env2.step(2)  # Player 1 → should score!
print(f"Player 1 played 0,1,2 → reward={reward} (expected: 1)")
env2.render()
print(f"Scores: {info['current_score']}")

print("\n=== Testing invalid action ===")
env3 = MyEnv(obstacle_num=0)
env3.reset(seed=1)
env3.step(0)  # Player 1 plays field 0
obs, reward, terminated, truncated, info = env3.step(0)  # Player -1 tries same field
print(f"Invalid action reward={reward} (expected: -1000), terminated={terminated} (expected: True)")
