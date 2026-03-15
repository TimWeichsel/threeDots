from threedots.env import MyEnv
from threedots.agents.randomAgent import RandomAgent
from threedots.agents.DQNAgent import DQNAgent
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--obstacle_num", type=int, default=5)
args = parser.parse_args()

env = MyEnv(obstacle_num=args.obstacle_num)
agent1 = DQNAgent(env, learning_rate=0.001, initial_epsilon=0.0,
                  epsilon_decay=0.0, final_epsilon=0.0)

saved = torch.load("dqn_agent1.pth")
agent1.q_net.load_state_dict(saved["q_net"])

agent2 = RandomAgent(env)

wins, losses, draws = 0, 0, 0

for episode in range(args.episodes):
    terminated = False
    prev_action = {-1: None, 1: None}
    prev_info = {-1: None, 1: None}
    prev_agent = None
    observation, info = env.reset()

    for step in range(36 - args.obstacle_num):
        if not terminated:
            current_agent_info = info["current_player"]
            current_agent = agent1 if current_agent_info == 1 else agent2
            current_action = current_agent.act(observation, info)
            prev_action[current_agent_info] = current_action
            observation, reward, terminated, truncated, info = env.step(current_action)
            prev_info[current_agent_info] = info
            prev_agent = current_agent_info

    score = info["current_score"]
    winner = 1 if score["1"] > score["-1"] else -1 if score["-1"] > score["1"] else 0
    if winner == 1: wins += 1
    elif winner == -1: losses += 1
    else: draws += 1

print(f"Results over {args.episodes} episodes:")
print(f"Wins:   {wins} ({wins/args.episodes*100:.1f}%)")
print(f"Losses: {losses} ({losses/args.episodes*100:.1f}%)")
print(f"Draws:  {draws} ({draws/args.episodes*100:.1f}%)")