from threedots.env import MyEnv
from threedots.agents.randomAgent import RandomAgent
from threedots.agents.DQNAgent import DQNAgent
def switch_player_perspective(observation):
    switched_obs = observation.copy()
    switched_obs[observation == -1] = 1
    switched_obs[observation == 1] = -1
    return switched_obs
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=10000)
parser.add_argument("--obstacle_num", type=int, default=5)
parser.add_argument("--agent_player", type=int, default=1, choices=[1, 2])
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args() 

env = MyEnv(obstacle_num=args.obstacle_num)
dqn = DQNAgent(env, learning_rate=0.001, initial_epsilon=0.0,
               epsilon_decay=0.0, final_epsilon=0.0)

saved = torch.load("dqn_agent1.pth")
dqn.q_net.load_state_dict(saved["q_net"])

if args.agent_player == 1:
    agent1, agent2 = dqn, RandomAgent(env)
else:
    agent1, agent2 = RandomAgent(env), dqn

wins, losses, draws = 0, 0, 0

for episode in range(args.episodes):
    terminated = False
    observation, info = env.reset()
    if args.verbose:
        print(f"=== Episode {episode+1} ===")
        env.render()
        print()

    for step in range(36 - args.obstacle_num):
        if not terminated:
            current_agent_info = info["current_player"]
            current_agent = agent1 if current_agent_info == 1 else agent2
            obs_for_agent = switch_player_perspective(observation) if current_agent_info == -1 else observation
            current_action = current_agent.act(obs_for_agent, info)
            observation, reward, terminated, truncated, info = env.step(current_action)
            if args.verbose:
                print(f"Step {step} — Agent{current_agent_info} played {current_action}, reward={reward}")
                env.render()
                print()

    score = info["current_score"]
    winner = 1 if score["1"] > score["-1"] else -1 if score["-1"] > score["1"] else 0
    if args.agent_player == 1:
        if winner == 1: wins += 1
        elif winner == -1: losses += 1
        else: draws += 1
    else:
        if winner == -1: wins += 1
        elif winner == 1: losses += 1
        else: draws += 1
    if args.verbose:
        print(f"Final score: {score}, Winner: {winner}\n")

print(f"Results over {args.episodes} episodes (DQN as player {args.agent_player}):")
print(f"Wins:   {wins} ({wins/args.episodes*100:.1f}%)")
print(f"Losses: {losses} ({losses/args.episodes*100:.1f}%)")
print(f"Draws:  {draws} ({draws/args.episodes*100:.1f}%)")
