from threedots.env import MyEnv
from threedots.agents.randomAgent import RandomAgent
import gymnasium as gym 
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--obstacle_num", type=int, default=5)
args = parser.parse_args()
episodes = args.episodes
obstacle_num = args.obstacle_num

env = MyEnv(obstacle_num=obstacle_num)
agent1 = RandomAgent(env)
agent2 = RandomAgent(env) #agent -1 is agent2 
terminated = False

for episode in range(episodes):
    terminated = False
    prev_reward = {-1: None, 1: None}
    prev_action = {-1: None, 1: None}
    prev_info = {-1: None, 1: None}
    prev_agent = None
    print(f"!!!!EPISODE {episode+1} of {episodes}!!!!")
    observation, info = env.reset()
    print(f"Start Playfield:")
    env.render()
    print()
    for step in range(36-obstacle_num):
        print()
        print(f"Step {step}:")
        if not terminated:
            
            current_agent_info = info["current_player"]
            if current_agent_info == 1:
                current_agent = agent1
            elif current_agent_info == -1:
                current_agent = agent2
            else:
                raise ValueError("Invalid player ID!")
            
            #Update Agent
            if prev_reward[current_agent_info] is not None:
                current_agent.update(observation, prev_action[current_agent_info], prev_reward[current_agent_info], terminated, prev_info[current_agent_info])

            #Get Action
            current_action = current_agent.act(observation, info)
            prev_action[current_agent_info] = current_action
            print(f"- Agent{current_agent_info} to move: {current_action}")

            #Perform Action and get reward + observation
            observation, reward, terminated, truncated, info = env.step(current_action)
            prev_reward[current_agent_info] = reward
            prev_info[current_agent_info] = info
            
            print(f"- Reward: {reward}")
            print(f"- Termination: {terminated}")
            
            score = info["current_score"]
            print(f"score: {score}")
            env.render()
            print()
            prev_agent = current_agent_info

        else: #Game Ends

            score = info["current_score"]
            winner = 1 if score["1"] > score["-1"] else -1 if score["-1"] > score["1"] else 0

            if prev_agent == 1 and winner == 1:
                agent1.update(observation, prev_action[1], prev_reward[1], terminated, prev_info[1])
                agent2.update(observation, prev_action[-1], prev_reward[-1]-100, terminated, prev_info[-1]) #agent2 looses
            if prev_agent == 1 and winner == -1:
                agent1.update(observation, prev_action[1], prev_reward[1], terminated, prev_info[1])
                agent2.update(observation, prev_action[-1], prev_reward[-1]+100, terminated, prev_info[-1]) #agent2 wins
            if prev_agent == -1 and winner == 1:
                agent1.update(observation, prev_action[1], prev_reward[1]+100, terminated, prev_info[1]) #agent1 wins
                agent2.update(observation, prev_action[-1], prev_reward[-1], terminated, prev_info[-1])
            if prev_agent == -1 and winner == -1:
                agent1.update(observation, prev_action[1], prev_reward[1]-100, terminated, prev_info[1]) #agent1 looses
                agent2.update(observation, prev_action[-1], prev_reward[-1], terminated, prev_info[-1])

            if winner == 0:  # Draw
                agent1.update(observation, prev_action[1], prev_reward[1], terminated, prev_info[1])
                agent2.update(observation, prev_action[-1], prev_reward[-1], terminated, prev_info[-1])
            
            print(f"Final score: {score}")
            break
                
    