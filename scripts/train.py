from threedots.env import MyEnv
from threedots.agents.randomAgent import RandomAgent
from threedots.agents.DQNAgent import DQNAgent
import gymnasium as gym 
import numpy as np
import argparse
import torch
import os

#Archtitecture: https://www.youtube.com/watch?v=s7KQoE7ZqEg

#trained 120000 on 5 obstacles and started the game
#trained 10000 on 7 obstacles and played the game as agent2
#trained 20000 on 6 obstacles and played the game as agent2
#trained 200000 on 5 obstacles and played the game as agent2
#trained 20000 on 6 obstacles and started the game

def switch_player_perspective(observation):
    switched_obs = observation.copy()
    switched_obs[observation == -1] = 1
    switched_obs[observation == 1] = -1
    return switched_obs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--obstacle_num", type=int, default=5)
    parser.add_argument("--agent_player", type=int, default=1, choices=[1,2])
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.001)
    parser.add_argument("--final_epsilon", type=float, default=0.05)
    parser.add_argument("--self_play", action="store_true")
    args = parser.parse_args()
    episodes = args.episodes
    obstacle_num = args.obstacle_num

    #initialize environment
    env = MyEnv(obstacle_num=obstacle_num)

    #Determine which player the agent is
    match args.agent_player:
        case 1:
            player1 = DQNAgent(env, learning_rate=args.learning_rate, initial_epsilon=args.initial_epsilon, 
                            epsilon_decay=args.epsilon_decay, final_epsilon=args.final_epsilon) #DQN Agent is player 1
            if os.path.exists("dqn_agent1.pth"): #continue training the old agent
                saved_agent = torch.load("dqn_agent1.pth")
                player1.q_net.load_state_dict(saved_agent["q_net"]) #Load q_net
                player1.target_net.load_state_dict(saved_agent["q_net"]) #Update target net
                player1.epsilon = saved_agent["epsilon"]
            if args.self_play:
                player2 = DQNAgent(env, learning_rate=args.learning_rate, initial_epsilon=args.initial_epsilon, 
                            epsilon_decay=args.epsilon_decay, final_epsilon=args.final_epsilon) 
                player2.q_net.load_state_dict(saved_agent["q_net"]) #Load q_net
                player2.target_net.load_state_dict(saved_agent["q_net"]) #Update target net
                player2.epsilon = 0.0 #No epsilon for frozen opponent
            else:
                player2 = RandomAgent(env) #Random Agent is player 2
        case 2:
            player2 = DQNAgent(env, learning_rate=args.learning_rate, initial_epsilon=args.initial_epsilon, 
                            epsilon_decay=args.epsilon_decay, final_epsilon=args.final_epsilon) #DQN Agent is player 2
            if os.path.exists("dqn_agent1.pth"): #continue training the old agent
                saved_agent = torch.load("dqn_agent1.pth")
                player2.q_net.load_state_dict(saved_agent["q_net"]) #Load q_net
                player2.target_net.load_state_dict(saved_agent["q_net"]) #Update target net
                player2.epsilon = saved_agent["epsilon"]
            if args.self_play:
                player1 = DQNAgent(env, learning_rate=args.learning_rate, initial_epsilon=args.initial_epsilon, 
                            epsilon_decay=args.epsilon_decay, final_epsilon=args.final_epsilon) 
                player1.q_net.load_state_dict(saved_agent["q_net"]) #Load q_net
                player1.target_net.load_state_dict(saved_agent["q_net"]) #Update target net
                player1.epsilon = 0.0 #No epsilon for frozen opponent
            else:
                player1 = RandomAgent(env) #Random Agent is player 1
        case _:
            raise ValueError("Agent can only be player 1 or 2")
    dqn_agent = player1 if args.agent_player == 1 else player2   


    terminated = False


    # Start Episode Training
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
                current_agent_indicator = info["current_player"]
                if current_agent_indicator == 1:
                    current_agent = player1
                elif current_agent_indicator == -1:
                    current_agent = player2
                else:
                    raise ValueError("Invalid player ID!")
                
                #determine agent observation (with considering which player it is)
                obs_for_agent = switch_player_perspective(observation) if current_agent_indicator == -1 else observation
                
                #Update Agent and handle self play
                if prev_reward[current_agent_indicator] is not None: #No update for first step since no action has been performed yet
                    opponent_reward = prev_reward[-current_agent_indicator] or 0
                    if current_agent is not dqn_agent and args.self_play: #No updated when opponent in selfplay
                        pass #skipt the frozen opponent
                    else: #all other cases an update is needed
                        current_agent.update(obs_for_agent, prev_action[current_agent_indicator], prev_reward[current_agent_indicator] - opponent_reward, terminated, prev_info[current_agent_indicator])

                #Get Action
                current_action = current_agent.act(obs_for_agent, info)
                prev_action[current_agent_indicator] = current_action
                print(f"- Agent{current_agent_indicator} to move: {current_action}")

                #Perform Action and get reward + observation
                observation, reward, terminated, truncated, info = env.step(current_action)
                prev_reward[current_agent_indicator] = reward
                prev_info[current_agent_indicator] = info
                
                print(f"- Reward: {reward}")
                print(f"- Terminated: {terminated}")
                
                score = info["current_score"]
                print(f"score: {score}")
                env.render()
                print()
                prev_agent = current_agent_indicator

        print("")

        #Determine winner and rewards (env already gives winner/looser reward to last player playing the game!)
        ## determine winner of the game
        score = info["current_score"]
        winner = 1 if score["1"] > score["-1"] else -1 if score["-1"] > score["1"] else 0

        ## winner/looser reward for second last player playing the game
        if prev_agent == 1:
            win_reward_p1 = prev_reward[1] #prev_reward already contains the reward for winning/loosing the game for player 1
            win_reward_p2 = prev_reward[-1] + (100 if winner == -1 else -100 if winner == 1 else 0)
        elif prev_agent == -1:
            win_reward_p2 = prev_reward[-1] #prev_reward already contains the reward for winning/loosing the game for player 2
            win_reward_p1 = prev_reward[1] + (100 if winner == 1 else -100 if winner == -1 else 0)

        # Load the final obervation for both agents
        obs_for_player_1 = observation #gets the non flipped version
        obs_for_player_2 = switch_player_perspective(observation) #board needs to be flipped

        # Update the players and check for self play (since no update is needed)
        if args.self_play:
            if args.agent_player == 1: #Only update player 1
                player1.update(obs_for_player_1, prev_action[1], win_reward_p1, terminated, prev_info[1])
            elif args.agent_player == 2: #Only update player 2
                player2.update(obs_for_player_2, prev_action[-1], win_reward_p2, terminated, prev_info[-1])
        
        else: #Update both agents
            player1.update(obs_for_player_1, prev_action[1], win_reward_p1, terminated, prev_info[1])
            player2.update(obs_for_player_2, prev_action[-1], win_reward_p2, terminated, prev_info[-1])

        print(f"Final score: {score}, Winner: {winner}")

        dqn_agent.decay_epsilon()
        
    torch.save({"q_net":dqn_agent.q_net.state_dict(), "epsilon": dqn_agent.epsilon},"dqn_agent1.pth")

if __name__ == "__main__":
    main()