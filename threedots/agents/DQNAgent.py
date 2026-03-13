import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque #buffer for action history
import random

class DQNAgent:
    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float,epsilon_decay: float,final_epsilon: float, buffer_maxlen:int = 10000):
        self.env = env
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.buffer_maxlen = buffer_maxlen
        self.buffer = deque(maxlen=buffer_maxlen)
        self.last_observation = None #get from update only the current observation

        #define the q network architecture
        self.q_net = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)


    def act(self, observation: np.ndarray, info: dict) -> int:
        self.last_observation = observation #safe observation as buffer needs observation and last obersvation
        valid_actions = info["valid_actions"]
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
            return action
        else:
            with torch.no_grad(): #No graph needed for action selection

                board_tensor = torch.tensor(observation, dtype=torch.float32)

                #forward pass with current board
                q_values = self.q_net(board_tensor)

                #get the highest q value for valid actions
                valid_actions_np = np.array(valid_actions) #get valid indizes as numpy array since pytorch doesn't support python list
                valid_q_values = q_values[valid_actions_np] #filter q values with valid action 
                best_q_index = valid_q_values.argmax().item() #get highest q value for a valid avtion as index (with item())
                return valid_actions[best_q_index] #return valid action with highest q value

                


    
    def update(self, next_observation: np.ndarray, action: int, reward: int, terminated: bool, info: dict):
        self.buffer.append((self.last_observation, action, reward, next_observation, terminated))
        if len(self.buffer) < 1000:
            return #training with experience replay, when buffer has 1000 samples
        else:
            batches = np.random.choice(self.buffer, size=32, replace= False) #32 random samples of buffer (no deletion since buffer is deque with maxlen)
            last_observations, actions, rewards, next_observations, terminations = zip(*batches)
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
