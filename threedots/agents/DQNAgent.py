import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque #buffer for action history
import random

class DQNAgent:
    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float, epsilon_decay: float, final_epsilon: float, buffer_maxlen:int = 10000, batch_size: int = 32, gamma: float = 0.99):
        self.env = env
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.buffer_maxlen = buffer_maxlen
        self.batch_size = batch_size
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

        #Target Net for max Q-Value in Bellman calculation (predicted q values)
        self.target_net = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_update_counter = 0


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
                valid_actions_np = np.array(valid_actions, dtype=np.int64) #get valid indizes as numpy array since pytorch doesn't support python list
                valid_actions_tensor = torch.tensor(valid_actions_np, dtype=torch.long)
                valid_q_values = q_values[valid_actions_tensor] #filter q values with valid action 
                best_q_index = valid_q_values.argmax().item() #get highest q value for a valid avtion as index (with item())
                return valid_actions[best_q_index] #return valid action with highest q value

                

    def _calculate_bellman_for_batches(self, rewards: torch.tensor, max_predicted_q_values: torch.tensor, terminationed_games_tensor: torch.tensor, gamma: float = None ) -> torch.tensor:
        if gamma is None:
            gamma = self.gamma
        return rewards + gamma * max_predicted_q_values * (1-terminationed_games_tensor) #only value future if game is not terminated
    
    def update(self, next_observation: np.ndarray, action: int, reward: int, terminated: bool, info: dict):
        self.buffer.append((self.last_observation, action, reward, next_observation, terminated))
        if len(self.buffer) < 1000:
            return #training with experience replay, when buffer has 1000 samples
        else:
            batches = random.sample(self.buffer, self.batch_size) #32 random samples of buffer (no deletion since buffer is deque with maxlen)
            last_observations_array_tuple, actions, rewards, next_observations_array_tuple, terminations = zip(*batches)

            #convert to types compatible with pytorch tensors
            last_observations = np.array(last_observations_array_tuple)
            next_observations = np.array(next_observations_array_tuple)

            #convert to tensors
            last_observation_tensor = torch.tensor(last_observations, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long) #torch long as for indizes
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            next_observation_tensor = torch.tensor(next_observations, dtype=torch.float32)
            terminationed_games_tensor = torch.tensor(terminations, dtype=torch.float32)

            #get q values
            q_values = self.q_net(last_observation_tensor) #q values of board before action was played for all batches
            q_values_played = q_values[range(self.batch_size), actions_tensor]  #extrat only the q value for each action played in each batch
            with torch.no_grad(): #No graph needed for predicted q values
                predicted_q_values = self.target_net(next_observation_tensor)
                max_predicted_q_values = predicted_q_values.max(dim=1).values #get max q value for each batch (dim=1 checks all q values in one batch)
            
            #Bellman calculation
            targets = self._calculate_bellman_for_batches(rewards_tensor,max_predicted_q_values, terminationed_games_tensor)
            
            #Loss calculation between targets and q_values plaayed
            loss_object = nn.MSELoss()
            loss = loss_object(q_values_played,targets)

            #Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.target_update_counter +=1
            if self.target_update_counter % 1000 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())




    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
