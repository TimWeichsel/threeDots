import gymnasium as gym
import numpy as np
from collections import deque #buffer for action history

class DQNAgent:
    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float,epsilon_decay: float,final_epsilon: float):
        self.env = env
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.buffer = deque(maxlen=10000)
        self.last_observation = None #geet from update only the current observation

    def act(self, observation: np.ndarray, info: dict) -> int:
        self.last_observation = observation #safe observation as buffer needs observation and last obersvation
        valid_actions = info["valid_actions"]
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
            return action
        else:
            pass


    
    def update(self, next_observation: np.ndarray, action: int, reward: int, terminated: bool, info: dict):



        self.buffer.append((self.last_observation, action, reward, next_observation, terminated))
        pass

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
