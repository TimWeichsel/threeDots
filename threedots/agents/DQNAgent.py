import gymnasium as gym
import numpy as np

class DQNAgent:
    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float,epsilon_decay: float,final_epsilon: float):
        self.env = env
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def act(self, observation: np.ndarray, info: dict) -> int:
        valid_actions = info["valid_actions"]
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            pass


    
    def update(self, observation: np.ndarray, action: int, reward: int, terminated: bool, info: dict):
        pass

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
