import gymnasium as gym
import numpy as np

class RandomAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, observation: np.ndarray, info: dict) -> int:
        valid_actions = info["valid_actions"]
        if len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        else:
            raise ValueError("No valid actions available")
    
    def update(self, observation: np.ndarray, action: int, reward: int, terminated: bool, info: dict):
        pass #no learning for this agent)