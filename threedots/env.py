import gymnasium as gym
import numpy as np

class MyEnv(gym.Env):
    # Define playfield of 6x6 with 36 possible actions
    def __init__(self, obstacle_num=4):
        self.action_space = gym.spaces.Discrete(36)
        self.observation_space = gym.spaces.Box(low=-1, high=2, shape=(36,), dtype=np.int8)
        self.obstacle_default = obstacle_num
        self.board = None
        self.current_player = None

    def valid_actions(self) -> list:
        return np.where(self.board == 0)[0].tolist()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) #Reset the environment and set the seed for reproducibility

        self.board = np.zeros(36, dtype=np.int8)
        obstacle_num = options.get("obstacle_num", self.obstacle_default) if options else self.obstacle_default #Takes obstacle_num if options and "obstacle_num" in options exist else it takes self.obstacle_default
        obstacle_indices = self.np_random.choice(36, size=obstacle_num, replace=False) #Randomly select obstacle_num obstacle position, with replace=False to not have duplicates
        self.board[obstacle_indices]=2
       
        self.current_player=1 #starting with player 1 and not -1

        observation = self.board.copy()
        info = {"current_player": self.current_player, "valid_actions": self.valid_actions()} 
        return observation, info