import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import numpy as np

class MyEnv(gym.Env[np.ndarray, int]): #ObsType is array of 36 integers, ActType is index of action from 0 to 36
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
    
    def _action_to_coordinates(action: int) -> tuple:
        row = action // 6 #int division for row
        column = action % 6 #modulo for column
        return row, column
    
    #Sliding window of 3 to check three in a axis
    def _check_three_in_a_axis(self, positions: list) -> tuple[bool,Optional[list[int]]]:
        for i in range(len(positions)-2):
            if self.board[positions[i]] == self.board[positions[i+1]] == self.board[positions[i+2]] == self.current_player: #Check if the three positions are the same and the current players mark
                return True, list(positions[i:i+3])
        return False, None

    def _check_win(self, action: int) -> tuple[bool,Optional[list[int]]]:
        row, column = self._action_to_coordinates(action)

        #horizontal
        if column == 0:
            win, win_positions = self._check_three_in_a_axis([action, action+1, action+2])
        elif column == 1:
            win, win_positions = self._check_three_in_a_axis([action-1, action, action+1, action+2])
        elif column == 4:
            win, win_positions = self._check_three_in_a_axis([action-2, action-1, action, action+1])
        elif column == 5:
            win, win_positions = self._check_three_in_a_axis([action-2, action-1, action])
        else:
            win, win_positions = self._check_three_in_a_axis([action-2, action-1, action, action+1, action+2])

        if win:
            return True, win_positions

        #vertical
        if row == 0:
            win, win_positions = self._check_three_in_a_axis([action, action+6, action+12])
        elif row == 1:
            win, win_positions = self._check_three_in_a_axis([action-6, action, action+6, action+12])
        elif row == 4:
            win, win_positions = self._check_three_in_a_axis([action-12, action-6, action, action+6])
        elif row == 5:
            win, win_positions = self._check_three_in_a_axis([action-12, action-6, action])
        else:
            win, win_positions = self._check_three_in_a_axis([action-12, action-6, action, action+6, action+12])

        if win:
            return True, win_positions

        steps_left1 = min(row, column, 2)
        steps_right1 = min(5-row, 5-column, 2)
        win, win_positions = self._check_three_in_a_axis([action + i*7 for i in range(-steps_left1, steps_right1+1)])

        if win:
            return True, win_positions

        steps_left2 = min(row, 5-column, 2)
        steps_right2 = min(5-row, column, 2)
        win, win_positions = self._check_three_in_a_axis([action + i*5 for i in range(-steps_left2, steps_right2+1)])

        if win:
            return True, win_positions

        return False, None


    def step(self, action: spaces.ActType):
        # Perform players action and update board
        self.board[action] = self.current_player

        # Determine if the current player has won with this action
        win, win_positions = self._check_win(action)
        reward = 1 if win else None

        # Check for Ddraw
        if len(self.valid_actions()) == 0 and not win:
            reward = 0

        

        row, column = self._action_to_coordinates(action)