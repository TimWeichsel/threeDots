import gymnasium as gym
from typing import Optional
import numpy as np

class MyEnv(gym.Env[np.ndarray, int]): #ObsType is array of 36 integers, ActType is index of action from 0 to 36
    # Define playfield of 6x6 with 36 possible actions
    def __init__(self, render_mode="human", obstacle_num=4):
        self.action_space = gym.spaces.Discrete(36)
        self.observation_space = gym.spaces.Box(low=-1, high=2, shape=(36,), dtype=np.int8)
        self.obstacle_default = obstacle_num
        self.board = None
        self.current_player = None
        self.current_score = {"-1": 0, "1": 0}
        self.current_score_positions = {"-1":[], "1": []} 
        self.render_mode = render_mode
        # determine truncated as always False (no time limit or other limitation)
        self.truncated = False

    def valid_actions(self) -> list:
        return np.where(self.board == 0)[0].tolist()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) #Reset the environment and set the seed for reproducibility

        self.board = np.zeros(36, dtype=np.int8)
        obstacle_num = options.get("obstacle_num", self.obstacle_default) if options else self.obstacle_default #Takes obstacle_num if options and "obstacle_num" in options exist else it takes self.obstacle_default
        obstacle_indices = self.np_random.choice(36, size=obstacle_num, replace=False) #Randomly select obstacle_num obstacle position, with replace=False to not have duplicates
        self.board[obstacle_indices]=2
       
        self.current_player=1 #starting with player 1 and not -1
        self.current_score = {"-1": 0, "1": 0}
        self.current_score_positions = {"-1":[], "1": []} 

        observation = self.board.copy()
        return observation, self._current_info()
    
    def _action_to_coordinates(self, action: int) -> tuple:
        row = action // 6 #int division for row
        column = action % 6 #modulo for column
        return row, column
    
    #Sliding window of 3 to check three in a axis
    def _check_three_in_a_axis(self, positions: list) -> tuple[int,Optional[list[int]]]:
        newPoints = 0
        newPointPositions = []
        for i in range(len(positions)-2):
            if self.board[positions[i]] == self.board[positions[i+1]] == self.board[positions[i+2]] == self.current_player: #Check if the three positions are the same and the current players mark
                newPoints += 1
                newPointPositions.append(positions[i:i+3])
        return newPoints, newPointPositions

    def _check_new_points(self, action: int) -> tuple[int,Optional[list[int]]]:
        row, column = self._action_to_coordinates(action)
        point_positions_h, point_positions_v, point_positions_d1, point_positions_d2 = [], [], [], []

        #horizontal
        if column == 0:
            newPoints_h, point_positions_h = self._check_three_in_a_axis([action, action+1, action+2])
        elif column == 1:
            newPoints_h, point_positions_h = self._check_three_in_a_axis([action-1, action, action+1, action+2])
        elif column == 4:
            newPoints_h, point_positions_h = self._check_three_in_a_axis([action-2, action-1, action, action+1])
        elif column == 5:
            newPoints_h, point_positions_h = self._check_three_in_a_axis([action-2, action-1, action])
        else:
            newPoints_h, point_positions_h = self._check_three_in_a_axis([action-2, action-1, action, action+1, action+2])


        #vertical
        if row == 0:
            newPoints_v, point_positions_v = self._check_three_in_a_axis([action, action+6, action+12])
        elif row == 1:
            newPoints_v, point_positions_v = self._check_three_in_a_axis([action-6, action, action+6, action+12])
        elif row == 4:
            newPoints_v, point_positions_v = self._check_three_in_a_axis([action-12, action-6, action, action+6])
        elif row == 5:
            newPoints_v, point_positions_v = self._check_three_in_a_axis([action-12, action-6, action])
        else:
            newPoints_v, point_positions_v = self._check_three_in_a_axis([action-12, action-6, action, action+6, action+12])

        steps_left1 = min(row, column, 2)
        steps_right1 = min(5-row, 5-column, 2)
        newPoints_d1, point_positions_d1 = self._check_three_in_a_axis([action + i*7 for i in range(-steps_left1, steps_right1+1)])


        steps_left2 = min(row, 5-column, 2)
        steps_right2 = min(5-row, column, 2)
        newPoints_d2, point_positions_d2 = self._check_three_in_a_axis([action + i*5 for i in range(-steps_left2, steps_right2+1)])

        return newPoints_h+newPoints_v+newPoints_d1+newPoints_d2, point_positions_h+point_positions_v+point_positions_d1+point_positions_d2
    
    def _update_scores(self, newPoints: int, point_positions: list[int]):
        self.current_score[str(self.current_player)] += newPoints
        self.current_score_positions[str(self.current_player)].extend(point_positions)

    def _determine_winner(self) -> int:
        if self.current_score["1"] > self.current_score["-1"]:
            return 1
        elif self.current_score["1"] < self.current_score["-1"]:
            return -1
        else:
            return 0 #Tie
        
    def _current_info(self) -> dict:
        return {"current_player": self.current_player, "valid_actions": self.valid_actions(), "current_score": self.current_score.copy(), "current_score_positions": {"-1": self.current_score_positions["-1"].copy(), "1": self.current_score_positions["1"].copy()}}



    def step(self, action: int):
        #check if action is valid
        if action not in self.valid_actions():
            observation = self.board.copy()
            info = self._current_info()
            reward = -1000
            terminated = True

        else:
            # Perform players action and update board
            self.board[action] = self.current_player

            # Determine if the current player scored points from the current action
            newPoints, point_positions = self._check_new_points(action)
            self._update_scores(newPoints, point_positions)

            reward = newPoints

            # Check for End if game and determine winner/draw
            terminated = len(self.valid_actions()) == 0
            if terminated:
                winner = self._determine_winner()
                if winner == self.current_player:
                    reward += 100
                elif winner == self.current_player * -1:
                    reward -= 100
            
                
            # Switch Player
            self.current_player *= -1
            
            # Obeservation is a copy of the current state of the board
            observation = self.board.copy()

            # Info contains the current player, the valid actions for the next player, the current scores and the positions of the points scored by each player
            info = self._current_info()

        return observation, reward, terminated, self.truncated, info

    def render (self):
        if self.render_mode == "human":
            for field_index in range(len(self.board)):
                if field_index % 6 == 0 and field_index != 0:
                    print()
                match self.board[field_index]:
                    case 1: print(" X ", end="")
                    case -1: print(" O ", end="")
                    case 0: print(" _ ", end="")
                    case 2: print(" # ", end="")
        else:
            raise NotImplementedError("Render mode not implemented yet")