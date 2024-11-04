from typing import List, Tuple

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


MOVE_LEFT = 0
MOVE_DOWN = 1
MOVE_RIGHT = 2
MOVE_UP = 3

class FrozenLakeEnv:
    '''
    An enhanced version of the frozen lake environment in Gymnasium.
    
    See https://gymnasium.farama.org/environments/toy_text/frozen_lake/ 
    for more information.
    '''
    def __init__(self) -> None:
        self.env = gym.make(
            'FrozenLake-v1', 
            desc=None, 
            map_name="4x4", 
            is_slippery=False, 
            render_mode='rgb_array'
        )
    
        # obs, info = self.reset()
        obs = self.reset()
        
        self.adjacency_info = {
            0: [1, 4],
            1: [0, 2, 5],
            2: [1, 3, 6],
            3: [2, 7],
            4: [0, 5, 8],
            5: [1, 4, 6, 9],
            6: [2, 5, 7, 10],
            7: [3, 6, 11],
            8: [4, 9, 12],
            9: [5, 8, 10, 13],
            10: [6, 9, 11, 14],
            11: [7, 10, 15],
            12: [8, 13],
            13: [9, 12, 14],
            14: [10, 13, 15],
            15: [11, 14]
        }
        self.state_space = [i for i in range(16)]
        self.termination_states = [5, 7, 11, 12, 15]
        self.init_state = 0
        self.goal_state = 15
        self.action_space = [MOVE_LEFT, MOVE_DOWN, MOVE_RIGHT, MOVE_UP]
        assert len(self.state_space) == self.env.observation_space.n
        assert len(self.action_space) == self.env.action_space.n

    def reset(self, return_info=False) -> Tuple[int, dict]:
        obs, info = self.env.reset()
        # print("obs = {}, info = {}".format(obs, info))
        # return obs, info if return_info else obs
        return obs

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, info

    def render(self) -> None:
        rgb_map = self.env.render()
        plt.imshow(rgb_map)        
        plt.axis('off')
        plt.show()

    def save_img(self, img_name: str='img.png') -> None:
        rgb_map = self.env.render()
        image = Image.fromarray(rgb_map)
        image.save(img_name)

    def draw_state_idx(self, show: bool = False) -> np.ndarray:
        rgb_map: np.ndarray = self.env.render()

        upper_left_corner_list = [(64*i, 64*j) for j in range(4) for i in range(4)]

        offset = (2, 17)
        for state in self.state_space:
            position = list(upper_left_corner_list[state])
            position[0] += offset[0]
            position[1] += offset[1]

            cv2.putText(
                img=rgb_map,
                text=str(state),
                org=position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(128, 0, 128),
                thickness=2, 
                lineType=cv2.LINE_AA
            )

        if show:
            plt.imshow(rgb_map)        
            plt.axis('off')
            plt.show()

        return rgb_map

    def draw_state_values(
        self, 
        state_values: List[Tuple], 
        with_state_idx: bool = True
    ) -> None:
        if with_state_idx:
            rgb_map: np.ndarray = self.draw_state_idx()
        else:
            rgb_map: np.ndarray = self.env.render()

        upper_left_corner_list = [(64*i, 64*j) for j in range(4) for i in range(4)]

        offset = (5, 39)

        for state in self.state_space:
            if state in self.termination_states:
                continue

            position = list(upper_left_corner_list[state])
            position[0] += offset[0]
            position[1] += offset[1]

            state_value = f"{state_values[state][1]:.3f}"
            cv2.putText(
                img=rgb_map, 
                text=str(state_value),
                org=position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.6, 
                color=(40, 40, 40),
                thickness=1, 
                lineType=cv2.LINE_AA
            )
        
        plt.imshow(rgb_map)        
        plt.axis('off')
        plt.show()

    def draw_policy(
        self, 
        policy, 
        with_state_idx: bool = True
    ) -> None:
        arrow_length = 20

        if with_state_idx:
            rgb_map: np.ndarray = self.draw_state_idx()
        else:
            rgb_map: np.ndarray = self.env.render()

        upper_left_corner_list = [(64*i, 64*j) for j in range(4) for i in range(4)]

        start_pt_offset = (32, 32)

        for state in self.state_space:
            if state in self.termination_states:
                continue

            start_position = list(upper_left_corner_list[state])
            start_position[0] += start_pt_offset[0]
            start_position[1] += start_pt_offset[1]
            
            if policy[state][1] == MOVE_LEFT:
                end_position = (start_position[0]-arrow_length, start_position[1])
            elif policy[state][1] == MOVE_RIGHT:
                end_position = (start_position[0]+arrow_length, start_position[1])
            elif policy[state][1] == MOVE_UP:
                end_position = (start_position[0], start_position[1]-arrow_length)
            else:
                end_position = (start_position[0], start_position[1]+arrow_length)

            cv2.arrowedLine(
                img=rgb_map, 
                pt1=start_position, 
                pt2=end_position,
                color=(40, 40, 40),
                thickness=3,
                line_type=cv2.LINE_8,
                tipLength=0.3
            )

        plt.imshow(rgb_map)        
        plt.axis('off')
        plt.show()

    def compute_greedy_policy(self, state_values: List[Tuple]) -> List[Tuple]:
        assert len(state_values) == len(self.state_space)

        greedy_policy = []
        for state_value in state_values:
            state, _ = state_value
            
            if state in self.termination_states:
                greedy_policy.append((state, None))
                continue

            adjacency_states = self.adjacency_info[state]

            if self.goal_state in adjacency_states:
                next_state = self.termination_states[-1]
            else:
                adjacency_values = []
                for adjacency_state in adjacency_states:
                    adjacency_values.append((adjacency_state, state_values[adjacency_state][1]))
                next_state = max(adjacency_values, key=lambda x: x[1])[0]

            if next_state - state == 4:
                greedy_policy.append((state, MOVE_DOWN))
            elif next_state - state == 1:
                greedy_policy.append((state, MOVE_RIGHT))
            elif next_state - state == -1:
                greedy_policy.append((state, MOVE_LEFT))
            else:
                greedy_policy.append((state, MOVE_UP))

        return greedy_policy

    @property
    def observation_dim(self) -> int:
        return self.env.observation_space.n

    @property
    def action_dim(self) -> int:
        return self.env.action_space.n


if __name__ == '__main__':
    env = FrozenLakeEnv()
    action = 2
    obs, reward, done, info = env.step(action)
    action = 2
    obs, reward, done, info = env.step(action)
    env.draw_state_idx(show=True)