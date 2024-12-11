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
    def __init__(self, is_slippery: bool = False) -> None:
        self.is_slippery = is_slippery
        self.env = gym.make(
            'FrozenLake-v1', 
            desc=None, 
            map_name="4x4", 
            is_slippery=is_slippery, 
            render_mode='rgb_array'
        )
    
        self.reset()
        
        self.state_space = [i for i in range(16)]
        self.termination_states = [5, 7, 11, 12, 15]
        self.init_state = 0
        self.goal_state = 15
        self.action_space = [MOVE_LEFT, MOVE_DOWN, MOVE_RIGHT, MOVE_UP]
        assert len(self.state_space) == self.env.observation_space.n
        assert len(self.action_space) == self.env.action_space.n

    def reset(self) -> int:
        obs, info = self.env.reset()
        return obs

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, info

    def render(self) -> None:
        rgb_map = self.env.render()
        plt.imshow(rgb_map)        
        plt.axis('off')
        plt.show()

    def save_img(self, rgb_map: np.ndarray, img_name: str='img.png') -> None:
        image = Image.fromarray(rgb_map)
        image.save(img_name)

    def draw_state_idx(self, show: bool = False) -> np.ndarray:
        self.env.reset()
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
        with_state_idx: bool = True,
        save: bool = False,
        img_name: str = 'state_values.png'
    ) -> None:
        if with_state_idx:
            rgb_map: np.ndarray = self.draw_state_idx()
        else:
            rgb_map: np.ndarray = self.env.render()

        upper_left_corner_list = [(64*i, 64*j) for j in range(4) for i in range(4)]

        offset = (5, 39)

        for state in self.state_space:
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
        
        if save:
            self.save_img(rgb_map, img_name)

        plt.imshow(rgb_map)        
        plt.axis('off')
        plt.show()

    def draw_policy(
        self, 
        policy, 
        with_state_idx: bool = True,
        save: bool = False,
        img_name: str = 'policy.png'
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

            l_end_position = []

            if float(policy[state][MOVE_LEFT]) != float(0):
                l_end_position.append((start_position[0]-arrow_length, start_position[1]))
            if float(policy[state][MOVE_RIGHT]) != float(0):
                l_end_position.append((start_position[0]+arrow_length, start_position[1]))
            if float(policy[state][MOVE_UP]) != float(0):
                l_end_position.append((start_position[0], start_position[1]-arrow_length))
            if float(policy[state][MOVE_DOWN]) != float(0):
                l_end_position.append((start_position[0], start_position[1]+arrow_length))

            for end_position in l_end_position:
                cv2.arrowedLine(
                    img=rgb_map, 
                    pt1=start_position, 
                    pt2=end_position,
                    color=(40, 40, 40),
                    thickness=3,
                    line_type=cv2.LINE_8,
                    tipLength=0.3
                )

        if save:
            self.save_img(rgb_map, img_name)

        plt.imshow(rgb_map)        
        plt.axis('off')
        plt.show()

    def compute_greedy_policy(
        self, 
        state_values: List[Tuple], 
        gamma: float = 1.00
    ) -> List[Tuple]:
        assert len(state_values) == len(self.state_space)

        greedy_policy = {}
        for state_value in state_values:
            state, _ = state_value

            if state in self.termination_states:
                greedy_policy[state] = None
                continue

            rewards_subsequent = []
            for action in self.action_space:
                next_state, reward = self.get_transition(state, action)
                rewards_subsequent.append((next_state, reward + gamma * state_values[next_state][1]))
            next_state = max(rewards_subsequent, key=lambda x: x[1])[0]

            if next_state - state == 4:
                greedy_policy[state] = [0, 1, 0, 0]
            elif next_state - state == 1:
                greedy_policy[state] = [0, 0, 1, 0]
            elif next_state - state == -1:
                greedy_policy[state] = [1, 0, 0, 0]
            else:
                greedy_policy[state] = [0, 0, 0, 1]

        return greedy_policy

    def set_state(self, desired_state: int, vis: bool = False) -> None:
        self.reset()

        if desired_state == 0:
            pass
        elif desired_state == 1:
            self.env.step(MOVE_RIGHT)
        elif desired_state == 2:
            self.env.step(MOVE_RIGHT)
            self.env.step(MOVE_RIGHT)
        elif desired_state == 3:
            self.env.step(MOVE_RIGHT)
            self.env.step(MOVE_RIGHT)
            self.env.step(MOVE_RIGHT)

        elif desired_state == 4:
            self.env.step(MOVE_DOWN)
        elif desired_state == 6:
            self.env.step(MOVE_RIGHT)
            self.env.step(MOVE_RIGHT)
            self.env.step(MOVE_DOWN)

        elif desired_state == 8:
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_DOWN)
        elif desired_state == 9:
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_RIGHT)
        elif desired_state == 10:
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_RIGHT)
            self.env.step(MOVE_RIGHT)

        elif desired_state == 13:
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_RIGHT)
            self.env.step(MOVE_DOWN)
        elif desired_state == 14:
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_RIGHT)
            self.env.step(MOVE_DOWN)
            self.env.step(MOVE_RIGHT)

        else:
            print("[ERROR] Invalid setting.")

        if vis:
            self.env.render()

    def get_transition(self, state: int, action: int) -> Tuple[int, float]:
        self.set_state(state)
        next_state, reward, _, _ = self.step(action)
        return next_state, reward

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
    # env.draw_state_idx(show=True)

    env.set_state(14, vis=True)