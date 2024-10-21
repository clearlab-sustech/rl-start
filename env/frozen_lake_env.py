from typing import List, Tuple

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt

from PIL import Image


class FrozenLakeEnv:
    '''

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

    def save_img(self, img_name: str = 'img.png') -> None:
        rgb_map = self.env.render()
        image = Image.fromarray(rgb_map)
        image.save(img_name)

    def draw_state_values(self, state_values: List[Tuple]):
        assert len(state_values) == 16 or len(state_values) == 11
        if len(state_values) == 16:
            state_values = [state_value for state_value in state_values if state_value[1] != 0]

        rgb_map = self.env.render()

        corner_list = [
            (64*0, 64*1),
            (64*1, 64*1),
            (64*2, 64*1),
            (64*3, 64*1),
            
            (64*0, 64*2),
            (64*2, 64*2),

            (64*0, 64*3),
            (64*1, 64*3),
            (64*2, 64*3),

            (64*1, 64*4),
            (64*2, 64*4),
        ]

        position_list = []
        offset = (5, -25)
        for corner in corner_list:
            position = list(corner)
            position[0] += offset[0]
            position[1] += offset[1]
            position_list.append(tuple(position))

        for i in range(len(position_list)):
            state_value = f"{state_values[i][1]:.3f}"
            cv2.putText(
                img=rgb_map, 
                text=str(state_value),
                org=position_list[i], 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.6, 
                color=(0, 0, 0), 
                thickness=2, 
                lineType=cv2.LINE_AA
            )
        
        plt.imshow(rgb_map)        
        plt.axis('off')
        plt.show()

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
    env.render()