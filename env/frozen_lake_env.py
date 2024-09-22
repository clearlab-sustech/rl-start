from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt


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

    @property
    def observation_dim(self) -> int:
        return self.env.observation_space.n

    @property
    def action_dim(self) -> int:
        return self.env.action_space.n


if __name__ == '__main__':
    env = FrozenLakeEnv()
    action = 0
    obs, reward, done, info = env.step(action)
    env.render()
    print(env.observation_dim)