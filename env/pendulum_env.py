from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class PendulumEnv:
    '''

    See https://gymnasium.farama.org/environments/classic_control/pendulum/
    for more information.
    '''
    def __init__(self) -> None:
        self.env = gym.make(
            'Pendulum-v1', 
            # num_envs=1,
            g=9.81, 
            render_mode='rgb_array'
        )
    
        # obs, info = self.reset()
        obs = self.reset()

    def reset(self, return_info=False) -> Tuple[int, dict]:    
        obs, info = self.env.reset()
        # print("obs = {}, info = {}".format(obs, info))
        # return obs, info if return_info else obs
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, info

    def render(self) -> None:
        rgb_map = self.env.render()
        plt.imshow(rgb_map)
        plt.axis('off')
        plt.show()

    @property
    def observation_dim(self) -> int:
        return self.env.observation_space.shape[0]
    
    @property
    def action_dim(self) -> int:
        return self.env.action_space.shape[0]
    
    @property
    def observation_space(self) -> np.ndarray:
        return self.env.observation_space

    @property
    def action_space(self) -> np.ndarray:
        return self.env.action_space


if __name__ == '__main__':
    env = PendulumEnv()
    print('observation dimension = {}'.format(env.observation_dim))
    print('action_dimension = {}'.format(env.action_dim))

    action = env.action_space.sample()
    print(type(action))
    print(action)
    obs, reward, done, info = env.step(action)
    print(obs.shape)
    print(reward.shape)
    print(done)
    print(type(info))