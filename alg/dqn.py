from collections import deque
from typing import List, Union

import numpy as np
import torch

from utils.mlp import MLP
    

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):  
        # TODO: check dtype for obs, action, next_obs, and done.
        obs = torch.tensor(obs, dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs
        next_obs = torch.tensor(next_obs, dtype=torch.float32) if not isinstance(next_obs, torch.Tensor) else next_obs
        action = torch.tensor(action, dtype=torch.int64) if not isinstance(action, torch.Tensor) else action
        reward = torch.tensor(reward, dtype=torch.float32) if not isinstance(reward, torch.Tensor) else reward
        done = torch.tensor(done, dtype=torch.float32) if not isinstance(done, torch.Tensor) else done
        
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        indices = torch.randint(0, len(self.buffer), (batch_size,))  # (batch_size, )
        batch = [self.buffer[idx] for idx in indices]

        obs, action, reward, next_obs, done = zip(*batch)
        obs = torch.stack(obs)           # (batch_size, obs_dim)
        next_obs = torch.stack(next_obs) # (batch_size, obs_dim)
        action = torch.stack(action)     # (batch_size, action_dim)
        if action.shape == torch.stack(reward).shape:   # action_dim = 1, shape of action: (batch_size, )
            action = action.unsqueeze(-1)  # (batch_size, action_dim)
        reward = torch.stack(reward).unsqueeze(-1) # (batch_size, 1)
        done = torch.stack(done).unsqueeze(-1)     # (batch_size, 1)

        return obs, action, reward, next_obs, done

    @property
    def current_num_items(self):
        return len(self.buffer)


class DQN:
    '''
    :param env: 
    :param learning_rate: learning rate
    :param replay_buffer_capacity: capacity of the replay buffer
    :param batch_size: Minibatch size for each gradient update
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    '''
    def __init__(
        self,
        env,
        obs_dim: int,
        num_actions: int,
        has_truncated_info: bool = False,
        learning_rate: float = 1e-4,
        replay_buffer_capacity: int = 1e6,
        gamma: float = 0.99,
        epsilon: float = 0.005,
        target_qnet_update: int = 10,
        batch_size: int = 32,
        hidden_size: List[int] = [128, 128],
        learning_starts: int = 100,
        device: str = 'cpu'
    ) -> None:
        
        self.env = env
        self.has_truncated_info = has_truncated_info
        self.device = device

        self.replay_buffer = ReplayBuffer(int(replay_buffer_capacity))
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_qnet_update = target_qnet_update

        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.Q_net = MLP(
            input_size=obs_dim,
            output_size=num_actions,
            hidden_size=hidden_size,
        ).to(device)
        self.target_Q_net = MLP(
            input_size=obs_dim,
            output_size=num_actions,
            hidden_size=hidden_size,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=learning_rate)

        self.qnet_update_counter = 0


    def epsilon_greedy_policy(self, obs: Union[np.ndarray, torch.Tensor]):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32) # (obs_dim, )
        obs = obs.unsqueeze(0)      # (1, obs_dim)
        q_values = self.Q_net(obs)  # (1, num_action)

        if torch.rand(1).item() < self.epsilon:
            action = torch.randint(0, self.num_actions, (1,)).item()
        else:
            action = torch.argmax(q_values, dim=1).item()
        
        return action        


    def _pre_data_collection(self):
        obs, _ = self.env.reset()
        truncated = False
        for _ in range(self.learning_starts):
            action = self.epsilon_greedy_policy(obs)
            if self.has_truncated_info:
                next_obs, reward, done, truncated, _ = self.env.step(action)
            else:
                next_obs, reward, done, _ = self.env.step(action)
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            if done or truncated:
                obs, _ = self.env.reset()


    def _update_qnet(self, obs, actions, rewards, next_obs, dones):
        with torch.no_grad():
            greedy_next_action_values = self.target_Q_net(next_obs).max(1)[0].unsqueeze(-1)       # (batch_size, 1)
            target_action_values = rewards + (1 - dones) * self.gamma * greedy_next_action_values # (batch_size, 1)

        estimated_action_values = torch.gather(self.Q_net(obs), dim=1, index=actions)

        loss = torch.nn.functional.mse_loss(estimated_action_values, target_action_values).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.qnet_update_counter % self.target_qnet_update == 0:
            self.target_Q_net.load_state_dict(self.Q_net.state_dict())

        self.qnet_update_counter += 1

        # print(loss.item())


    def train(self, num_learning_iters: int):

        l_return = []

        self._pre_data_collection()

        # start training
        for iter_idx in range(num_learning_iters):
            episode_return = 0
            done = False
            # obs_ 
            obs_, _ = self.env.reset()
            while not done:
                action = self.epsilon_greedy_policy(obs_)
                if self.has_truncated_info:
                    next_obs, reward, done, truncated, _ = self.env.step(action)
                else:
                    next_obs, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(obs_, action, reward, next_obs, done)
                obs_ = next_obs
                episode_return = reward + self.gamma * episode_return

                obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
                self._update_qnet(obs, actions, rewards, next_obs, dones)

            l_return.append(episode_return)
            print(iter_idx, episode_return)


def test_from_stable_baseline():
    import gymnasium as gym

    from stable_baselines3 import DQN

    env = gym.make("CartPole-v1", render_mode="human")

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("dqn_cartpole")

    del model # remove to demonstrate saving and loading

    model = DQN.load("dqn_cartpole")

    obs, info = env.reset()
    while True:
        action, _obss = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


if __name__ == '__main__':
    # test_from_stable_baseline()

    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="human")

    alg = DQN(
        env,
        obs_dim=4,
        num_actions=2,
        has_truncated_info=True,
        hidden_size=[64, 32],
        learning_rate=2e-3,
    )

    alg.train(num_learning_iters=500)