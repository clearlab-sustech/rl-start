from collections import defaultdict

import numpy as np


class MonteCarloStateValueEstimator:

    def __init__(
        self, 
        env,
        gamma: float = 1.0,
        type: str = 'every-visit',
    ) -> None:
        assert type in ['first-visit', 'every-visit']

        self.env = env
        self.policy = self.create_random_policy()
        self.gamma = gamma

        self.type = type

        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.values_table = defaultdict(float)

    def create_random_policy(self):
        policy = {}
        for state in range(self.env.observation_dim):
            policy[state] = np.ones(self.env.action_dim) / self.env.action_dim
        return policy

    def run_episode(self):
        traj = []
        obs = self.env.reset()
        while True:
            action = np.random.choice(np.arange(self.env.action_dim), p=self.policy[obs])
            next_obs, reward, done, info = self.env.step(action)
            traj.append((obs, action, reward))
            obs = next_obs

            if done:
                break

        return traj
    
    def update_state_values(self, traj):
        G = 0
        if self.type == 'every-visit':
            for t in reversed(range(len(traj))):
                obs, action, reward = traj[t]
                G = reward + self.gamma * G
                self.returns_sum[obs] += G
                self.returns_count[obs] += 1
                self.values_table[obs] = self.returns_sum[obs] / self.returns_count[obs]
        else:
            visited_states = set()
            for t in reversed(range(len(traj))):
                obs, action, reward = traj[t]
                G = reward + self.gamma * G
                if obs not in visited_states:
                    self.returns_sum[obs] += G
                    self.returns_count[obs] += 1
                    self.values_table[obs] = self.returns_sum[obs] / self.returns_count[obs]
                    visited_states.add(obs)

    def estimate_state_values(self, num_episode=10000):
        for _ in range(num_episode):
            traj = self.run_episode()
            self.update_state_values(traj)

    @property
    def sorted_values_table(self):
        for state in self.env.termination_states:
            self.values_table[state] = 0.
        return tuple(sorted(self.values_table.items()))


if __name__ == '__main__':
    
    from env.frozen_lake_env import FrozenLakeEnv

    env = FrozenLakeEnv()
    mc_value_estimator = MonteCarloStateValueEstimator(env, type='every-visit')
    
    mc_value_estimator.estimate_state_values(num_episode=20000)

    greedy_policy = env.compute_greedy_policy(mc_value_estimator.sorted_values_table)
    print(greedy_policy)
    env.draw_state_values(mc_value_estimator.sorted_values_table)
    env.draw_policy(greedy_policy)