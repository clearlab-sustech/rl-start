from collections import defaultdict

import numpy as np


class TdStateValueEstimator:

    def __init__(self, env, gamma=1.0) -> None:
        self.env = env
        self.policy = self.create_random_policy()
        self.gamma = gamma

        self.values_table = defaultdict(float)
        self.states_count = defaultdict(int)

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
            traj.append((obs, action, reward, next_obs))
            obs = next_obs

            if done:
                break

        return traj
    
    def update_state_values(self, traj):
        for t in reversed(range(len(traj))):
            obs, action, reward, next_obs = traj[t]
            TD_target = reward + self.gamma * self.values_table[next_obs]
            TD_error = TD_target - self.values_table[obs]
            
            self.states_count[obs] += 1
            alpha = 1 / (self.states_count[obs] + 1)
            self.values_table[obs] += alpha * TD_error

    def estimate_state_values(self, num_episode=10000):
        for _ in range(num_episode):
            traj = self.run_episode()
            self.update_state_values(traj)

    @property
    def sorted_values_table(self):
        return tuple(sorted(self.values_table.items()))


if __name__ == '__main__':
    
    from env.frozen_lake_env import FrozenLakeEnv

    env = FrozenLakeEnv()
    td_value_estimator = TdStateValueEstimator(env)
    
    td_value_estimator.estimate_state_values(num_episode=10000)

    greedy_policy = env.compute_greedy_policy(td_value_estimator.sorted_values_table)
    print(greedy_policy)
    env.draw_state_values(td_value_estimator.sorted_values_table)
    env.draw_policy(greedy_policy)