from collections import defaultdict
import numpy as np

class TdStateValueEstimator:

    def __init__(
        self, 
        env, 
        gamma: float = 1.0,
        constant_alpha: bool = False,
        alpha: float = 0.02,
    ) -> None:
        self.env = env
        self.policy = self.create_random_policy()
        self.gamma = gamma

        self.constant_alpha = constant_alpha
        self.alpha = alpha if constant_alpha else None
        
        self.values_table = defaultdict(float)
        self.states_count = defaultdict(int)

    def create_random_policy(self):
        policy = {}
        for state in range(self.env.observation_dim):
            policy[state] = np.ones(self.env.action_dim) / self.env.action_dim
        return policy

    def run_episode(self):
        obs = self.env.reset()

        while True:
            action = np.random.choice(np.arange(self.env.action_dim), p=self.policy[obs])
            next_obs, reward, done, info = self.env.step(action)

            # TD(0): 1-step TD
            TD_target = reward + self.gamma * self.values_table[next_obs]
            TD_error = TD_target - self.values_table[obs]

            self.states_count[obs] += 1
            if self.constant_alpha:
                alpha = self.alpha
            else:
                alpha = 1 / (self.states_count[obs] + 1)
            self.values_table[obs] += alpha * TD_error
            obs = next_obs

            if done:
                break

    def estimate_state_values(self, num_episode=10000):
        for _ in range(num_episode):
            self.run_episode()

    @property
    def sorted_values_table(self):
        for state in self.env.termination_states:
            self.values_table[state] = 0.
        return tuple(sorted(self.values_table.items()))


if __name__ == '__main__':

    from env.frozen_lake_env import FrozenLakeEnv

    env = FrozenLakeEnv()
    td_value_estimator = TdStateValueEstimator(
        env,
        gamma=0.99,
        constant_alpha=True,
        alpha=0.05
    )

    td_value_estimator.estimate_state_values(num_episode=20000)

    greedy_policy = env.compute_greedy_policy(td_value_estimator.sorted_values_table)
    print(greedy_policy)
    env.draw_state_values(td_value_estimator.sorted_values_table)
    env.draw_policy(greedy_policy)