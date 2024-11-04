from collections import defaultdict

import numpy as np


class DynamicProgrammingValueEstimator:

    def __init__(self, env, gamma=1.0, theta=1e-10) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.values_table = defaultdict(float)

    def policy_evaluation(self, policy):
        prob = 1  # prob of state transition

        while True:
            delta = 0
            for state in self.env.state_space:
                if state in self.env.termination_states:
                    continue
                current_vs = self.values_table[state]
                next_vs = 0
                for action in self.env.action_space:
                    next_state, reward = self.env.get_transition(state, action)
                    next_vs += policy[state][action] * prob * (reward + self.gamma * self.values_table[next_state])
                self.values_table[state] = next_vs
                delta = max(delta, abs(current_vs - next_vs))

            if delta < self.theta:
                break

    def estimate_state_values(self, policy):
        self.policy_evaluation(policy)

    @property
    def sorted_values_table(self):
        return tuple(sorted(self.values_table.items()))


if __name__ == '__main__':

    from env.frozen_lake_env import FrozenLakeEnv

    env = FrozenLakeEnv()

    policy = {state: np.ones(env.action_dim) / env.action_dim for state in range(env.observation_dim)}

    dp_value_estimator = DynamicProgrammingValueEstimator(env)

    dp_value_estimator.estimate_state_values(policy)

    greedy_policy = env.compute_greedy_policy(dp_value_estimator.sorted_values_table)
    print(greedy_policy)
    env.draw_state_values(dp_value_estimator.sorted_values_table)
    env.draw_policy(greedy_policy)