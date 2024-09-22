import numpy as np
from collections import defaultdict

class TdActionValueEstimator:
         
    def __init__(self, env, gamma=1.0) -> None:
        self.env = env
        self.policy = self.create_random_policy()
        self.gamma = gamma

        self.q_table = defaultdict(lambda: np.zeros(self.env.action_dim))
        self.states_actions_count = defaultdict(lambda: np.zeros(self.env.action_dim))

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
    
    def update_q_values(self, traj):
        for t in reversed(range(len(traj))):
            obs, action, reward, next_obs = traj[t]
            
            # TD target uses the maximum Q value of the next state
            best_next_action = np.argmax(self.q_table[next_obs])
            TD_target = reward + self.gamma * self.q_table[next_obs][best_next_action]
            TD_error = TD_target - self.q_table[obs][action]
            
            # Update the count for state-action pair
            self.states_actions_count[obs][action] += 1
            alpha = 1 / (self.states_actions_count[obs][action] + 1)  # Learning rate
            self.q_table[obs][action] += alpha * TD_error

    def estimate_q_values(self, num_episodes=10000):
        for _ in range(num_episodes):
            traj = self.run_episode()
            self.update_q_values(traj)

    @property
    def sorted_q_table(self):
        return {state: list(q_values) for state, q_values in sorted(self.q_table.items())}


if __name__ == '__main__':
    
    from env.frozen_lake_env import FrozenLakeEnv

    env = FrozenLakeEnv()
    td_q_estimator = TdActionValueEstimator(env)
    
    td_q_estimator.estimate_q_values(num_episodes=10000)

    print(td_q_estimator.sorted_q_table)
    env.render()