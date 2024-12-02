from alg.monte_carlo import MonteCarloStateValueEstimator
from env.frozen_lake_env import FrozenLakeEnv

env = FrozenLakeEnv()
mc_value_estimator = MonteCarloStateValueEstimator(
    env,
    gamma=0.99,
    type='first-visit',
    incremental=False,
    constant_alpha=True,
    alpha=0.05
)

env.draw_policy(
    mc_value_estimator.create_random_policy(), 
    save=True, 
    img_name='./plot/random_policy.png'
)

mc_value_estimator.estimate_state_values(num_episode=20000)

greedy_policy = env.compute_greedy_policy(mc_value_estimator.sorted_values_table)
env.draw_state_values(
    mc_value_estimator.sorted_values_table,
    save=True,
    img_name='./plot/state_values.png'
)
env.draw_policy(
    greedy_policy,
    save=True,
    img_name='./plot/greedy_policy.png'
)