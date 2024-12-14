import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def compute_advantage(gamma, lmbda, td_delta):
    """
    Compute advantage using GAE (Generalized Advantage Estimation)
    Args:
        gamma: discount factor
        lmbda: GAE parameter
        td_delta: temporal difference
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PPO:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        lmbda: float,
        epochs: int,
        eps: float,
        gamma: float,
        device: str,
    ):
        """Initialize PPO algorithm with networks and parameters

        Args:
            state_dim: Dimension of state space
            hidden_dim: Dimension of hidden layers in neural networks
            action_dim: Dimension of action space
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic network
            lmbda: Lambda parameter for GAE (Generalized Advantage Estimation)
            epochs: Number of epochs when optimizing the surrogate objective
            eps: Clip parameter for PPO
            gamma: Discount factor
            device: Device to run the model on (cpu/cuda)
        """
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state, test=False):
        """
        Select action based on state
        Args:
            state: current state
            test: if True, select action deterministically
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        if test:
            action = torch.argmax(probs, dim=1)
        else:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        """
        Update actor and critic networks
        Args:
            transition_dict: contains states, actions, rewards, next_states, dones
        """
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        # Compute TD target and advantage
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu())
        advantage = advantage.to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # PPO update for multiple epochs
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def train_episode(env, agent):
    """
    Train for one episode
    Returns:
        episode_return: total reward for this episode
    """
    states, actions, rewards, next_states, dones = [], [], [], [], []
    state = env.reset()
    if isinstance(state, tuple):  # Handle new gym API
        state = state[0]
    episode_return = 0
    done = False

    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Handle both termination conditions

        if isinstance(next_state, tuple):
            next_state = next_state[0]

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

        state = next_state
        episode_return += reward

    transition_dict = {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "next_states": np.array(next_states),
        "dones": np.array(dones),
    }
    agent.update(transition_dict)
    return episode_return


def train(env, agent, num_episodes):
    """
    Training loop
    Args:
        env: gym environment
        agent: PPO agent
        num_episodes: number of episodes to train
    Returns:
        return_list: list of returns for each episode
    """
    return_list = []
    for i in range(num_episodes):
        episode_return = train_episode(env, agent)
        return_list.append(episode_return)
        if (i + 1) % 10 == 0:
            print(f"Episode {i+1}, Return: {episode_return}")
    return return_list


if __name__ == "__main__":
    # Hyperparameters
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Environment setup
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.action_space.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent and train
    agent = PPO(
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    )
    return_list = train(env, agent, num_episodes)

    # Plot results
    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("PPO on {}".format(env_name))
    plt.savefig("ppo_training_curve_moving_average.png")
    plt.close()

    env.close()
