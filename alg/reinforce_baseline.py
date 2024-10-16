import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import utils.rl_utils as rl_utils


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


class REINFORCEwithBaseline:
    def __init__(self, state_dim, hidden_dim, action_dim, policynet_lr, valuenet_lr, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.value_net = ValueNet(state_dim, hidden_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                lr=policynet_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),
                                                 lr=valuenet_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)

            value = self.value_net(state)
            G = self.gamma * G + reward
            delta = G - value.item()
        
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            policy_loss = -log_prob * delta
            policy_loss.backward()
            
            value_loss = torch.nn.functional.mse_loss(value, torch.tensor([[G]]).to(self.device))
            value_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()


if __name__ == "__main__":
    policynet_lr = 1e-3
    valuenet_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.action_space.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEwithBaseline(state_dim, hidden_dim, action_dim, policynet_lr, valuenet_lr, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("REINFORCE on {}".format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("REINFORCE on {}".format(env_name))
    plt.show()

    env.close()
