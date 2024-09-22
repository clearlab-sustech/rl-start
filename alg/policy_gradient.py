import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

class VPGAgent:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.gamma = gamma  # Discount factor
        self.learning_rate = learning_rate
        
        # Policy Network
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.policy_network = PolicyNetwork(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
    
    def choose_action(self, state):
        """Sample an action based on the policy distribution."""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)
    
    def compute_returns(self, rewards):
        """Compute the discounted cumulative rewards (returns)."""
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns)
        return returns
    
    def train(self, num_episodes):
        """Train the agent using the Vanilla Policy Gradient algorithm."""
        for episode in range(num_episodes):
            state, _ = self.env.reset()  # gymnasium reset() returns tuple (state, info)
            log_probs = []
            rewards = []
            total_reward = 0
            done = False
            
            while not done:
                action, log_prob = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                
                state = next_state
            
            # Compute the returns (discounted cumulative rewards)
            returns = self.compute_returns(rewards)
            
            # Compute the policy gradient and perform gradient ascent
            loss = []
            for log_prob, return_value in zip(log_probs, returns):
                loss.append(-log_prob * return_value)
            loss = torch.cat(loss).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print(f"Episode: {episode+1}/{num_episodes}, Total Reward: {total_reward}")


if __name__ == '__main__':
    
    import gymnasium as gym

    env = gym.make('CartPole-v1')

    learning_rate = 1e-4
    gamma = 0.99
    num_episodes = 1000

    agent = VPGAgent(env, learning_rate, gamma)

    agent.train(num_episodes)