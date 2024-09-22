import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001, batch_size=64, max_memory_size=10000):
        self.env = env
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay
        self.epsilon_min = epsilon_min  # Minimum epsilon
        self.learning_rate = learning_rate  # Learning rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory_size)  # Replay buffer
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        
        # Neural network model
        self.model = DQN(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()  # Exploit: select action with max Q-value

    def replay(self):
        """Sample a batch from memory and perform a training step."""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Q(s, a) for current state-action pairs
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # Q(s', a') for next state-action pairs
        next_q = self.model(next_states).max(1)[0]
        expected_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss and optimize the model
        loss = self.criterion(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes):
        """Train the agent using Deep Q-Learning."""
        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.input_dim])
            done = False
            total_reward = 0
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.input_dim])
                total_reward += reward
                
                # Store experience in memory
                self.remember(state, action, reward, next_state, done)
                
                # Move to the next state
                state = next_state
                
                # Train the model by replaying experiences
                self.replay()
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            print(f"Episode: {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
    
    def save(self, filename):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), filename)
    
    def load(self, filename):
        """Load the model from a file."""
        self.model.load_state_dict(torch.load(filename))