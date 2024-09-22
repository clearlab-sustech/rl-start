import numpy as np

class SARSA:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[state, :])

    def train(self, num_episodes):
        """Train the agent using the SARSA algorithm."""
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                
                # SARSA update rule
                self.Q[state, action] += self.learning_rate * (
                    reward + self.discount_factor * self.Q[next_state, next_action] - self.Q[state, action])
                
                state = next_state
                action = next_action
            
            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
    
    def get_q_table(self):
        """Returns the learned Q-table."""
        return self.Q