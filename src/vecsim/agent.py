import numpy as np
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size   # 4 inputs
        self.action_size = action_size  # 2 outputs
        self.epsilon     = 1.0          # start fully exploring
        self.epsilon_min = 0.01         # never stop exploring completely
        self.epsilon_decay = 0.995      # how fast to shift to exploitation
        self.learning_rate = 0.001
        self.memory      = []           # replay buffer
        self.memory_size = 1000

        # Neural network weights — random initialization
        self.w1 = np.random.randn(state_size, 16) * 0.1
        self.b1 = np.zeros((1, 16))
        self.w2 = np.random.randn(16, action_size) * 0.1
        self.b2 = np.zeros((1, action_size))

    def _relu(self, x):
        return np.maximum(0, x)

    def _forward(self, state):
        state = np.array(state).reshape(1, -1)
        self.z1 = state @ self.w1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        return self.z2   # Q-values for each action

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = self._forward(state)
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def learn(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in batch:
            q_values    = self._forward(state)
            next_q      = self._forward(next_state)
            target      = reward + 0.9 * np.max(next_q)
            error       = target - q_values[0][action]

            # Backpropagation
            dz2         = np.zeros_like(q_values)
            dz2[0][action] = -error
            dw2         = self.a1.T @ dz2
            db2         = dz2
            da1         = dz2 @ self.w2.T
            dz1         = da1 * (self.z1 > 0)
            state_arr   = np.array(state).reshape(1, -1)
            dw1         = state_arr.T @ dz1
            db1         = dz1

            self.w1    -= self.learning_rate * dw1
            self.b1    -= self.learning_rate * db1
            self.w2    -= self.learning_rate * dw2
            self.b2    -= self.learning_rate * db2

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def save(self, filepath):
        np.save(filepath, {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2
        })

    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True).item()
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']