import numpy as np
import matplotlib.pyplot as plt

class QLearningGridworld:
    def __init__(self, grid_size=4, n_actions=4, goal_state=15,
                 learning_rate=0.8, discount_factor=0.95,
                 exploration_prob=0.2, epochs=1000):

        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = n_actions
        self.goal_state = goal_state

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_prob
        self.epochs = epochs

        self.Q_table = np.zeros((self.n_states, self.n_actions))
        # print(self.Q_table)

    def get_next_state(self, state, action):
        row, col = divmod(state, self.grid_size)

        if action == 0 and col > 0:               # left
            col -= 1
        elif action == 1 and col < self.grid_size - 1:  # right
            col += 1
        elif action == 2 and row > 0:             # up
            row -= 1
        elif action == 3 and row < self.grid_size - 1:  # down
            row += 1

        return row * self.grid_size + col

    def train(self):
        for _ in range(self.epochs):
            current_state = np.random.randint(0, self.n_states)

            while True:
                # Exploration vs exploitation
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = np.argmax(self.Q_table[current_state])

                next_state = self.get_next_state(current_state, action)

                reward = 1 if next_state == self.goal_state else 0

                # Q-learning update
                self.Q_table[current_state, action] += self.lr * (
                    reward + self.gamma * np.max(self.Q_table[next_state])
                    - self.Q_table[current_state, action]
                )

                if next_state == self.goal_state:
                    break

                current_state = next_state

    def plot_q_values(self):
        print(f'q table {self.Q_table}')
        q_values_grid = np.max(self.Q_table, axis=1).reshape((self.grid_size, self.grid_size))
        print(q_values_grid)
        plt.figure(figsize=(6, 6))
        plt.imshow(q_values_grid, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Q-value')
        plt.title('Learned Q-values for Each State')
        plt.xticks(np.arange(self.grid_size), [str(i) for i in range(self.grid_size)])
        plt.yticks(np.arange(self.grid_size), [str(i) for i in range(self.grid_size)])
        plt.gca().invert_yaxis()
        plt.grid(True)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                plt.text(j, i, f'{q_values_grid[i, j]:.2f}',
                         ha='center', va='center', color='black')

        plt.show()

    def print_q_table(self):
        print("Learned Q-table:")
        print(self.Q_table)


# -------------------------
# Run the Q-learning agent
# -------------------------

agent = QLearningGridworld()
agent.train()
agent.plot_q_values()
agent.print_q_table()