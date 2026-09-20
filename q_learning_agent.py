"""
A from-scratch Q-learning agent for Gymnasium's CartPole-v1.

No RL library (stable-baselines3, etc.) is used — the Q-table, the
epsilon-greedy policy, and the Bellman update are all implemented manually
so every step is explainable.

CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
CartPole actions: 0 = push left, 1 = push right
"""

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_actions: int = 2,
        n_bins: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Bounds for each of the 4 CartPole state dimensions.
        # Velocity terms are theoretically unbounded, so we clip to a
        # reasonable practical range for discretization.
        self.state_bounds = [
            (-2.4, 2.4),    # cart position
            (-3.0, 3.0),    # cart velocity (clipped)
            (-0.21, 0.21),  # pole angle (radians, ~12 degrees)
            (-3.5, 3.5),    # pole angular velocity (clipped)
        ]

        # Q-table: one entry per (discretized state, action) pair.
        # Shape: (n_bins, n_bins, n_bins, n_bins, n_actions)
        self.q_table = np.zeros((n_bins,) * 4 + (n_actions,))

    def discretize_state(self, state):
        """Convert a continuous CartPole state into a tuple of bin indices."""
        bins = []
        for value, (low, high) in zip(state, self.state_bounds):
            value = np.clip(value, low, high)
            # Map value in [low, high] to a bin index in [0, n_bins - 1]
            scaled = (value - low) / (high - low)
            bin_index = int(scaled * (self.n_bins - 1))
            bins.append(bin_index)
        return tuple(bins)

    def choose_action(self, state_bins):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        # Exploit: best known action for this state
        return int(np.argmax(self.q_table[state_bins]))

    def update(self, state_bins, action, reward, next_state_bins, done):
        """
        Bellman update:
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        If the episode ended, there is no future value to bootstrap from.
        """
        current_q = self.q_table[state_bins][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_bins])
        self.q_table[state_bins][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
