import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Hyperparameters ---
NUM_EPISODES = 10000
LEARNING_RATE = 0.1      # alpha: how much new info overrides old info
DISCOUNT_FACTOR = 0.99   # gamma: importance of future rewards
EPSILON_START = 1.0      # start 100% exploring
EPSILON_END = 0.01       # minimum exploration rate
EPSILON_DECAY = 0.9995   # decay multiplier per episode

def create_bins():
    """
    CartPole continuous state space has 4 values:
    [Cart Position, Cart Velocity, Pole Angle, Pole Velocity at Tip]
    
    We need to bin these continuous values into discrete states for Q-learning.
    Here we define 10 boundaries for each, resulting in 11 bins.
    """
    bins = [
        np.linspace(-2.4, 2.4, 10),     # Cart Position
        np.linspace(-3.0, 3.0, 10),     # Cart Velocity
        np.linspace(-0.209, 0.209, 10), # Pole Angle (radians, ~12 degrees)
        np.linspace(-3.0, 3.0, 10)      # Pole Velocity
    ]
    return bins

def discretize_state(state, bins):
    """
    Converts a continuous state from the environment into a discrete tuple of bin indices.
    """
    discrete_state = []
    for i in range(len(state)):
        # np.digitize returns the index of the bin to which the value belongs
        # -1 makes it 0-indexed (0 to 10 for 11 bins)
        bin_index = np.digitize(state[i], bins[i]) - 1
        discrete_state.append(bin_index)
    return tuple(discrete_state)

def train_q_learning():
    env = gym.make('CartPole-v1')
    bins = create_bins()
    
    # Initialize Q-table
    # The shape is (11, 11, 11, 11, 2) because we have 4 state variables with 11 bins each,
    # and 2 possible actions (0=Left, 1=Right). We initialize with zeros.
    q_table_shape = (11, 11, 11, 11, env.action_space.n)
    q_table = np.zeros(q_table_shape)
    
    epsilon = EPSILON_START
    episode_rewards = []
    
    for episode in range(NUM_EPISODES):
        # Reset environment for a new episode
        state, info = env.reset()
        discrete_state = discretize_state(state, bins)
        
        total_reward = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            # --- Epsilon-Greedy Action Selection ---
            if np.random.random() < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()
            else:
                # Exploit: choose the best action according to our Q-table
                action = np.argmax(q_table[discrete_state])
                
            # Take the action and observe the outcome
            next_state, reward, done, truncated, info = env.step(action)
            next_discrete_state = discretize_state(next_state, bins)
            
            # --- Q-Table Update (Bellman Equation) ---
            # Q(s, a) = Q(s, a) + alpha * [Reward + gamma * max(Q(s', a')) - Q(s, a)]
            best_next_action = np.argmax(q_table[next_discrete_state])
            
            # Current Q-value
            current_q = q_table[discrete_state][action]
            # Max possible Q-value for the next state
            max_future_q = q_table[next_discrete_state][best_next_action]
            
            # Target Q-value based on immediate reward and discounted future reward
            if done and not truncated:
                td_target = reward
            else:
                td_target = reward + DISCOUNT_FACTOR * max_future_q
                
            # Calculate the difference (TD Error)
            td_error = td_target - current_q
            
            # Update the Q-table
            q_table[discrete_state][action] += LEARNING_RATE * td_error
            
            # Move to the next state
            discrete_state = next_discrete_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        
        # Decay epsilon: reduce exploration over time
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-500:])
            print(f"Episode {episode + 1:5d} | Avg Reward (last 500): {avg_reward:5.1f} | Epsilon: {epsilon:.3f}")
            
    env.close()
    return episode_rewards

if __name__ == "__main__":
    print("Starting Q-learning training on CartPole-v1...")
    rewards = train_q_learning()
    
    # Calculate moving average for a smoother learning curve
    window_size = 100
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, color='gray', label='Episode Reward')
    plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, color='blue', linewidth=2, label=f'{window_size}-Episode Moving Average')
    plt.title('Q-Learning on CartPole-v1: Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Duration)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('learning_curve.png')
    print("Training complete! Learning curve saved to 'learning_curve.png'.")
