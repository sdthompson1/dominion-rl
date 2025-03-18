import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from datetime import datetime
import time
from typing import List, Dict, Tuple, Optional, Deque, Any

from environment import Environment
from strategy import Strategy

# Experience replay buffer structure
class Experience:
    def __init__(self, state, action, reward, next_state, next_legal_actions):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.next_legal_actions = next_legal_actions

# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

        # Initialize weights using He initialization
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN Agent implementation
class DQNAgent:
    def __init__(self,
                 env: Environment,
                 learning_rate,
                 discount_factor,
                 exploration_rate,
                 min_exploration_rate,
                 exploration_decay,
                 predefined_strategy_probability,
                 predefined_strategy_decay,
                 predefined_strategy: Strategy,
                 batch_size,
                 replay_buffer_size,
                 target_update_freq):

        self.env = env
        self.state_size = env.state_size
        self.num_actions = env.num_actions

        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = min_exploration_rate
        self.epsilon_decay = exploration_decay
        self.predefined_strategy_probability = predefined_strategy_probability
        self.predefined_strategy_decay = predefined_strategy_decay
        self.predefined_strategy = predefined_strategy
        self.batch_size = batch_size
        self.replay_memory_size = replay_buffer_size
        self.target_network_update_freq = target_update_freq
        self.training_steps = 0

        # Initialize primary and target networks
        self.primary_network = QNetwork(self.state_size, self.num_actions)
        self.target_network = QNetwork(self.state_size, self.num_actions)

        # Make target network have same weights as primary network
        self.target_network.load_state_dict(self.primary_network.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.primary_network.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Stat tracking for plots
        self.episode_rewards = []
        self.sample_rewards = []
        self.episode_steps = []
        self.episode_epsilon = []
        self.episode_predefined_prob = []
        self.q_value_stats = []
        self.q_value_sample_freq = 20
        self.action_distributions = []

    # Choose action based on epsilon-greedy policy (or predefined strategy, if available)
    def choose_action(self, state, legal_actions):
        # Count the number of legal actions
        num_legal_actions = legal_actions.bit_count()

        # Apply predefined_strategy if applicable
        if self.predefined_strategy and random.random() <= self.predefined_strategy_probability:
            return self.predefined_strategy.choose_action(self.env, legal_actions)

        # Otherwise, with probability epsilon, choose a random action (exploration)
        if random.random() <= self.epsilon:
            # Select a random legal action
            r = random.randint(0, num_legal_actions - 1)

            # Find the rth set bit
            for i in range(self.num_actions):
                bit = (1 << i)
                if (legal_actions & bit) != 0:
                    if r == 0:
                        return i
                    r -= 1

            # This shouldn't happen if the above code is correct
            raise RuntimeError("failed to select random action")

        # Otherwise, choose the best action according to the model (exploitation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            q_values = self.primary_network(state_tensor)
            q_values = q_values.squeeze().numpy()

            best_idx = 0
            best_value = float('-inf')

            for i in range(self.num_actions):
                bit = (1 << i)

                if (legal_actions & bit) != 0 and q_values[i] >= best_value:
                    best_value = q_values[i]
                    best_idx = i

            return best_idx

    # Store experience in replay buffer
    def remember(self, state, action, reward, next_state, next_legal_actions):
        self.replay_buffer.append(Experience(state, action, reward, next_state, next_legal_actions))

    # Sample batch and train network
    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough experiences yet

        # Randomly sample from replay buffer
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        # Prepare batch tensors
        states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        actions = np.zeros(self.batch_size, dtype=np.int64)
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        next_states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        in_progress = np.zeros(self.batch_size, dtype=np.float32)
        nla_mask = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)

        # Fill tensors
        for i, experience in enumerate(minibatch):
            states[i] = experience.state
            actions[i] = experience.action
            rewards[i] = experience.reward
            next_states[i] = experience.next_state

            # Process nla_mask - set very negative values for illegal actions
            for j in range(self.num_actions):
                nla_mask[i, j] = 0.0 if (experience.next_legal_actions & (1 << j)) != 0 else -1e9

            # 0 if done, 1 if still in progress
            in_progress[i] = 1.0 if experience.next_legal_actions != 0 else 0.0

        # Convert numpy arrays to torch tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(next_states)
        in_progress_tensor = torch.FloatTensor(in_progress).unsqueeze(1)
        nla_mask_tensor = torch.FloatTensor(nla_mask)

        # Compute current Q values
        current_q_values = self.primary_network(states_tensor).gather(1, actions_tensor)

        # Compute target Q values
        with torch.no_grad():
            next_q_predictions = self.target_network(next_states_tensor)
            # Apply nla_mask by adding it (illegal actions become very negative)
            masked_next_q_values = next_q_predictions + nla_mask_tensor
            next_q_values = masked_next_q_values.max(1, keepdim=True)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values * in_progress_tensor

        # Compute loss and update weights
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        # Update target network if needed
        self.training_steps += 1
        if self.training_steps % self.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.primary_network.state_dict())

    # Train the agent
    def train(self, num_episodes, print_freq, sample_freq):
        total_reward = 0.0
        total_steps = 0

        for episode in range(num_episodes):
            self.env.reset()
            episode_reward = 0.0
            steps = 0
            done = False

            # Get initial state
            state = self.env.get_state()
            legal_actions = self.env.get_legal_actions()

            if legal_actions == 0:
                raise RuntimeError("unexpected: no actions in initial state")

            while not done:
                # Choose and take action
                action = self.choose_action(state, legal_actions)
                reward = self.env.take_action(action)
                next_state = self.env.get_state()

                # Check if game is over
                legal_actions = self.env.get_legal_actions()
                done = (legal_actions == 0)

                # Store experience
                self.remember(state, action, reward, next_state, legal_actions)

                # Train network
                self.replay()

                # Update state
                state = next_state
                episode_reward += reward
                steps += 1
                total_steps += 1

            total_reward += episode_reward

            # Save episode stats for plotting
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)
            self.episode_epsilon.append(self.epsilon)
            self.episode_predefined_prob.append(self.predefined_strategy_probability if self.predefined_strategy else 0.0)

            # Print episode stats
            if (episode + 1) % print_freq == 0:
                avg_reward = total_reward / print_freq
                print(f"Episode: {episode + 1:,}, "
                      f"Avg steps: {total_steps / print_freq:.1f}, "
                      f"Epsilon: {self.epsilon:.4f}, ", end="")

                if self.predefined_strategy and self.predefined_strategy_probability > 0.0:
                    print(f"Predefined: {self.predefined_strategy_probability:.4f}, ", end="")

                print(f"Avg reward: {avg_reward:.4f}")

                total_reward = 0.0
                total_steps = 0

            # Run sample games periodically
            if (episode + 1) % sample_freq == 0:
                sample_reward, action_counts = self.run_sample_games(500)
                self.sample_rewards.append(sample_reward)
                self.action_distributions.append((episode, action_counts))

            # Sample Q values periodically
            if (episode + 1) % self.q_value_sample_freq == 0:
                q_stats = self.sample_q_value_distribution()
                if q_stats:
                    self.q_value_stats.append((episode + 1, q_stats))

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Decay predefined strategy probability after each episode (no min for now)
            self.predefined_strategy_probability *= self.predefined_strategy_decay

    # Sample Q values from the replay buffer
    def sample_q_value_distribution(self):
        """Sample Q-values from current replay buffer to track their distribution"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample states from replay buffer
        sample = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([exp.state for exp in sample])

        # Get Q-values for all actions
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            q_values = self.primary_network(states_tensor).numpy()

        # Compute statistics
        stats = {
            'mean': float(np.mean(q_values)),
            'std': float(np.std(q_values)),
            'min': float(np.min(q_values)),
            'max': float(np.max(q_values)),
            'median': float(np.median(q_values)),
            'q1': float(np.percentile(q_values, 25)),  # 1st quartile
            'q3': float(np.percentile(q_values, 75))   # 3rd quartile
        }

        return stats

    # Run multiple sample games and track actions taken.
    # Returns average reward and a dictionary of action counts.
    def run_sample_games(self, num_games):
        action_counts = {}
        total_actions = 0
        print("\n----- SAMPLE GAMES -----")

        # Set epsilon to 0 for deterministic policy during sample
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        original_predefined_strategy_probability = self.predefined_strategy_probability
        self.predefined_strategy_probability = 0.0

        game_rewards = []

        for _ in range(num_games):
            self.env.reset()
            game_reward = 0.0

            while True:
                legal_actions = self.env.get_legal_actions()

                # Check if game is over
                if legal_actions == 0:
                    break

                # Choose action
                state = self.env.get_state()
                action = self.choose_action(state, legal_actions)

                # Track action
                action_str = self.env.get_action_name(action)
                if action_str in action_counts:
                    action_counts[action_str] += 1
                else:
                    action_counts[action_str] = 1

                # Take action
                reward = self.env.take_action(action)
                game_reward += reward
                total_actions += 1

            # Track reward for each game
            game_rewards.append(game_reward)

        # Calculate statistics
        game_rewards_array = np.array(game_rewards)
        avg_reward = np.mean(game_rewards_array)
        min_reward = np.min(game_rewards_array)
        max_reward = np.max(game_rewards_array)
        std_dev = np.std(game_rewards_array)

        # Print results
        print(f"Average score: {avg_reward:.2f}, Stdev: {std_dev:.2f}, Min: {min_reward:.0f}, Max: {max_reward:.0f}")
        print(f"Average steps taken: {total_actions / num_games:.1f}")

        # Print the actions taken (and average number of times for each)
        print("Actions taken:")

        # Group actions by type for more organized output
        grouped_actions = {
            "Buy Cards": {},
            "Gain Cards": {},
            "Play Cards": {},
            "Trash Cards": {},
            "Other Actions": {}
        }

        for action_str, count in action_counts.items():
            if action_str.startswith("Play "):
                grouped_actions["Play Cards"][action_str] = count
            elif action_str.startswith("Gain "):
                grouped_actions["Gain Cards"][action_str] = count
            elif action_str.startswith("Buy "):
                grouped_actions["Buy Cards"][action_str] = count
            elif action_str.startswith("Trash "):
                grouped_actions["Trash Cards"][action_str] = count
            else:
                grouped_actions["Other Actions"][action_str] = count

        # Print grouped actions
        groups = ["Buy Cards", "Gain Cards", "Play Cards", "Trash Cards", "Other Actions"]

        for group in groups:
            print(f"  {group}:")

            # Create a list of pairs for sorting
            sorted_actions = sorted(
                grouped_actions[group].items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Print sorted actions
            for action_str, count in sorted_actions:
                print(f"    {action_str:<20}: {count / num_games:.2f}")

        print("----- END SAMPLE -----\n")

        # Restore original epsilon
        self.epsilon = original_epsilon
        self.predefined_strategy_probability = original_predefined_strategy_probability

        # Return average reward for the sample
        average_action_counts = {k: v / num_games for k, v in action_counts.items()}
        return avg_reward, average_action_counts

    # Save model weights
    def save_model(self, filename):
        torch.save(self.primary_network.state_dict(), filename)
        print(f"Model saved to {filename}")

    # Load model weights
    def load_model(self, filename):
        self.primary_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.primary_network.state_dict())
        print(f"Model loaded from {filename}")

    def plot_graphs(self, save_path=None):
        """
        Plot training metrics:
        1. Episode rewards, 100-episode moving average, and sample rewards
        2. Average steps per episode
        3. Epsilon and predefined strategy probability over time
        4. Q-value statistics

        Args:
            save_path (str, optional): Directory to save the plots. If None, plots are displayed only.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import MaxNLocator
        from matplotlib import colormaps

        # Create figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Rewards
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, label='Episode Rewards')

        # Calculate and plot moving average (if we have enough episodes)
        if len(self.episode_rewards) >= 100:
            moving_avg = []
            for i in range(len(self.episode_rewards) - 99):
                moving_avg.append(np.mean(self.episode_rewards[i:i+100]))
            ax1.plot(np.arange(100, len(self.episode_rewards) + 1), moving_avg, 'r-',
                     label='100-Episode Moving Average')

        # Plot sample rewards if available
        if len(self.sample_rewards) > 0:
            # Calculate at which episodes the samples were taken
            sample_freq = len(self.episode_rewards) // len(self.sample_rewards)
            sample_episodes = np.arange(sample_freq, len(self.episode_rewards) + 1, sample_freq)
            ax1.plot(sample_episodes, self.sample_rewards, 'go-', label='Sample Game Rewards')

        ax1.set_title('Rewards over Training')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot 2: Action distributions as a stacked graph
        if len(self.action_distributions) > 0:
            # Find all unique actions across all samples
            all_actions = set()
            for _, dist in self.action_distributions:
                all_actions.update(dist.keys())

            # Find the top N most common actions across all samples
            action_total_freqs = {}
            for _, dist in self.action_distributions:
                for action, freq in dist.items():
                    if action in action_total_freqs:
                        action_total_freqs[action] += freq
                    else:
                        action_total_freqs[action] = freq

            # Limit to top 15 most common actions
            MAX_ACTIONS = 15
            top_actions = sorted(all_actions,
                                 key=lambda x: action_total_freqs.get(x, 0),
                                 reverse=True)[:MAX_ACTIONS]

            # Get the remaining actions for "Other" category
            other_actions = all_actions - set(top_actions)

            # Prepare data for plotting
            sample_episodes = [ep for ep, _ in self.action_distributions]
            action_values = []
            action_labels = top_actions.copy()

            # Extract frequencies for each top action at each sampled episode
            for action in top_actions:
                action_freq = []
                for _, dist in self.action_distributions:
                    action_freq.append(dist.get(action, 0))
                action_values.append(action_freq)

            # Add "Other" category if required
            if other_actions:
                other_freq = []
                for _, dist in self.action_distributions:
                    # Sum frequencies of all non-top actions
                    other_sum = sum(dist.get(action, 0) for action in other_actions)
                    other_freq.append(other_sum)

                action_values.append(other_freq)
                action_labels.append("Other")

            # Get distinct colors for each category
            if len(action_labels) <= 9:
                cmap = colormaps['Set1']
            else:
                cmap = colormaps['tab20']
            colors = [cmap(i) for i in range(len(action_labels))]

            # Create stacked plot
            ax2.stackplot(sample_episodes, action_values, labels=action_labels, alpha=0.8, colors=colors)
            ax2.set_title('Action Distribution Over Training')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Frequency')
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax2.grid(True, alpha=0.3)

        # Plot 3: Epsilon and Predefined Strategy Probability
        ax3.plot(episodes, self.episode_epsilon, 'r-', label='Epsilon')
        if len(self.episode_predefined_prob) > 0 and self.predefined_strategy:
            ax3.plot(episodes, self.episode_predefined_prob, 'b-', label='Predefined Strategy Probability')
        ax3.set_title('Exploration Parameters')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Parameter Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot 4: Q-value statistics
        if len(self.q_value_stats) > 0:
            # Extract data
            episodes = [stat[0] for stat in self.q_value_stats]
            means = [stat[1]['mean'] for stat in self.q_value_stats]
            stds = [stat[1]['std'] for stat in self.q_value_stats]
            mins = [stat[1]['min'] for stat in self.q_value_stats]
            maxs = [stat[1]['max'] for stat in self.q_value_stats]
            medians = [stat[1]['median'] for stat in self.q_value_stats]
            q1s = [stat[1]['q1'] for stat in self.q_value_stats]
            q3s = [stat[1]['q3'] for stat in self.q_value_stats]

            ax4.plot(episodes, means, 'b-', label='Mean Q-value')
            ax4.fill_between(episodes,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color='b', label='±1 Std Dev')
            ax4.plot(episodes, mins, 'r--', alpha=0.5, label='Min')
            ax4.plot(episodes, maxs, 'g--', alpha=0.5, label='Max')
            ax4.set_title('Q-value Statistics Over Time')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Q-value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            # Create timestamp for unique filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_path}/training_plots_{timestamp}.png"
            plt.savefig(filename)
            print(f"Plots saved to {filename}")

        # Display the figure
        plt.show()

