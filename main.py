from dominion import DominionEnv, CardId
from buy_menu_strategy import BuyMenuStrategy, BuyMenuItem
from learning import DQNAgent
import os

# Main function to run DQN agent in Dominion
def main():
    # Create a kingdom with some cards
    kingdom = [
        CardId.Alchemist,
        CardId.Bazaar,
        CardId.ScryingPool,
        CardId.University,
        CardId.Vineyard,
        CardId.Wharf
    ]

    # Buy menu loosely inspired by the "Drunk Marine Students" strategy from:
    # https://github.com/Geronimoo/DominionSim/blob/master/src/main/java/be/aga/dominionSimulator/DomBots.xml
    # Note: This buy menu isn't particularly good, but the DQN agent is able to optimize it
    # into something much better!
    buy_menu = [
        # Always buy Provinces when able
        BuyMenuItem(CardId.Province, 99),

        # Get some Universities - gives +Actions and allows to gain more action cards
        BuyMenuItem(CardId.University, 3),

        # Alternate between Wharves and Scrying Pools
        BuyMenuItem(CardId.Wharf, 2),
        BuyMenuItem(CardId.ScryingPool, 2),
        BuyMenuItem(CardId.Wharf, 3),
        BuyMenuItem(CardId.ScryingPool, 3),

        # Get some Alchemists and Bazaars
        BuyMenuItem(CardId.Alchemist, 3),
        BuyMenuItem(CardId.Bazaar, 3),

        # Get Vineyards after we have a few actions in the deck
        BuyMenuItem(CardId.Vineyard, 99),

        # More Alchemists and Bazaars
        BuyMenuItem(CardId.Alchemist, 10),
        BuyMenuItem(CardId.Bazaar, 10),

        # Get up to 3 Potions
        BuyMenuItem(CardId.Potion, 3)
    ]

    buy_menu_strategy = BuyMenuStrategy(buy_menu)

    # Create Dominion environment
    env = DominionEnv(kingdom)

    # Create DQN agent
    agent = DQNAgent(
        env,

        learning_rate=1e-3,           # Learning rate
        discount_factor=0.9999,       # Discount factor (high, to encourage long action sequences)

        exploration_rate=0.25,        # Initial exploration rate
        min_exploration_rate=0.01,    # Min exploration rate
        exploration_decay=0.999,      # Exploration decay

        predefined_strategy_probability=1.0,    # Initial probability to use the predefined strategy
        predefined_strategy_decay=0.999,        # Predefined strategy probability decay
        predefined_strategy=None,     # buy_menu_strategy, or None to disable

        batch_size=64,                # Batch size
        replay_buffer_size=10000,     # Replay buffer size
        target_update_freq=1000       # Target update freq
    )

    # Uncomment to run sample games without training
    # agent.run_sample_games(2000)

    # Try to load existing model
    try:
        agent.load_model("dominion_dqn.pt")
    except FileNotFoundError:
        print("No existing model found, starting fresh training")

    # Train the agent
    print("Training DQN agent...")
    agent.train(num_episodes=4000, print_freq=100, sample_freq=500)

    # Plot graphs
    if not os.path.exists("plots"):
        os.makedirs("plots")
    agent.plot_graphs(save_path="plots")

    # Save the trained model
    agent.save_model("dominion_dqn.pt")

if __name__ == "__main__":
    main()
