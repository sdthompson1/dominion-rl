# Dominion Reinforcement Learning

A reinforcement learning implementation for a simplified version of the Dominion card game.


## Overview

This project implements:

1. A simplified, single-player version of Dominion as a reinforcement learning environment
2. A Deep Q-Network (DQN) agent that learns to play the game
3. A baseline fixed strategy ("Buy Menu" strategy) that can be used for imitation learning

The agent uses a combination of imitation learning from a predefined strategy and randomized exploration to discover optimal play patterns.

A full write-up of the project, including graphs, can be found at
https://www.solarflare.org.uk/dominion.


## Components

### Abstract base classes (environment.py, strategy.py)

- `Environment` defines a reinforcement learning environment, including states, actions and rewards
- `Strategy` defines a fixed strategy for an `Environment`
  - Fixed strategies can either be evaluated on their own, or used as starting points for training

### Dominion Environment (dominion.py)

- `DominionEnv` defines an `Environment` for single-player Dominion including:
  - Card definitions with types, costs and effects
    - Including approx 20 cards so far, from a mixture of the base set, Alchemy and Seaside
  - Game state management (deck, hand, discard pile etc.)
  - Game phases (Action and Buy phases, together with special phases for handling certain cards)
  - Game state representation for the RL agent
  - Reward calculation (the agent is rewarded for scoring victory points)
  - Game end conditions (currently the game just ends after a fixed number of turns; empty piles are ignored)

### Buy Menu Strategy (buy_menu_strategy.py)

- `BuyMenuStrategy` instantiates the `Strategy` class with a simple, priority-based approach to playing the game:
  - Follows a predefined "buy menu" for purchasing cards
  - Uses heuristics for playing action cards and trashing

### Learning Agent (learning.py)

`learning.py` defines a reinforcement learning agent that attempts to learn an optimal policy for any given `Environment`, including:

- Neural network architecture for Q-value approximation
- Experience replay buffer for stable learning
- Epsilon-greedy exploration strategy
- (Optional) Hybrid learning that starts from a predefined `Strategy` and gradually transitions to self-learned policy
- Functions for training, saving, and loading the model
- Tools for evaluating the agent and plotting graphs of training progress


## Example Usage

Run the main script to train the agent:

```
python main.py
```

The script will:
1. Initialize the environment with a specific kingdom
2. Create a DQN agent with reinforcement learning, starting from a randomized initial strategy
3. Train the agent over 4,000 games
4. Save graphs of training progress to a "plots" folder (which will be created if required)
5. Save the trained model as `dominion_dqn.pt`

In the default kingdom, the agent learns a "Big Money + Wharf" strategy, which scores around 31 points in the time available.

To use the predefined strategy instead:

1. Edit the DQNAgent creation code in `main.py`, changing `predefined_strategy=None` to `predefined_strategy=buy_menu_strategy`
2. Delete the existing `dominion_dqn.pt` file if it exists
3. Run `python main.py` again

In this case, the agent begins with an "imitation learning" phase where it learns to copy the predefined strategy, scoring around 10 points. As training progresses, the agent shifts away from the predefined strategy and instead explores freely on its own, allowing it to discover a new strategy involving University, Alchemist, Wharf and Vineyard, scoring around 40--60 points on average (although it might be necessary to train for longer than the default 4,000 episodes in order to see this).

You can also re-run `main.py` to resume training (starting from the saved `dominion_dqn.pt` model) if desired. (If doing this, it might be desirable to decrease the initial epsilon and/or predefined strategy probability, as you don't need as much exploration when restarting from a previous run.)

For further experimentation, training parameters can be adjusted by editing `main.py`, and the neural network architecture can be changed by editing the `QNetwork` class in `learning.py`. Also, of course, different combinations of kingdom cards can be tried, by modifying the list near the top of `main.py`.


## Customization

### Adding New Cards

To add new cards, you need to:
1. Add an entry to the `CardId` enum in `dominion.py`
2. Create a `CardInfo` object with appropriate properties and effects
3. Update the `CARD_INFO` dictionary with the new card
4. (Optional) Edit the code for the `DominionEnv` itself to implement special rules (see existing examples for Alchemist, Gardens and Vineyard)

### Creating Custom Strategies

You can create custom strategies by:
1. Implementing the `Strategy` abstract base class
2. Overriding the `choose_action` method to select actions based on your strategy

Alternatively, if your strategy can be expressed as a "buy menu", then passing a new buy menu to `BuyMenuStrategy` (and perhaps adjusting the action and trashing heuristics in `buy_menu_strategy.py`) might be sufficient.

Implementing a new `Strategy` subclass that can execute some of the strategies from [Geronimoo's Dominion simulator](https://github.com/Geronimoo/DominionSim) might be an interesting project.


## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib

After installing Python, the remaining requirements can be installed with `pip install torch numpy matplotlib`.


## Future Improvements

Potential areas for enhancement:
- Implement more Dominion cards and expansions
- Add a multi-player mode, including self-play for multi-agent learning
- Add more sophisticated exploration mechanisms, to reduce the need for imitation learning
- Experiment with different RL algorithms (PPO, A2C, etc.) and neural network architectures
- Improve the state representation for better learning
- Add full game logging so we can see the played strategies in more detail
- In the single-player case, experiment with mean-variance optimization (i.e. search for strategies that have high expected scores but also low variance of points scored)


## Contact Details

This project was created by Stephen Thompson.

Email: stephen (at) solarflare.org.uk

Website: https://www.solarflare.org.uk/
