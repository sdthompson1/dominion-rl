from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional

class Environment(ABC):
    """Abstract base class for environments to be used with the DQN agent."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment to its initial state."""
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the environment.

        Returns:
            np.ndarray: A numpy array representing the state
        """
        pass

    @abstractmethod
    def get_legal_actions(self) -> int:
        """
        Get a bitmask of legal actions from the current state.

        Returns:
            int: A bitmask where the bit at position i is 1 if action i is legal
        """
        pass

    @abstractmethod
    def take_action(self, action_int: int) -> float:
        """
        Take an action in the environment.

        Args:
            action_int: The action to take

        Returns:
            float: The reward received for taking the action
        """
        pass

    @abstractmethod
    def get_action_name(self, action_int: int) -> str:
        """
        Get a human-readable name for an action.

        Args:
            action_int: The action ID

        Returns:
            str: A human-readable name for the action
        """
        pass

    @property
    @abstractmethod
    def state_size(self) -> int:
        """Get the size of the state vector."""
        pass

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Get the number of possible actions."""
        pass
