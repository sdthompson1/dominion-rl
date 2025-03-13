from abc import ABC, abstractmethod

class Strategy(ABC):
    """Abstract base class representing a fixed strategy"""

    @abstractmethod
    def choose_action(self, env, legal_actions) -> int:
        """
        Choose an action based on the current environment state.

        Args:
            env: The game environment
            legal_actions: Bitmask of legal actions

        Returns:
            int: The chosen action
        """
        pass
