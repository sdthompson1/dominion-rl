from dataclasses import dataclass
from collections import defaultdict

from dominion import Phase, CardId
from strategy import Strategy

@dataclass
class BuyMenuItem:
    card: CardId
    num_to_buy: int


class BuyMenuStrategy(Strategy):
    """Strategy that follows a predefined buy menu."""

    def __init__(self, buy_menu):
        """
        Initialize with a buy menu.

        Args:
            buy_menu: List of BuyMenuItem objects
        """
        self.menu = buy_menu

    def choose_action(self, env, legal_actions):
        # A simple buy menu implementation

        if env.phase == Phase.Action:
            # In action phase, use heuristics to play action cards

            # Priority order for action cards (higher value = higher priority)
            action_priority = {
                CardId.ScryingPool: 11,  # Gets us more action cards
                CardId.Laboratory: 10,   # +2 cards, +1 action (strictly better than others)
                CardId.Alchemist: 9,    # +2 cards, +1 action
                CardId.University: 8,   # +2 actions and we can gain more action cards
                CardId.Bazaar: 7,       # +1 card, +2 actions, +1 coin
                CardId.Village: 6,      # +1 card, +2 actions (action generator)
                CardId.Market: 5,       # +1 card, +1 action, +1 coin, +1 buy (versatile)
                CardId.Festival: 4,     # +2 actions, +1 buy, +2 coins (good economy)
                CardId.Workshop: 3,     # Terminal, Gaining cards
                CardId.Wharf: 2,        # Terminal, +Cards now and next turn
                CardId.Smithy: 1,       # Terminal, +3 cards (use after action generators)
                CardId.Chapel: 0        # Trash cards (use last, or early game)
            }

            best_card = None
            highest_priority = -1

            # Find the highest priority action card we can play
            for card, priority in action_priority.items():
                card_position = env.card_id_to_position[int(card)]
                if card_position >= 0:
                    action_number = card_position + 1

                    # Check if this action is legal
                    if (legal_actions & (1 << action_number)) != 0:
                        # We can play this card
                        if priority > highest_priority:
                            highest_priority = priority
                            best_card = card

            # If we found a card to play, do it
            if best_card is not None:
                return env.card_id_to_position[int(best_card)] + 1
            else:
                # No action cards to play, end action phase
                return 0

        elif env.phase == Phase.Trash:
            # In trash phase, use heuristics to decide what to trash

            # Priority for trashing (higher = trash first)
            trash_priority = {
                CardId.Curse: 5,    # Always trash curses first
                CardId.Estate: 4,   # Early game, trash estates
                CardId.Copper: 3,   # Thin deck by removing coppers
                CardId.Chapel: 1    # Only trash Chapel if we have multiple
                # (Other cards are not trashed)
            }

            best_to_trash = None
            highest_trash_priority = 0  # We need to find something better than priority zero

            for card, priority in trash_priority.items():
                card_position = env.card_id_to_position[int(card)]
                if card_position >= 0:
                    trash_action = card_position + 1

                    # Check if we can trash this card
                    if (legal_actions & (1 << trash_action)) != 0:
                        if priority > highest_trash_priority:
                            highest_trash_priority = priority
                            best_to_trash = card

            if best_to_trash is not None:
                return env.card_id_to_position[int(best_to_trash)] + 1
            else:
                # Nothing to trash, end trashing
                return 0

        elif env.phase == Phase.Buy or env.phase == Phase.Gain:
            # In buy or gain phase, follow the buy menu

            # Count total cards we currently have in all zones
            card_counts = defaultdict(int)

            # Count cards in all zones
            for card in env.deck + env.hand + env.discard_pile + env.played_cards + env.duration_cards:
                card_counts[card] += 1

            # Check buy menu items in order
            for item in self.menu:
                # Check if we already have enough of this card
                if card_counts[item.card] >= item.num_to_buy:
                    continue  # We already have enough, skip this item

                # Check if we can buy this card
                # Calculate its position in the action space
                card_position = env.card_id_to_position[int(item.card)]
                if card_position >= 0:
                    buy_action = card_position + 1  # +1 because action 0 is "end phase"

                    # Check if buying/gaining this card is a legal action
                    if (legal_actions & (1 << buy_action)) != 0:
                        # Legal to buy/gain this card, so choose this action
                        return buy_action

            # If no card was bought, end the buy phase
            return 0

        # Error - unknown phase
        raise RuntimeError("BuyMenuStrategy failed")
