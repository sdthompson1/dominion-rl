from enum import IntEnum
import random
import numpy as np
from typing import List, Dict, Tuple, Callable, Set, Optional, Any
from dataclasses import dataclass
import copy

from environment import Environment

# Card IDs
class CardId(IntEnum):
    Copper = 0
    Silver = 1
    Gold = 2
    Potion = 3

    Estate = 4
    Duchy = 5
    Province = 6
    Curse = 7

    Alchemist = 8
    Bazaar = 9
    Chapel = 10
    Festival = 11
    Gardens = 12
    Laboratory = 13
    Market = 14
    ScryingPool = 15
    Smithy = 16
    University = 17
    Village = 18
    Vineyard = 19
    Wharf = 20
    Workshop = 21

    NUM_CARD_IDS = 22

# Card types (Victory, Treasure etc.)
class CardType(IntEnum):
    Victory = 1
    Treasure = 2
    Action = 4
    Curse = 8
    Duration = 16
    ALL_CARD_TYPES = 31

# Game phase enum
class Phase(IntEnum):
    Action = 1
    Buy = 2
    Trash = 3     # used for Chapel
    Gain = 4      # used for Workshop


# Number of standard cards (Copper/Silver/Gold/Potion, Estate/Duchy/Province, Curse)
NUM_STANDARD_CARDS = 8

# Max Kingdom cards allowed in a game
MAX_KINGDOM = 10

# Max cards in game is standard cards plus kingdom cards
MAX_CARDS_IN_GAME = NUM_STANDARD_CARDS + MAX_KINGDOM


# State size:
#   9 base features
#   6 zones for cards (supply, hand, deck, discard, in-play, duration)
STATE_SIZE = 9 + 6 * MAX_CARDS_IN_GAME

# Action numbers:
#   Action zero is the "Skip current phase" action
#   Actions 1 and above correspond to CardIds
NUM_ACTIONS = MAX_CARDS_IN_GAME + 1


# Card Effects
CardEffect = Callable[['DominionEnv'], None]

# +n Actions
def plus_actions(n: int) -> CardEffect:
    def effect(env: 'DominionEnv') -> None:
        env.actions += n
    return effect

# +n Cards
def plus_cards(n: int) -> CardEffect:
    def effect(env: 'DominionEnv') -> None:
        env.draw_cards(n)
    return effect

# +n Coins
def plus_coins(n: int) -> CardEffect:
    def effect(env: 'DominionEnv') -> None:
        env.coins += n
    return effect

# +n Potions
def plus_potions(n: int) -> CardEffect:
    def effect(env: 'DominionEnv') -> None:
        env.potions += n
    return effect

# +n Buys
def plus_buys(n: int) -> CardEffect:
    def effect(env: 'DominionEnv') -> None:
        env.buys += n
    return effect

# Trash upto n cards from your hand.
# (Enters Phase.Trash so the agent can choose which card(s) to trash.)
def trash_upto_from_hand(n: int) -> CardEffect:
    def effect(env: 'DominionEnv') -> None:
        env.trashes = n
        env.phase = Phase.Trash
    return effect

# You may gain a card costing upto n.
# It must match one of the CardTypes in the mask.
# (Enters Phase.Gain so the agent can choose which card to gain.)
def gain_card_costing_upto(n, mask) -> CardEffect:
    def effect(env: 'DominionEnv') -> None:
        env.phase = Phase.Gain
        env.gain_max_cost = n
        env.gain_types = mask
    return effect

# Scrying pool effect:
# (This is slightly simplified compared to the real card.)
# Reveal the top card of the deck; if it is not an Action, discard it.
# Then, reveal cards from deck until a non-Action is revealed; put all revealed cards into hand.
def scrying_pool() -> CardEffect:
    def effect(env: 'DominionEnv') -> None:
        # If top card is not an action, discard it.
        if env.deck:
            top_card = env.deck[-1]
            if not (CARD_INFO[top_card].types & int(CardType.Action)):
                env.deck.pop()
                env.discard_pile.append(top_card)

        # Keep drawing cards into hand (unless both deck and discard pile are empty)
        while env.deck or env.discard_pile:
            # Draw 1 card
            env.draw_cards(1)
            # If the drawn card (now at env.hand[-1]) wasn't an action, then stop.
            drawn_card = env.hand[-1]
            if not (CARD_INFO[drawn_card].types & int(CardType.Action)):
                break
    return effect

# Allows multiple CardEffects to be executed in sequence.
def sequence(funcs: List[CardEffect]) -> CardEffect:
    def effect(env: 'DominionEnv') -> None:
        for func in funcs:
            func(env)
    return effect


# Card information structure
@dataclass
class CardInfo:
    coin_cost: int     # Number of coins required to buy this card
    potion_cost: int   # Number of potions required to buy this card
    supply: int        # Initial number of this card in the Supply
    types: int         # Bitmask of CardType values
    points: int        # Victory points for holding this card (0 for non-victory cards)
    on_action: Optional[CardEffect] = None         # What happens when you play this as an Action
    next_turn_effect: Optional[CardEffect] = None  # What happens in the 2nd turn (for Duration cards)
    on_treasure: Optional[CardEffect] = None       # What happens when you play this as a Treasure


# Card database
CARD_INFO = {
    CardId.Copper: CardInfo(0, 0, 60, int(CardType.Treasure), 0, None, None, plus_coins(1)),
    CardId.Silver: CardInfo(3, 0, 40, int(CardType.Treasure), 0, None, None, plus_coins(2)),
    CardId.Gold:   CardInfo(6, 0, 30, int(CardType.Treasure), 0, None, None, plus_coins(3)),
    CardId.Potion: CardInfo(4, 0, 16, int(CardType.Treasure), 0, None, None, plus_potions(1)),

    CardId.Estate:   CardInfo(2, 0, 12, int(CardType.Victory), 1),
    CardId.Duchy:    CardInfo(5, 0, 12, int(CardType.Victory), 3),
    CardId.Province: CardInfo(8, 0, 12, int(CardType.Victory), 6),
    CardId.Curse:    CardInfo(0, 0, 20, int(CardType.Curse),  -1),

    # The cleanup effect of Alchemist is hard-coded in end_buy_phase
    CardId.Alchemist: CardInfo(3, 1, 10, int(CardType.Action), 0,
                               sequence([plus_cards(2), plus_actions(1)])),

    CardId.Bazaar: CardInfo(5, 0, 10, int(CardType.Action), 0,
                            sequence([plus_cards(1), plus_actions(2), plus_coins(1)])),

    CardId.Chapel: CardInfo(2, 0, 10, int(CardType.Action), 0,
                            trash_upto_from_hand(4)),

    CardId.Festival: CardInfo(5, 0, 10, int(CardType.Action), 0,
                              sequence([plus_actions(2), plus_buys(1), plus_coins(2)])),

    # Victory points for Gardens are hard-coded in calculate_score
    CardId.Gardens: CardInfo(4, 0, 10, int(CardType.Victory), 0),

    CardId.Laboratory: CardInfo(5, 0, 10, int(CardType.Action), 0,
                                sequence([plus_cards(2), plus_actions(1)])),

    CardId.Market: CardInfo(5, 0, 10, int(CardType.Action), 0,
                            sequence([plus_cards(1), plus_actions(1), plus_buys(1), plus_coins(1)])),

    CardId.ScryingPool: CardInfo(2, 1, 10, int(CardType.Action), 0, scrying_pool()),

    CardId.Smithy: CardInfo(4, 0, 10, int(CardType.Action), 0, plus_cards(3)),

    CardId.University: CardInfo(2, 1, 10, int(CardType.Action), 0,
                                sequence([plus_actions(2),
                                          gain_card_costing_upto(5, CardType.Action)])),

    CardId.Village: CardInfo(3, 0, 10, int(CardType.Action), 0,
                            sequence([plus_cards(1), plus_actions(2)])),

    # Victory points for Vineyard are hard-coded in calculate_score
    CardId.Vineyard: CardInfo(0, 1, 10, int(CardType.Victory), 0),

    CardId.Wharf: CardInfo(5, 0, 10, int(CardType.Action) | int(CardType.Duration), 0,
                          sequence([plus_cards(2), plus_buys(1)]),   # first turn
                          sequence([plus_cards(2), plus_buys(1)])),  # subsequent turn

    CardId.Workshop: CardInfo(3, 0, 10, int(CardType.Action), 0,
                             gain_card_costing_upto(4, CardType.ALL_CARD_TYPES))
}


# Dominion Environment
class DominionEnv(Environment):
    def __init__(self, kingdom: List[CardId]):
        if len(kingdom) > MAX_KINGDOM:
            raise RuntimeError("Kingdom too big")

        self.kingdom = kingdom
        self.reset()

    @property
    def state_size(self) -> int:
        return STATE_SIZE

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    # Reset the environment for a new game
    def reset(self):
        # Setup initial player state
        self.deck: List[CardId] = []
        self.hand: List[CardId] = []
        self.discard_pile: List[CardId] = []
        self.played_cards: List[CardId] = []
        self.duration_cards: List[CardId] = []

        # Initial deck: 7 Coppers and 3 Estates
        self.deck.extend([CardId.Copper] * 7)
        self.deck.extend([CardId.Estate] * 3)

        # Setup initial turn
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.potions = 0
        self.trashes = 0
        self.gain_max_cost = 0
        self.gain_types = 0

        self.phase = Phase.Action

        self.turn = 1
        self.max_turn = 15

        # Setup the supply
        self.supply: Dict[CardId, int] = {}

        # Standard cards
        standard_cards = [
            CardId.Copper,
            CardId.Silver,
            CardId.Gold,
            CardId.Potion,
            CardId.Curse,
            CardId.Estate,
            CardId.Duchy,
            CardId.Province
        ]

        # Initialize supply
        for card in self.kingdom:
            self.supply[card] = CARD_INFO[card].supply

        for card in standard_cards:
            self.supply[card] = CARD_INFO[card].supply

        # Map card IDs to positions in the state vector
        self.card_id_to_position = [-1] * int(CardId.NUM_CARD_IDS)
        self.position_to_card_id = []
        self.num_cards_in_game = 0

        for card in range(int(CardId.NUM_CARD_IDS)):
            if CardId(card) in self.supply:
                # This card is being used in the game
                self.card_id_to_position[card] = self.num_cards_in_game
                self.position_to_card_id.append(CardId(card))
                self.num_cards_in_game += 1

        if self.num_cards_in_game != len(self.kingdom) + NUM_STANDARD_CARDS:
            # Something went wrong
            raise RuntimeError("unexpected num_cards_in_game")

        # Shuffle and deal initial hand
        self.shuffle_deck()
        self.draw_cards(5)

        # End action phase immediately because there are no actions
        # to play on the first turn
        self.end_action_phase()

    # Shuffle the deck
    def shuffle_deck(self):
        random.shuffle(self.deck)

    # Draw a number of cards
    def draw_cards(self, count: int):
        for _ in range(count):
            if not self.deck:
                # If deck is empty, shuffle discards to create a new deck
                if not self.discard_pile:
                    return  # No cards left to draw
                self.deck = self.discard_pile  # Put discard pile in deck
                self.discard_pile = []         # Discard pile is now empty
                self.shuffle_deck()            # Shuffle the deck

            # Draw one card from deck, add it to hand
            card = self.deck.pop()
            self.hand.append(card)

    # Play an action card
    def play_action_card(self, card_id: CardId) -> bool:
        if self.phase != Phase.Action:
            # Not in action phase
            return False

        if self.actions <= 0:
            # No actions left
            return False

        if card_id not in self.hand:
            # We do not have this card in hand
            return False

        info = CARD_INFO[card_id]

        if not (info.types & int(CardType.Action)):
            # It's not an action card
            return False

        # Remove from hand and add to played cards
        self.hand.remove(card_id)
        self.played_cards.append(card_id)

        # Use up an action
        self.actions -= 1

        # Apply card effects
        if info.on_action:
            info.on_action(self)

        return True

    # End action phase
    def end_action_phase(self) -> bool:
        if self.phase != Phase.Action:
            # Not in action phase
            return False

        # Enter buy phase
        self.phase = Phase.Buy

        # Play all treasures automatically
        i = 0
        while i < len(self.hand):
            card_id = self.hand[i]
            info = CARD_INFO[card_id]

            if info.types & int(CardType.Treasure):
                # Remove from hand and add to played cards
                del self.hand[i]
                self.played_cards.append(card_id)

                # Apply card effects
                if info.on_treasure:
                    info.on_treasure(self)
            else:
                i += 1

        return True

    # Buy a card
    def buy_card(self, card_id: CardId) -> bool:
        if self.phase != Phase.Buy:
            # Not in buy phase
            return False

        info = CARD_INFO[card_id]

        if self.coins < info.coin_cost or self.potions < info.potion_cost:
            # Not enough coins or potions
            return False

        if self.buys <= 0:
            # Not enough buys
            return False

        if self.supply.get(card_id, 0) <= 0:
            # Out of stock
            return False

        # Buy the card (add to discard pile)
        self.buys -= 1
        self.coins -= info.coin_cost
        self.potions -= info.potion_cost
        self.supply[card_id] -= 1
        self.discard_pile.append(card_id)

        return True

    # End buy phase
    def end_buy_phase(self) -> bool:
        if self.phase != Phase.Buy:
            # Not in buy phase
            return False

        # Alchemist effect: At start of Clean-up, if you have a Potion in play, you
        # may put the Alchemist onto your deck. (We assume the player always does
        # this if possible.)
        if CardId.Alchemist in self.played_cards and CardId.Potion in self.played_cards:
            alchemists = [c for c in self.played_cards if c == CardId.Alchemist]
            non_alchemists = [c for c in self.played_cards if c != CardId.Alchemist]
            # Using deck.extend is like putting cards on top of deck, because
            # draw_cards takes cards from the back of the list (using pop())
            self.deck.extend(alchemists)
            self.played_cards = non_alchemists

        # Discard previous duration cards
        self.discard_pile.extend(self.duration_cards)
        self.duration_cards = []

        # Move new duration cards from played_cards to duration_cards
        i = 0
        while i < len(self.played_cards):
            info = CARD_INFO[self.played_cards[i]]
            if info.types & int(CardType.Duration):
                self.duration_cards.append(self.played_cards[i])
                del self.played_cards[i]
            else:
                i += 1

        # Discard hand and any remaining played cards
        self.discard_pile.extend(self.hand)
        self.discard_pile.extend(self.played_cards)
        self.hand = []
        self.played_cards = []

        # Draw a new hand
        self.draw_cards(5)

        # Go into a new action phase
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.potions = 0
        self.phase = Phase.Action

        # Increase turn count (we are starting a new turn)
        self.turn += 1

        # Apply the next_turn_effect of any duration_cards
        for card in self.duration_cards:
            info = CARD_INFO[card]
            if info.next_turn_effect:
                info.next_turn_effect(self)

        return True

    # Trash a card
    def trash_card(self, card: CardId) -> bool:
        if self.phase != Phase.Trash:
            # Not in trashing phase
            return False

        if self.trashes <= 0:
            # No trashes remaining
            return False

        if card not in self.hand:
            # We don't have this card in hand
            return False

        # Trash the card from hand
        # (We don't currently maintain a separate trash pile -- it is just gone for good.)
        self.trashes -= 1
        self.hand.remove(card)

        return True

    # End trash phase, go back to regular action phase
    def end_trash_phase(self) -> bool:
        if self.phase != Phase.Trash:
            # Not in trashing phase
            return False

        # Reset trashes and return to action phase
        self.trashes = 0
        self.phase = Phase.Action

        return True

    # Gain a card and go back to Action phase
    # (This is used for Workshop, University)
    def gain_card(self, card_id: CardId) -> bool:
        if self.phase != Phase.Gain:
            # Not in "Gain" phase
            return False

        info = CARD_INFO[card_id]

        if info.coin_cost > self.gain_max_cost or info.potion_cost > 0:
            # Card costs more than the limit (or requires potions - cards requiring potions
            # cannot be gained this way)
            return False

        if not (info.types & self.gain_types):
            # Card is not one of the allowed types for gaining
            return False

        if self.supply.get(card_id, 0) <= 0:
            # Out of stock
            return False

        # Gain the card
        self.supply[card_id] -= 1
        self.discard_pile.append(card_id)

        # Only one gain is allowed currently (there is no "num_gains")
        # so go straight back to Action phase
        self.gain_max_cost = 0
        self.gain_types = 0
        self.phase = Phase.Action

        return True

    # End gain phase (without gaining anything) and go back to regular action phase
    def end_gain_phase(self) -> bool:
        if self.phase != Phase.Gain:
            # We are not currently in Gain phase
            return False

        self.gain_max_cost = 0
        self.gain_types = 0
        self.phase = Phase.Action

        return True

    # Calculate current score (victory points)
    def calculate_score(self) -> int:
        score = 0
        num_gardens = 0
        num_vineyards = 0
        num_actions = 0
        num_cards = 0

        # Function to add score for a list of cards
        def add_score(cards: List[CardId]) -> None:
            nonlocal score, num_gardens, num_vineyards, num_actions, num_cards
            for card_id in cards:
                info = CARD_INFO[card_id]
                score += info.points

                num_cards += 1

                if card_id == CardId.Gardens:
                    num_gardens += 1

                if card_id == CardId.Vineyard:
                    num_vineyards += 1

                if info.types & int(CardType.Action):
                    num_actions += 1

        # Add up scores from all zones
        add_score(self.hand)
        add_score(self.deck)
        add_score(self.discard_pile)
        add_score(self.played_cards)
        add_score(self.duration_cards)

        # Add points for Gardens (10 VP per card, rounding down)
        score += (num_cards // 10) * num_gardens

        # Add points for Vineyards (1 VP per 3 Action cards, rounding down)
        score += (num_actions // 3) * num_vineyards

        return score

    # Get legal actions
    def get_legal_actions(self) -> int:
        if self.turn > self.max_turn:
            # Game has ended
            return 0

        # We can always "end the current phase" (action 0)
        legal_actions = 1  # bit 0 is set for "end phase" action

        if self.phase == Phase.Action:
            # Action phase

            # If we have actions, we can play any action card in hand
            if self.actions > 0:
                for card_id in self.hand:
                    info = CARD_INFO[card_id]
                    if info.types & int(CardType.Action):
                        # Action pos+1 corresponds to playing card at position pos
                        # (position = card_id_to_position[card_id])
                        legal_actions |= (1 << (self.card_id_to_position[int(card_id)] + 1))

        elif self.phase == Phase.Buy:
            # Buy phase

            # If we have buys, we can buy any available card that we can afford
            if self.buys > 0:
                for card_id, count in self.supply.items():
                    if count > 0:
                        info = CARD_INFO[card_id]
                        if self.coins >= info.coin_cost and self.potions >= info.potion_cost:
                            # Action pos+1 corresponds to buying card at position pos
                            legal_actions |= (1 << (self.card_id_to_position[int(card_id)] + 1))

        elif self.phase == Phase.Trash:
            # Trashing phase
            # (sub-phase of action phase during Chapel play)

            # If we have trashes, we can trash any card we have in hand
            if self.trashes > 0:
                for card_id in self.hand:
                    legal_actions |= (1 << (self.card_id_to_position[int(card_id)] + 1))

        elif self.phase == Phase.Gain:
            # Gain phase (Workshop, University etc.)
            # We can gain any card in the supply costing up to self.gain_max_cost,
            # but it must match one of the self.gain_types
            for card_id, count in self.supply.items():
                if count > 0:
                    info = CARD_INFO[card_id]
                    if info.coin_cost <= self.gain_max_cost and info.potion_cost == 0:
                        if info.types & self.gain_types:
                            legal_actions |= (1 << (self.card_id_to_position[int(card_id)] + 1))

        return legal_actions

    # Take an action
    def take_action(self, action_int: int) -> float:
        # Calculate score before the action
        old_score = self.calculate_score()

        # Execute the action:
        result = False

        if self.phase == Phase.Action:
            if action_int == 0:
                # End current phase
                result = self.end_action_phase()
            elif 1 <= action_int <= self.num_cards_in_game:
                # Play a card
                # Note: action_int - 1 is the card "position"; we must convert this to a CardId
                result = self.play_action_card(self.position_to_card_id[action_int - 1])

        elif self.phase == Phase.Buy:
            if action_int == 0:
                # End current phase
                result = self.end_buy_phase()
            elif 1 <= action_int <= self.num_cards_in_game:
                # Buy a card
                result = self.buy_card(self.position_to_card_id[action_int - 1])

        elif self.phase == Phase.Trash:
            if action_int == 0:
                # End current phase
                result = self.end_trash_phase()
            elif 1 <= action_int <= self.num_cards_in_game:
                # Trash a card
                result = self.trash_card(self.position_to_card_id[action_int - 1])

        elif self.phase == Phase.Gain:
            if action_int == 0:
                result = self.end_gain_phase()
            elif 1 <= action_int <= self.num_cards_in_game:
                result = self.gain_card(self.position_to_card_id[action_int - 1])

        if not result:
            raise RuntimeError("Action failed")

        # If there is exactly one action from the new state then
        # take it (ignore the reward as we will recompute below)
        new_actions = self.get_legal_actions()
        if new_actions != 0 and (new_actions & (new_actions - 1)) == 0:
            # Only one bit is set in new_actions
            self.take_action(new_actions.bit_length() - 1)  # Get the position of the only set bit

        # Calculate score after the action(s)
        new_score = self.calculate_score()

        # Reward is score difference
        reward = float(new_score - old_score)

        return reward

    # Get state as a vector of floats for RL agent
    def get_state(self) -> np.ndarray:
        state = np.zeros(STATE_SIZE, dtype=np.float32)

        # Game phase and turn number
        state[0] = float(self.phase)
        state[1] = float(self.turn) / float(self.max_turn)

        # Current resources
        state[2] = self.actions / 5.0
        state[3] = self.buys / 3.0
        state[4] = self.coins / 10.0
        state[5] = self.potions / 2.0
        state[6] = self.trashes / 4.0
        state[7] = self.gain_max_cost / 5.0
        state[8] = self.gain_types / float(CardType.ALL_CARD_TYPES)

        base_pos = 9

        # Count cards in each zone
        for card in self.hand:
            state[base_pos + self.card_id_to_position[int(card)]] += 0.2

        base_pos += self.num_cards_in_game

        for card in self.deck:
            state[base_pos + self.card_id_to_position[int(card)]] += 0.1

        base_pos += self.num_cards_in_game

        for card in self.discard_pile:
            state[base_pos + self.card_id_to_position[int(card)]] += 0.1

        base_pos += self.num_cards_in_game

        for card in self.played_cards:
            state[base_pos + self.card_id_to_position[int(card)]] += 0.2

        base_pos += self.num_cards_in_game

        for card in self.duration_cards:
            state[base_pos + self.card_id_to_position[int(card)]] += 0.2

        base_pos += self.num_cards_in_game

        for card_id, count in self.supply.items():
            supply_scale = CARD_INFO[card_id].supply
            if supply_scale == 0:
                supply_scale = 10  # prevent division by zero
            current_supply = count
            state[base_pos + self.card_id_to_position[int(card_id)]] += float(current_supply) / float(supply_scale)

        return state

    # Get human-readable action names
    def get_action_name(self, action_int: int) -> str:
        if self.phase == Phase.Action:
            if action_int == 0:
                return "End Action Phase"
            elif 1 <= action_int <= self.num_cards_in_game:
                return f"Play {self.position_to_card_id[action_int - 1].name}"

        elif self.phase == Phase.Buy:
            if action_int == 0:
                return "End Buy Phase"
            elif 1 <= action_int <= self.num_cards_in_game:
                return f"Buy {self.position_to_card_id[action_int - 1].name}"

        elif self.phase == Phase.Trash:
            if action_int == 0:
                return "Stop Trashing"
            elif 1 <= action_int <= self.num_cards_in_game:
                return f"Trash {self.position_to_card_id[action_int - 1].name}"

        elif self.phase == Phase.Gain:
            if action_int == 0:
                return "End Gain Phase"
            elif 1 <= action_int <= self.num_cards_in_game:
                return f"Gain {self.position_to_card_id[action_int - 1].name}"

        return "UNKNOWN"
