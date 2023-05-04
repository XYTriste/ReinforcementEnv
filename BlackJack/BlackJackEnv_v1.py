"""
### Description
    Card Values:

    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.

    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.

    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    ### Action Space
    There are two actions: stick (0), and hit (1).

    ### Observation Space
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).

    ### Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
"""
from typing import Optional, Union, List, Tuple

import gym
from gym.core import RenderFrame, ActType, ObsType
from gym import spaces

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def cmp(a, b):
    return float(a > b) - float(a < b)


def draw_card(np_random):
    return int(np_random.choice(deck))


def sum_hand(hand):
    return sum(hand)


def is_bust(hand):
    return sum_hand(hand) > 21


def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):
    return sorted(hand) == [1, 10]


def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21


class BlackJackEnvironment(gym.Env):

    def __init__(self):
        """
        observation_space包含状态、庄家初始点数、是否可用ACE的信息
        """
        self.player = None
        self.dealer = None
        self.playerHandCard = None
        self.dealerHandCard = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        if action:
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:
            terminated = True
            while sum(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            # if is_natural(self.player) and not is_natural(self.dealer):
            #     reward = 1.5

        return self._get_observation(), reward, terminated, False, {}

    def _get_observation(self):
        return sum_hand(self.player), self.dealer[0], usable_ace(self.player)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.dealer = [draw_card(self.np_random)]
        self.player = [draw_card(self.np_random)]

        return self._get_observation(), {}