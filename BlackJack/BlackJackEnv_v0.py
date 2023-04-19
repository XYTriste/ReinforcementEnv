from typing import Optional, Union, List, Tuple

import gym
from gym.core import RenderFrame, ActType, ObsType
from gym import spaces


def cmp(a, b):
    return float(a > b) - float(a < b)


def draw_card(np_random, deck):
    card = np_random.choice(deck)
    deck.remove(card)
    return int(card)


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
        self.deck = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        if action:
            self.player.append(draw_card(self.np_random, self.deck))
            if is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:
            terminated = True
            while sum(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random, self.deck))
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
        self.deck = [i for i in range(1, 11)] * 4 + [10] * 12
        self.dealer = [draw_card(self.np_random, self.deck)]
        self.player = [draw_card(self.np_random, self.deck)]

        return self._get_observation(), {}
