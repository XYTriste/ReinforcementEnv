from typing import Optional, Union, List, Tuple

import gym
import numpy as np
from gym import spaces
from gym.core import RenderFrame, ActType, ObsType


def isLegalIndex(index, grid_width, grid_height):
    x, y = index
    return 0 <= x < grid_width \
        and 0 <= x < grid_width \
        and 0 <= y < grid_height \
        and 0 <= y < grid_height


def visit_end(index, end_index):
    return index == end_index


class WindGridEnv(gym.Env):
    winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def __init__(self, grid_width=7, grid_height=10, start_index=(3, 0), end_index=(3, 7), rewards=(0, -1, 1)):
        self.start_index = start_index
        self.end_index = end_index
        self.rewards = rewards  # 环境中不同位置的奖励，第一个是移动奖励，第二个是撞墙的奖励，第三个是到达终点的奖励

        self.player_index = None

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_map = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.grid_width), spaces.Discrete(grid_height)))

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        transition = self.get_transition(action)
        position, transition_flag = self.state_transition(transition)
        if transition_flag:
            reward = self.rewards[0]
            if visit_end(position, self.end_index):
                reward = self.rewards[2]
                terminated = True
            else:
                terminated = False
        else:
            reward = self.rewards[1]
            terminated = False
        return self._get_observation(), reward, terminated, False, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            rewards: Optional[Tuple] = None,
    ) -> Tuple[ObsType, dict]:
        self.grid_map = np.zeros((self.grid_width, self.grid_height))
        self.player_index = self.start_index
        if rewards is not None:
            self.rewards = rewards

        return self._get_observation(), {}

    def _get_observation(self):
        return self.player_index

    def get_transition(self, action):
        assert action in self.action_space, "无效的动作"
        if action == 0:
            return -1, 0
        elif action == 1:
            return 0, 1
        elif action == 2:
            return 1, 0
        elif action == 3:
            return 0, -1

    def state_transition(self, transition):
        assert transition is not None
        transition_x, transition_y = transition
        after_index = (self.player_index[0] + transition_x, self.player_index[1] + transition_y)
        transition_flag = False
        if isLegalIndex(after_index, self.grid_width, self.grid_height):
            self.player_index = after_index
            transition_flag = True
            self.blow()
        return self.player_index, transition_flag

    def blow(self):
        """
        风将玩家吹到别的位置去
        """
        assert len(WindGridEnv.winds) == self.grid_height, "风的宽度和格子世界的宽度不同"
        player_index_x, player_index_y = self.player_index
        blowing_rate = WindGridEnv.winds[player_index_y]
        after_blow_x = player_index_x - blowing_rate
        after_blow_y = player_index_y
        if after_blow_x < 0:
            after_blow_x = 0
        elif after_blow_x < self.end_index[0] < player_index_x:
            self.player_index = self.end_index
            return
        self.player_index = after_blow_x, after_blow_y
