import math
import random

import gym
from gym import spaces
import numpy as np
import pygame
import time


class MazeEnv(gym.Env):
    def __init__(self, width, height, render_mode="rgb_array", frame_frequency = 30):
        super(MazeEnv, self).__init__()

        assert render_mode == "rgb_array" or render_mode == "human"
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(4)  # 0: 上, 1: 下, 2: 左, 3: 右
        self.observation_space = spaces.Discrete(width * height)

        self.grid = np.zeros((width, height), dtype=np.int32)
        self.visit_count = np.zeros((width, height), dtype=np.int32)
        self.goal = (width - 1, height - 1)  # 终点坐标
        self.grid[self.goal] = 2

        self.MAX_VISIT = 255
        # 设置初始位置
        self.agent_pos = np.array([0, 0])
        self.N_STATES = 2   # 保存智能体状态所需的大小
        self.N_ACTIONS = 4
        self.grid[self.agent_pos] = 1

        # 在副对角线上设置障碍物
        # for i in range(self.width // 2):
        #     j = self.width - 1 - i
        #     self.grid[i][j] = -1
        obstacles_freq = math.floor(width * height * 0.2)
        for _ in range(obstacles_freq):
            obstacle_row = random.randint(0, width - 1)
            obstacle_col = random.randint(0, height - 1)
            self.grid[obstacle_row][obstacle_col] = -1

        self.frame_freq = 1 / frame_frequency

        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen_width = min(800, width * 15)
            self.screen_height = min(600, height * 15)
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Maze Environment")

        # 配置颜色
        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)
        }

    def step(self, action):
        if action == 0:  # 上
            new_pos = np.array([self.agent_pos[0], max(0, self.agent_pos[1] - 1)])
        elif action == 1:  # 下
            new_pos = np.array([self.agent_pos[0], min(self.height - 1, self.agent_pos[1] + 1)])
        elif action == 2:  # 左
            new_pos = np.array([max(0, self.agent_pos[0] - 1), self.agent_pos[1]])
        elif action == 3:  # 右
            new_pos = np.array([min(self.width - 1, self.agent_pos[0] + 1), self.agent_pos[1]])

        if self.grid[new_pos[0], new_pos[1]] != -1:  # 检查是否为障碍物
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
            self.agent_pos = new_pos
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 1

        self.visit_count[self.agent_pos[0], self.agent_pos[1]] += 1
        done = (self.agent_pos[0] == self.goal[0] and self.agent_pos[1] == self.goal[1])  # 判断是否到达终点
        reward = 1.0 if done else 0.0

        if self.render_mode == "human":
            self.render()
            time.sleep(self.frame_freq)

        return self.agent_pos, reward, done, {}, {}

    def reset(self):
        self.grid[self.agent_pos] = 0
        self.agent_pos = (0, 0)
        self.grid[self.agent_pos] = 1
        if self.render_mode == "human":
            self.render()
        return self.agent_pos, {}

    def render(self):
        if self.render_mode == 'human':
            for event in pygame.event.get():  # 添加事件处理循环
                if event.type == pygame.QUIT:  # 检查是否点击了关闭窗口按钮
                    pygame.quit()  # 退出Pygame
                    return  # 退出render函数

            self.screen.fill(self.colors["white"])
            cell_width = self.screen_width // self.width
            cell_height = self.screen_height // self.height

            for row in range(self.width):
                for col in range(self.height):
                    cell_x = col * cell_width
                    cell_y = row * cell_height

                    if self.grid[row][col] == 0:  # 空白
                        pygame.draw.rect(self.screen, self.colors["white"], (cell_x, cell_y, cell_width, cell_height))
                    elif self.grid[row][col] == 1:  # 代表代理
                        # pygame.draw.rect(self.screen, self.colors["green"], (cell_x, cell_y, cell_width, cell_height))
                        visit_freq = self.visit_count[row][col]
                        green_intensity = min(255, int(visit_freq / self.MAX_VISIT) * 255)
                        grid_color = (0, green_intensity, 0)
                        pygame.draw.rect(self.screen, grid_color, (cell_x, cell_y, cell_width, cell_height))
                    elif self.grid[row][col] == 2:  # 代表终点
                        pygame.draw.rect(self.screen, self.colors["blue"], (cell_x, cell_y, cell_width, cell_height))
                    elif self.grid[row][col] == -1: # 代表障碍物
                        pygame.draw.rect(self.screen, self.colors["red"], (cell_x, cell_y, cell_width, cell_height))

            # 绘制网格线
            for row in range(self.width):
                pygame.draw.line(self.screen, self.colors["black"], (0, row * cell_height),
                                 (self.screen_width, row * cell_height))
            for col in range(self.height):
                pygame.draw.line(self.screen, self.colors["black"], (col * cell_width, 0),
                                 (col * cell_width, self.screen_height))

            pygame.display.flip()
            time.sleep(0.016)
        else:
            super(MazeEnv, self).render()

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = MazeEnv(80, 80)
    while True:
        env.reset()
        for _ in range(20):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            time.sleep(0.016)
            if done:
                print("Episode finished.")
                break
    env.close()
