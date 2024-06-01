import gymnasium as gym
import threading
import pygame

# 创建CartPole-v1环境
env = gym.make('CartPole-v1', render_mode='human')

# 收集专家数据
expert_data = []
num_episodes = 3


def register_input():
    global quit, restart
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                return 0
            if event.key == pygame.K_RIGHT:
                return 1  # set 1.0 for wheels to block to zero rotation
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_ESCAPE:
                quit = True
        if event.type == pygame.QUIT:
            quit = True


def manual_control():
    global expert_data
    state, _ = env.reset()
    done = False
    episode_data = []

    while True:
        env.render()

        action = register_input()  # 从键盘输入动作
        next_state, reward, done, _, _ = env.step(action)
        episode_data.append((state, action))
        state = next_state

        if done:
            expert_data.extend(episode_data)
            break


# 创建一个线程用于手动控制
control_thread = threading.Thread(target=manual_control)
control_thread.start()

# 等待手动控制线程结束
control_thread.join()

# 关闭环境显示
env.close()

# 打印收集到的专家数据
print("Expert Data:")
for transition in expert_data:
    print(transition)
