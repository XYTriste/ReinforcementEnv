from WindGridEnv import *
import gym

env = WindGridEnv()
done = False
env.reset()
experience_step = 0
while not done:
    action = env.action_space.sample()
    cur_index, _, done, _, _ = env.step(action)
    print("当前坐标: {}".format(cur_index))
    experience_step += 1
print("总共走了 {} 步".format(experience_step))
