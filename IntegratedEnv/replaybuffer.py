# -*- coding = utf-8 -*-
# @time: 6/12/2023 2:13 AM
# Author: Yu Xia
# @File: replaybuffer.py
# @software: PyCharm
import numpy as np


class ReplayBuffer:
    def __init__(self, memory_capacity=100000, frame_history_len=4):
        self.MEMORY_CAPACITY = memory_capacity
        self.frame_history_len = frame_history_len

        self.memory = None
        self.obs_shape = None

        self.learning_starts = 200

        self.memory_counter = 0
        self.store_index = 0

    def store_memory_obs(self, frame):
        # if len(frame.shape) > 1:
        #     frame = frame.transpose(2, 0, 1)  # 转换数据存储维度以符合一般约定
        if self.obs_shape is None:
            self.obs_shape = frame.shape

        if self.memory is None:
            self.memory = [dict() for i in range(self.MEMORY_CAPACITY)]

        self.memory[self.store_index]['obs'] = frame
        index = self.store_index

        self.store_index = (self.store_index + 1) % self.MEMORY_CAPACITY
        self.memory_counter = min(self.memory_counter + 1, self.MEMORY_CAPACITY)

        return index

    def store_memory_effect(self, index, action, reward, done):
        self.memory[index]['action'] = action
        self.memory[index]['reward'] = reward
        self.memory[index]['done'] = done

    def _check_index(self, current_index):
        """
        如果经验回放缓冲区中的缓存数量不满足 frame_history_len 的帧数要求，则补 0 帧以进行对齐。

        current_index: 采样的起始下标
        frame_history_len: frame_history_len帧图片组成一个batch

        situation 1: current_index < frame_history_len and self.memory is not full
        该情况下说明缓冲区中不足 frame_history_len 帧，需要进行补 0 帧。

        situation 2: current_index < frame_history_len and self.memory is full
        注意，缓存区以栈的形式存储。
        该情况下说明从栈底取出若干个数量小于 frame_history_len 的帧，并从栈顶取出 frame_history_len - current_index
        数量的帧来拼接为一个batch。

        situation 3: 出现了游戏结束帧，需要进行 0 帧补齐。

        situation 4: 其他情况，不需要进行补帧。

        """
        end_index = current_index + 1
        start_index = end_index - self.frame_history_len
        is_sit_3 = False

        if start_index < 0:
            start_index = 0
            missing_context = self.frame_history_len - (end_index - start_index)

            if self.memory_counter != self.MEMORY_CAPACITY:
                for index in range(start_index, end_index - 1):
                    if 'done' in self.memory[index % self.MEMORY_CAPACITY] and self.memory[index % self.MEMORY_CAPACITY]['done']:
                        start_index = index + 1
                        is_sit_3 = True

                    if is_sit_3:
                        missing_context = self.frame_history_len - (end_index - start_index)
                        return 3, missing_context, start_index, end_index
                return 1, missing_context, start_index, end_index
            else:
                for index in range(start_index, end_index - 1):
                    if 'done' in self.memory[index % self.MEMORY_CAPACITY] and self.memory[index % self.MEMORY_CAPACITY]['done']:
                        start_index = index + 1
                        is_sit_3 = True

                    if is_sit_3:
                        missing_context = self.frame_history_len - (end_index - start_index)
                        return 3, missing_context, start_index, end_index

                for i in range(missing_context, 0, -1):
                    index = self.MEMORY_CAPACITY - i
                    if 'done' in self.memory[index % self.MEMORY_CAPACITY] and self.memory[index % self.MEMORY_CAPACITY]['done']:
                        start_index = (index + 1) % self.MEMORY_CAPACITY
                        is_sit_3 = True

                    if is_sit_3:
                        if start_index > end_index:
                            missing_context = self.frame_history_len - (self.MEMORY_CAPACITY - start_index + end_index)
                        else:
                            missing_context = self.frame_history_len - (end_index - start_index)
                        return 3, missing_context, start_index, end_index
                start_index = self.MEMORY_CAPACITY - missing_context
                return 2, 0, start_index, end_index

        for index in range(start_index, end_index - 1):
            if 'done' in self.memory[index % self.MEMORY_CAPACITY] and self.memory[index % self.MEMORY_CAPACITY]['done']:
                start_index = index + 1
                is_sit_3 = True

            if is_sit_3:
                missing_context = self.frame_history_len - (end_index - start_index)
                return 3, missing_context, start_index, end_index

        return 4, 0, start_index, end_index

    def _encoder_observation(self, current_index):
        encoded_observation = []

        index_flag, missing_context, start_index, end_index = self._check_index(current_index)
        if missing_context > 0:
            for i in range(missing_context):
                encoded_observation.append(np.zeros_like(self.memory[0]['obs']))

        if start_index > end_index:
            for index in range(start_index, self.MEMORY_CAPACITY):
                encoded_observation.append(self.memory[index % self.MEMORY_CAPACITY]['obs'])
            for index in range(end_index):
                encoded_observation.append(self.memory[index % self.MEMORY_CAPACITY]['obs'])
        else:
            for index in range(start_index, end_index):
                encoded_observation.append(self.memory[index % self.MEMORY_CAPACITY]['obs'])
        encoded_observation = np.concatenate(encoded_observation, 0)
        return encoded_observation

    def encoder_recent_observation(self):
        assert self.memory_counter > 0
        current_index = self.store_index - 1
        if current_index < 0:
            current_index = self.MEMORY_CAPACITY - 1

        return self._encoder_observation(current_index)

    def sample_memories(self, batch_size):
        sample_indexs = np.random.randint(0, self.memory_counter - 1, batch_size)

        # obs_batch = np.zeros(
        #     [batch_size, self.obs_shape[0] * self.frame_history_len, self.obs_shape[1], self.obs_shape[2]]
        # )
        obs_batch = np.zeros(
            [batch_size, self.frame_history_len, 84, 84]
        )
        action_batch = np.zeros([batch_size, 1])
        reward_batch = np.zeros([batch_size, 1])
        next_obs_batch = obs_batch.copy()
        done_batch = []

        for i in range(batch_size):
            obs_batch[i] = self._encoder_observation(sample_indexs[i])
            action_batch[i] = self.memory[sample_indexs[i]]['action']
            reward_batch[i] = self.memory[sample_indexs[i]]['reward']
            done_batch.append(self.memory[sample_indexs[i]]['done'])
            next_obs_batch[i] = self._encoder_observation(sample_indexs[i] + 1)
        return obs_batch, action_batch, reward_batch, next_obs_batch, np.array(done_batch)
# import numpy as np
# import random
#
#
# def sample_n_unique(sampling_f, n):
#     """Helper function. Given a function `sampling_f` that returns
#     comparable objects, sample n such unique objects.
#     """
#     res = []
#     while len(res) < n:
#         candidate = sampling_f()
#         if candidate not in res:
#             res.append(candidate)
#     return res
#
#
# class ReplayBuffer(object):
#     def __init__(self, size, frame_history_len):
#         """This is a memory efficient implementation of the replay buffer.
#         The sepecific memory optimizations use here are:
#             - only store each frame once rather than k times
#               even if every observation normally consists of k last frames
#             - store frames as np.uint8 (actually it is most time-performance
#               to cast them back to float32 on GPU to minimize memory transfer
#               time)
#             - store frame_t and frame_(t+1) in the same buffer.
#         For the tipical use case in Atari Deep RL buffer with 1M frames the total
#         memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes
#         Warning! Assumes that returning frame of zeros at the beginning
#         of the episode, when there is less frames than `frame_history_len`,
#         is acceptable.
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         frame_history_len: int
#             Number of memories to be retried for each observation.
#         """
#         self.size = size
#         self.frame_history_len = frame_history_len
#
#         self.next_idx = 0
#         self.num_in_buffer = 0
#         self.learning_starts = 50000
#         self.MEMORY_CAPACITY = 100000
#         self.frame_history_len = frame_history_len
#
#         self.learning_starts = 50000
#
#         self.memory_counter = 0
#         self.store_index = 0
#
#         self.obs = None
#         self.action = None
#         self.reward = None
#         self.done = None
#
#     def can_sample(self, batch_size):
#         """Returns true if `batch_size` different transitions can be sampled from the buffer."""
#         return batch_size + 1 <= self.num_in_buffer
#
#     def _encode_sample(self, idxes):
#         obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
#         act_batch = self.action[idxes]
#         rew_batch = self.reward[idxes]
#         next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
#         done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
#
#         return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask
#
#     def sample_memories(self, batch_size):
#         """Sample `batch_size` different transitions.
#         i-th sample transition is the following:
#         when observing `obs_batch[i]`, action `act_batch[i]` was taken,
#         after which reward `rew_batch[i]` was received and subsequent
#         observation  next_obs_batch[i] was observed, unless the epsiode
#         was done which is represented by `done_mask[i]` which is equal
#         to 1 if episode has ended as a result of that action.
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#         Returns
#         -------
#         obs_batch: np.array
#             Array of shape
#             (batch_size, img_c * frame_history_len, img_h, img_w)
#             and dtype np.uint8
#         act_batch: np.array
#             Array of shape (batch_size,) and dtype np.int32
#         rew_batch: np.array
#             Array of shape (batch_size,) and dtype np.float32
#         next_obs_batch: np.array
#             Array of shape
#             (batch_size, img_c * frame_history_len, img_h, img_w)
#             and dtype np.uint8
#         done_mask: np.array
#             Array of shape (batch_size,) and dtype np.float32
#         """
#         assert self.can_sample(batch_size)
#         idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
#         return self._encode_sample(idxes)
#
#     def encoder_recent_observation(self):
#         """Return the most recent `frame_history_len` frames.
#         Returns
#         -------
#         observation: np.array
#             Array of shape (img_c * frame_history_len, img_h, img_w)
#             and dtype np.uint8, where observation[i*img_c:(i+1)*img_c, :, :]
#             encodes frame at time `t - frame_history_len + i`
#         """
#         assert self.num_in_buffer > 0
#         return self._encode_observation((self.next_idx - 1) % self.size)
#
#     def _encode_observation(self, idx):
#         end_idx = idx + 1  # make noninclusive
#         start_idx = end_idx - self.frame_history_len
#         # this checks if we are using low-dimensional observations, such as RAM
#         # state, in which case we just directly return the latest RAM.
#         if len(self.obs.shape) == 2:
#             return self.obs[end_idx - 1]
#         # if there weren't enough frames ever in the buffer for context
#         if start_idx < 0 and self.num_in_buffer != self.size:
#             start_idx = 0
#         for idx in range(start_idx, end_idx - 1):
#             if self.done[idx % self.size]:
#                 start_idx = idx + 1
#         missing_context = self.frame_history_len - (end_idx - start_idx)
#         # if zero padding is needed for missing context
#         # or we are on the boundry of the buffer
#         if start_idx < 0 or missing_context > 0:
#             frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
#             for idx in range(start_idx, end_idx):
#                 frames.append(self.obs[idx % self.size])
#             return np.concatenate(frames, 0)  # c, h, w instead of h, w c
#         else:
#             # this optimization has potential to saves about 30% compute time \o/
#             # c, h, w instead of h, w c
#             img_h, img_w = self.obs.shape[2], self.obs.shape[3]
#             return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)
#
#     def store_memory_obs(self, frame):
#         """Store a single frame in the buffer at the next available index, overwriting
#         old frames if necessary.
#         Parameters
#         ----------
#         frame: np.array
#             Array of shape (img_h, img_w, img_c) and dtype np.uint8
#             the frame to be stored
#         Returns
#         -------
#         idx: int
#             Index at which the frame is stored. To be used for `store_effect` later.
#         """
#         # if observation is an image...
#         if len(frame.shape) > 1:
#             # transpose image frame into c, h, w instead of h, w, c
#             frame = frame.transpose(2, 0, 1)
#
#         if self.obs is None:
#             self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
#             self.action = np.empty([self.size], dtype=np.int32)
#             self.reward = np.empty([self.size], dtype=np.float32)
#             self.done = np.empty([self.size], dtype=np.bool_)
#         self.obs[self.next_idx] = frame
#
#         ret = self.next_idx
#         self.next_idx = (self.next_idx + 1) % self.size
#         self.store_index = self.next_idx
#         self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
#         self.memory_counter = self.num_in_buffer
#
#         return ret
#
#     def store_memory_effect(self, idx, action, reward, done):
#         """Store effects of action taken after obeserving frame stored
#         at index idx. The reason `store_frame` and `store_effect` is broken
#         up into two functions is so that once can call `encode_recent_observation`
#         in between.
#         Paramters
#         ---------
#         idx: int
#             Index in buffer of recently observed frame (returned by `store_frame`).
#         action: int
#             Action that was performed upon observing this frame.
#         reward: float
#             Reward that was received when the actions was performed.
#         done: bool
#             True if episode was finished after performing that action.
#         """
#         self.action[idx] = action
#         self.reward[idx] = reward
#         self.done[idx] = done
