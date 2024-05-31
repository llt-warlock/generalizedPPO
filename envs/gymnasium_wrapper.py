import gymnasium
import torch
import envs.randomized_v3
import envs.warehouse_0
import envs.driving
import envs.warehouse
import envs.robot
import envs.randomized_v2
import envs.final_big
import envs.test_1
import envs.test_2
import envs.new_test_2
import envs.evaluate_test
import envs.new_test_1
from pycolab import rendering
from typing import Callable
import gym
from gym import spaces
from gym.utils import seeding
import copy
import numpy as np
import time

from stable_baselines3.common.utils import set_random_seed

import envs.env_0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GymWrapper(gymnasium.Env):
    """Gym wrapper for pycolab environment"""

    def __init__(self, env_id):
        self.env_id = env_id
        self.termination_reason = None

        if env_id == 'randomized_v3':
            self.layers = ('#', 'P', 'F', 'C', 'S', 'V')
            self.width = 16
            self.height = 16
            self.num_actions = 9

        elif env_id == 'warehouse_0':
            self.layers = ('#', 'P', '$', '&', 'A', 'B')
            self.width = 32
            self.height = 5
            self.num_actions = 7

        elif env_id == 'driving':
            self.layers = ('#', 'G', 'P')
            self.width = 14
            self.height = 5
            self.num_actions = 6

        elif env_id == 'warehouse':
            self.layers = ('#', 'G', 'C', 'P')
            self.width = 32
            self.height = 7
            self.num_actions = 9

        elif env_id == 'robot':
            self.layers = ('#', 'G', 'C', 'P')
            self.width = 22
            self.height = 6
            self.num_actions = 9

        elif env_id == 'randomized_v2':
            self.layers = ('#', 'P', 'C', 'H', 'G')
            self.width = 8
            self.height = 8
            self.num_actions = 9

        elif env_id == 'test_1':
            self.layers = ('#', 'P', 'C', 'G')
            self.width = 14
            self.height = 6
            self.num_actions = 5

        elif env_id == 'test_2':
            self.layers = ('#', 'P', 'C')
            self.width = 8
            self.height = 5
            self.num_actions = 6

        elif env_id == 'final_big':
            self.layers = ('#', 'P', 'I', 'V', 'G')
            self.width = 22
            self.height = 22
            self.num_actions = 25

        elif env_id == 'new_test_1':
            #self.layers = ('#', 'P', 'C', 'V', 'F', 'G')
            self.layers = ('#', 'P', 'O', 'V', 'C', 'F', 'G')
            self.width = 12
            self.height = 12
            self.num_actions = 9

        elif env_id == 'new_test_2':
            self.layers = ('#', 'P', 'V', 'C', 'F', 'G')
            self.width = 26
            self.height = 6
            self.num_actions = 18

        self.game = None
        self.np_random = None

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.width, self.height, len(self.layers)),
            dtype=np.int32
        )

        self.renderer = rendering.ObservationToFeatureArray(self.layers)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _obs_to_np_array(self, obs):
        return copy.copy(self.renderer(obs))

    def reset(self):
        if self.env_id == 'randomized_v3':
            self.game = envs.randomized_v3.make_game()

        elif self.env_id == 'warehouse_0':
            self.game = envs.warehouse_0.make_game()

        elif self.env_id == 'driving':
            self.game = envs.driving.make_game()

        elif self.env_id == 'warehouse':
            self.game = envs.warehouse.make_game()

        elif self.env_id == 'robot':
            self.game = envs.robot.make_game()

        elif self.env_id == 'randomized_v2':
            self.game = envs.randomized_v2.make_game()

        elif self.env_id == 'test_1':
            self.game = envs.test_1.make_game()

        elif self.env_id == 'test_2':
            self.game = envs.test_2.make_game()

        elif self.env_id == 'final_big':
            self.game = envs.final_big.make_game()

        elif self.env_id == 'new_test_1':
            self.game = envs.new_test_1.make_game()

        elif self.env_id == 'new_test_2':
            self.game = envs.new_test_2.make_game()

        obs, _, _ = self.game.its_showtime()

        return self._obs_to_np_array(obs)

    def step(self, action):
        obs, reward, _ = self.game.play(action)

        if self.game.game_over:
            if self.game.the_plot['terminal_reason'] == 'max_step':
                self.game.the_plot['dw'] = 0
                self.game.the_plot['do'] = 1
                #print("max step")
                return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot
            elif self.game.game_over and self.game.the_plot['terminal_reason'] == 'terminal_state':
                self.game.the_plot['dw'] = 1
                self.game.the_plot['do'] = 1
                #print("terminate state")
                return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot
        else:
            self.game.the_plot['dw'] = 0
            self.game.the_plot['do'] = 0
            return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot

        #return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = GymWrapper(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

class VecEnv:
    def __init__(self, env_id, n_envs):
        self.env_list = [make_env(env_id, i, (int(str(time.time()).replace('.', '')[-8:]) + i))() for i in range(n_envs)]
        self.n_envs = n_envs
        self.env_id = env_id
        self.action_space = self.env_list[0].action_space
        self.observation_space = self.env_list[0].observation_space

    def reset(self):
        obs_list = []
        for i in range(self.n_envs):
            obs_list.append(self.env_list[i].reset())

        return np.stack(obs_list, axis=0)

    def step(self, actions):
        obs_list = []
        rew_list = []
        done_list = []
        dw_list = []
        do_list = []
        info_list = []
        for i in range(self.n_envs):
            obs_i, rew_i, done_i, info_i = self.env_list[i].step(actions[i])
            #obs_i, rew_i, done_i, dw_i, do_i, info_i = self.env_list[i].step(actions[i])
            if done_i:
                obs_i = self.env_list[i].reset()

            obs_list.append(obs_i)
            rew_list.append(rew_i)
            done_list.append(done_i)
            # dw_list.append(dw_i)
            # do_list.append(do_i)
            info_list.append(info_i)

        #print("dw_i ", dw_list, " do_i ", do_list)

        #print(info_list[1])

        return np.stack(obs_list, axis=0), rew_list, done_list, info_list
        #return np.stack(obs_list, axis=0), rew_list, done_list, info_list


    # def rollout(self, ppo, states_tensor, number_rollout, number_of_imagination):
    #     rollout = [[] for _ in range(self.n_envs)]
    #     for _ in range(number_rollout):
    #         imagination = [[] for _ in range(self.n_envs)]
    #         episode_end = [False for _ in range(self.n_envs)]
    #
    #         input_states = states_tensor
    #
    #         for i_ in range(number_of_imagination):
    #             actions, log_probs = ppo.act(input_states)
    #             next_states, rewards, done, info = self.step(actions)
    #
    #             for w_i in range(self.n_envs):
    #                 if not episode_end[w_i]:
    #                     imagination[w_i].append(info[w_i]['P_pos'])
    #                     if done[w_i]:
    #                         episode_end[w_i] = True
    #
    #             input_states = torch.tensor(next_states).float().to(device)
    #
    #         for w__i in range(self.n_envs):
    #             rollout[w__i].append(imagination[w__i])
    #
    #     return rollout
