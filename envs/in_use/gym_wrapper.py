import sys

import pygame
import torch


from envs.in_use.v_3 import v_3_idx_to_scalar, V_3_LEFT_REGION, V_3_RIGHT_REGION
import envs.in_use.v_1
import envs.in_use.v_2
import envs.in_use.v_3
import envs.in_use.v_4
import envs.in_use.v_5
import envs.in_use.v_6
import envs.in_use.v_3v
import envs.in_use.two_objs
import envs.in_use.small_test
from envs.in_use.v_4 import WAREHOUSE_FG_COLOURS_1
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

class GymWrapper(gym.Env):
    """Gym wrapper for pycolab environment"""

    def __init__(self, env_id):
        self.env_id = env_id
        self.current_observation = None
        self.termination_reason = None

        # Initialize Pygame variables
        self.screen = None
        self.clock = None

        if env_id == 'v_1':
            self.layers = ('#', 'P', 'C', 'G')
            self.width = 11
            self.height = 5
            self.num_actions = 9

        elif env_id == 'v_2':
            self.layers = ('#', 'P', 'C', 'V')
            self.width = 10
            self.height = 10
            self.num_actions = 9

        elif env_id == 'v_3':
            self.layers = ('#', 'P', 'C', 'V')
            self.width = 8
            self.height = 8
            self.num_actions = 9

        elif env_id == 'v_3v':
            self.layers = ('#', 'P', 'C', 'V')
            self.width = 8
            self.height = 8
            self.num_actions = 9

        elif env_id == 'v_4':
            self.layers = ('#', 'P', 'C', 'V', 'G')
            self.width = 12
            self.height = 12
            self.num_actions = 9

        elif env_id == 'v_5':
            self.layers = ('#', 'P', 'C', 'G', 'V')
            self.width = 10
            self.height = 10
            self.num_actions = 9

        elif env_id == 'v_6':
            self.layers = ('#', 'P', 'C', 'G', 'V', 'F')
            self.width = 12
            self.height = 12
            self.num_actions = 9


        elif env_id == 'two_objs':
            self.layers = ('#', 'P', 'G', 'V')
            self.width = 8
            self.height = 8
            self.num_actions = 9

        elif env_id == 'small_test':
            self.layers = ('#', 'P', 'C', 'V')
            self.width = 8
            self.height = 8
            self.num_actions = 9



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

    def get_obs_array(self):
        return self._obs_to_np_array(self.current_observation)


    def close(self):
        self.close()

    def reset(self, msg=None):
        if self.env_id == 'v_1':
            self.game = envs.in_use.v_1.make_game()

        elif self.env_id == 'v_2':
            self.game = envs.in_use.v_2.make_game()

        elif self.env_id == 'v_3':
            if msg is None:
                self.game = envs.in_use.v_3.make_game()
            else:
                self.game = envs.in_use.v_3.make_game(msg=msg)

        elif self.env_id == 'v_3v':
            if msg is None:
                self.game = envs.in_use.v_3v.make_game()
            else:
                self.game = envs.in_use.v_3v.make_game(msg=msg)

        elif self.env_id == 'v_4':
            if msg is None:
                self.game = envs.in_use.v_4.make_game()
            else:
                self.game = envs.in_use.v_4.make_game(msg)

        elif self.env_id == 'v_5':
            if msg is None:
                self.game = envs.in_use.v_5.make_game()
            else:
                self.game = envs.in_use.v_5.make_game(msg=msg)

        elif self.env_id == 'v_6':
            self.game = envs.in_use.v_6.make_game()

        elif self.env_id == 'two_objs':
            self.game = envs.in_use.two_objs.make_game()

        elif self.env_id == 'small_test':
            self.game = envs.in_use.small_test.make_game()


        obs, _, _ = self.game.its_showtime()

        self.current_observation = obs

        return self._obs_to_np_array(obs)

    def step(self, action):
        #print("before game : ", self.game.game_over)

        obs, reward, _ = self.game.play(action)

        #print("after game : ", self.game.game_over)
        self.current_observation = obs
        if self.game.game_over:
            if self.game.the_plot['terminal_reason'] == 'max_step':
                self.game.the_plot['dw'] = 0
                self.game.the_plot['do'] = 1
                #print("max step")
                return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot

            elif self.game.game_over and self.game.the_plot['terminal_reason'] == 'terminal_state':
                self.game.the_plot['dw'] = 1
                self.game.the_plot['do'] = 1
                #print("terminal")
                return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot
        else:
            self.game.the_plot['dw'] = 0
            self.game.the_plot['do'] = 0
            return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot

        return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot

    def render(self, mode='human'):
        drawn_tiles = np.zeros((self.height, self.width), dtype=bool)  # Track drawn tiles

        if mode != 'human':
            raise NotImplementedError("Only 'human' render mode is supported.")

        if self.current_observation is None:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * 50, self.height * 50))
            self.clock = pygame.time.Clock()

        # Convert observation to a numpy array
        obs_array = self._obs_to_np_array(self.current_observation)

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw the left and right region backgrounds
        for row in range(self.height):
            for col in range(self.width):
                background_color = (255, 255, 255)
                if v_3_idx_to_scalar(row, col) in V_3_LEFT_REGION:
                    background_color = (225, 229, 204)
                if v_3_idx_to_scalar(row, col) in V_3_RIGHT_REGION:
                    background_color = (204, 229, 255)

                # # # One transition version, transition middle
                # if idx_to_scalar(row, col) in TRANSITION_REGION_1:
                #     background_color = (255, 204, 229)
                # #
                # # Two transition version
                # if idx_to_scalar(row, col) in TRANSITION_REGION_2_LEFT or idx_to_scalar(row, col) in TRANSITION_REGION_2_RIGHT:
                #     background_color = (255, 255, 204)

                # else:
                #     background_color = (255, 255, 255)  # Default background color

                pygame.draw.rect(self.screen, background_color, (col * 50, row * 50, 50, 50))

        # Draw the observation
        for num_layer in range(4):
            for row in range(self.height):
                for col in range(self.width):
                    tile_value = obs_array[num_layer][row, col]


                    # Only draw if a tile has not been drawn here yet
                    if not drawn_tiles[row, col]:
                        if tile_value == 1:  # Check if tile_value equals 1
                            color = WAREHOUSE_FG_COLOURS_1[self.layers[num_layer]]
                            pygame.draw.rect(self.screen, color, (col * 50, row * 50, 50, 50))
                            drawn_tiles[row, col] = True
                        # else:
                        #     pos = idx_to_scalar(row, col)
                        #     if pos in LEFT_REGION:
                        #         color = (255, 299, 204)  # Default color
                        #     elif pos in RIGHT_REGION:
                        #         color = (204, 229, 255)
                        #     pygame.draw.rect(self.screen, color, (col * 50, row * 50, 50, 50))

        pygame.display.flip()
        self.clock.tick(3)  # Cap the frame rate

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def _get_color_from_tile_value(self, tile_value, n_layer):
        # Define this method to map tile values to colors
        # Example:
        if tile_value.any() == 1:  # Assuming 1 represents walls
            color = WAREHOUSE_FG_COLOURS_1[n_layer]
            return color  # Black color for walls
        # Add more conditions for other tile types
        else:
            return (255, 255, 255)  # Default color


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
            #print("done i : ", done_i, "  info i : ", info_i)
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
