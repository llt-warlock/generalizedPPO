import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import torch.nn.init as init
# Use GPU if available
from torch.utils.data import DataLoader

from adaption.auxiliary import preference_context_generate
from envs.in_use.v_4 import LEFT_REGION, RIGHT_REGION, REGION_0, REGION_1, REGION_2, REGION_3
from utils.region import region_check, rolling_window_sum
from envs import driving
from envs.in_use.v_4 import idx_to_scalar, LEFT_REGION, RIGHT_REGION
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TrajectoryDataset:
    def __init__(self, batch_size, n_workers):
        self.batch_size = batch_size
        self.count = 0
        self.n_workers = n_workers
        self.trajectories = []

        # objective
        self.buffer = [{'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'dw': [], 'do': [],
                        'log_probs': [], 'latents': None, 'logs': [], 'leftA':[], 'leftB':[], 'rightA':[], 'rightB':[],
                        'leftAmax': None, 'rightAmax':None, 'leftBmax':None, 'rightBmax':None, 'experts':[], 'weights':[]}
                       for i in range(n_workers)]


        self.n_step_returns = []

    def remove_first_element_from_buffer(self):
        for key, value in self.buffer.items():
            if isinstance(value,list) and len(value) > 0:
                self.buffer[key] = value[1:]


    def reset_buffer(self, i):
        self.buffer[i] = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'dw': [],
                          'do': [], 'log_probs': [], 'latents': None, 'logs': [], 'leftA':[], 'leftB':[], 'rightA':[], 'rightB':[],
                          'leftAmax': None, 'rightAmax':None, 'leftBmax':None, 'rightBmax':None, 'experts':[], 'weights':[]}

    def reset_trajectories(self):
        self.trajectories = []
        self.n_step_returns = []

    def copy_trajectories(self):
        return self.trajectories.copy()

    def write_tuple(self, states, actions, next_states, done, log_probs, logs=None, info=None,
                    expert_reward=None, rewards=None, weights=None):
        # Takes states of shape (n_workers, state_shape[0], state_shape[1])
        for i in range(self.n_workers):
            self.buffer[i]['states'].append(states[i])
            self.buffer[i]['actions'].append(actions[i])
            self.buffer[i]['next_states'].append(next_states[i])
            #self.buffer[i]['rewards'].append(rewards[i])
            self.buffer[i]['dones'].append(done[i])
            self.buffer[i]['log_probs'].append(log_probs[i])

            if rewards is not None:
                #print("rewards is not None")
                self.buffer[i]['rewards'].append(rewards[i])
            if info is not None:
                self.buffer[i]['dw'].append(info[i]['dw'])
                self.buffer[i]['do'].append(info[i]['do'])

                self.buffer[i]['leftA'].append(info[i]['left_item_A'])
                self.buffer[i]['rightA'].append(info[i]['right_item_A'])

                self.buffer[i]['leftB'].append(info[i]['left_item_B'])
                self.buffer[i]['rightB'].append(info[i]['right_item_B'])

                self.buffer[i]['leftAmax'] = info[i]['item_A_left_max']
                self.buffer[i]['rightAmax'] = info[i]['item_A_right_max']

                self.buffer[i]['leftBmax'] = info[i]['item_B_left_max']
                self.buffer[i]['rightBmax'] = info[i]['item_B_right_max']

            if logs is not None:
                self.buffer[i]['logs'].append(logs[i])  # reward vector

            if expert_reward is not None:
                self.buffer[i]['experts'].append(expert_reward[i])

            if weights is not None:
                self.buffer[i]['weights'].append(weights[i])

            if done[i]:
                self.trajectories.append(self.buffer[i].copy())
                self.reset_buffer(i)

        if len(self.trajectories) >= self.batch_size:
            return True
        else:
            return False

    def write_n_step_returns(self, trajectory_return):
        self.n_step_returns.append(trajectory_return)

    def log_n_step_returns(self):
        # Calculates (undiscounted) returns in self.trajectories
        returns = [0 for i in range(len(self.n_step_returns))]
        for i, tau in enumerate(self.n_step_returns):
            returns[i] = tau
        return np.array(returns)

    def log_item_A(self):
        left_item_A_ratio = []
        right_item_A_ratio = []
        for i, tau in enumerate(self.trajectories):
            left_item_A_ratio.append(np.array(tau['leftA']).sum()/tau['leftAmax'])
            right_item_A_ratio.append(np.array(tau['rightA']).sum()/tau['rightAmax'])
        return np.array(left_item_A_ratio), np.array(right_item_A_ratio)

    def log_item_B(self):
        left_item_B_ratio = []
        right_item_B_ratio = []
        for i, tau in enumerate(self.trajectories):
            left_item_B_ratio.append(np.array(tau['leftB']).sum()/tau['leftBmax'])
            right_item_B_ratio.append(np.array(tau['rightB']).sum()/tau['rightBmax'])
        return np.array(left_item_B_ratio), np.array(right_item_B_ratio)

    def log_returns(self):
        # Calculates achieved objectives objectives in self.trajectories
        trajectory_0 = self.filter_trajectories_by_weight([1.0, 0.0])
        trajectory_1 = self.filter_trajectories_by_weight([0.0, 1.0])
        # trajectory_2 = self.filter_trajectories_by_weight([0.0, 1.0])
        # trajectory_3 = self.filter_trajectories_by_weight([0.3, 0.7])
        # trajectory_4 = self.filter_trajectories_by_weight([0.0, 1.0])
        # trajectory_5 = self.filter_trajectories_by_weight([0.5, 0.5])
        # trajectory_6 = self.filter_trajectories_by_weight([0.4, 0.6])
        # trajectory_7 = self.filter_trajectories_by_weight([0.3, 0.7])
        # trajectory_8 = self.filter_trajectories_by_weight([0.2, 0.8])
        # trajectory_9 = self.filter_trajectories_by_weight([0.1, 0.9])
        # trajectory_10 = self.filter_trajectories_by_weight([0.0, 1.0])

        # Calculates (undiscounted) returns in self.trajectories
        returns_0 = [0 for i in range(len(trajectory_0))]
        for i, tau in enumerate(trajectory_0):
            returns_0[i] = sum(tau['rewards'])
        # Calculates (undiscounted) returns in self.trajectories
        returns_1 = [0 for i in range(len(trajectory_1))]
        for i, tau in enumerate(trajectory_1):
            returns_1[i] = sum(tau['rewards'])
        # # Calculates (undiscounted) returns in self.trajectories
        # returns_2 = [0 for i in range(len(trajectory_2))]
        # for i, tau in enumerate(trajectory_2):
        #     returns_2[i] = sum(tau['rewards'])
        # # Calculates (undiscounted) returns in self.trajectories
        # returns_3 = [0 for i in range(len(trajectory_3))]
        # for i, tau in enumerate(trajectory_3):
        #     returns_3[i] = sum(tau['rewards'])
        # # Calculates (undiscounted) returns in self.trajectories
        # returns_4 = [0 for i in range(len(trajectory_4))]
        # for i, tau in enumerate(trajectory_4):
        #     returns_4[i] = sum(tau['rewards'])
        # Calculates (undiscounted) returns in self.trajectories
        # returns_5 = [0 for i in range(len(trajectory_5))]
        # for i, tau in enumerate(trajectory_5):
        #     returns_5[i] = sum(tau['rewards'])
        # # Calculates (undiscounted) returns in self.trajectories
        # returns_6 = [0 for i in range(len(trajectory_6))]
        # for i, tau in enumerate(trajectory_6):
        #     returns_6[i] = sum(tau['rewards'])
        # # Calculates (undiscounted) returns in self.trajectories
        # returns_7 = [0 for i in range(len(trajectory_7))]
        # for i, tau in enumerate(trajectory_7):
        #     returns_7[i] = sum(tau['rewards'])
        # # Calculates (undiscounted) returns in self.trajectories
        # returns_8 = [0 for i in range(len(trajectory_8))]
        # for i, tau in enumerate(trajectory_8):
        #     returns_8[i] = sum(tau['rewards'])
        # # Calculates (undiscounted) returns in self.trajectories
        # returns_9 = [0 for i in range(len(trajectory_9))]
        # for i, tau in enumerate(trajectory_9):
        #     returns_9[i] = sum(tau['rewards'])
        # # Calculates (undiscounted) returns in self.trajectories
        # returns_10 = [0 for i in range(len(trajectory_10))]
        # for i, tau in enumerate(trajectory_10):
        #     returns_10[i] = sum(tau['rewards'])

        return np.array(returns_0), np.array(returns_1)


    def filter_trajectories_by_weight(self, target_weight):
        # Convert target_weight to a tensor for comparison
        target_weight_tensor = torch.tensor(target_weight).to(device)

        filtered_traj = []
        for traj in self.trajectories:
            if all(torch.equal(w, target_weight_tensor) for w in traj['weights']):
                filtered_traj.append(traj)

        return filtered_traj


    def log_objectives_base(self):
        # Calculates achieved objectives objectives in self.trajectories
        objective_logs = []
        for i, tau in enumerate(self.trajectories):
            objective_logs.append(list(np.array(tau['logs']).sum(axis=0)))

        return np.array(objective_logs)


    def log_objectives(self):
        # objective_logs = []
        # for i, tau in enumerate(self.trajectories):
        #     objective_logs.append(list(np.array(tau['logs']).sum(axis=0)))
        trajectory_0 = self.filter_trajectories_by_weight([1.0, 0.0])
        trajectory_1 = self.filter_trajectories_by_weight([0.5, 0.5])
        trajectory_2 = self.filter_trajectories_by_weight([0.0, 1.0])

        # Calculates achieved objectives objectives in self.trajectories
        # trajectory_0 = self.filter_trajectories_by_weight([1.0, 0.0, 0.0])
        # trajectory_1 = self.filter_trajectories_by_weight([0.0, 0.0, 1.0])
        # trajectory_2 = self.filter_trajectories_by_weight([0.0, 1.0, 0.0])
        # trajectory_3 = self.filter_trajectories_by_weight([0.75, 0.25, 0.0])
        # trajectory_4 = self.filter_trajectories_by_weight([0.75, 0.0, 0.25])
        # trajectory_5 = self.filter_trajectories_by_weight([0.5, 0.5, 0.0])
        # trajectory_6 = self.filter_trajectories_by_weight([0.5, 0.0, 0.5])
        # trajectory_7 = self.filter_trajectories_by_weight([0.25, 0.75, 0.0])
        # trajectory_8 = self.filter_trajectories_by_weight([0.25, 0.0, 0.75])
        # trajectory_9 = self.filter_trajectories_by_weight([0.0, 0.25, 0.75])
        # trajectory_10 = self.filter_trajectories_by_weight([0.0, 0.75, 0.25])
        # trajectory_11 = self.filter_trajectories_by_weight([0.0, 0.5, 0.5])

        objective_logs_pre_0 = []
        objective_logs_pre_1 = []
        objective_logs_pre_2 = []
        # objective_logs_pre_3 = []
        # objective_logs_pre_4 = []
        # objective_logs_pre_5 = []
        # objective_logs_pre_6 = []
        # objective_logs_pre_7 = []
        # objective_logs_pre_8 = []
        # objective_logs_pre_9 = []
        # objective_logs_pre_10 = []
        # objective_logs_pre_11 = []

        # print("len : ", len(trajectory_0), len(trajectory_1), len(trajectory_2), len(trajectory_3))


        for i, tau in enumerate(trajectory_0):
            objective_logs_pre_0.append(list(np.array(tau['logs']).sum(axis=0)))
        for i, tau in enumerate(trajectory_1):
            objective_logs_pre_1.append(list(np.array(tau['logs']).sum(axis=0)))
        for i, tau in enumerate(trajectory_2):
            objective_logs_pre_2.append(list(np.array(tau['logs']).sum(axis=0)))
        # for i, tau in enumerate(trajectory_3):
        #     objective_logs_pre_3.append(list(np.array(tau['logs']).sum(axis=0)))
        # for i, tau in enumerate(trajectory_4):
        #     objective_logs_pre_4.append(list(np.array(tau['logs']).sum(axis=0)))
        # for i, tau in enumerate(trajectory_5):
        #     objective_logs_pre_5.append(list(np.array(tau['logs']).sum(axis=0)))
        # for i, tau in enumerate(trajectory_6):
        #     objective_logs_pre_6.append(list(np.array(tau['logs']).sum(axis=0)))
        # for i, tau in enumerate(trajectory_7):
        #     objective_logs_pre_7.append(list(np.array(tau['logs']).sum(axis=0)))
        # for i, tau in enumerate(trajectory_8):
        #     objective_logs_pre_8.append(list(np.array(tau['logs']).sum(axis=0)))
        # for i, tau in enumerate(trajectory_9):
        #     objective_logs_pre_9.append(list(np.array(tau['logs']).sum(axis=0)))
        # for i, tau in enumerate(trajectory_10):
        #     objective_logs_pre_10.append(list(np.array(tau['logs']).sum(axis=0)))
        # for i, tau in enumerate(trajectory_11):
        #     objective_logs_pre_11.append(list(np.array(tau['logs']).sum(axis=0)))

        return np.array(objective_logs_pre_0), np.array(objective_logs_pre_1), np.array(objective_logs_pre_2)
            # , np.array(objective_logs_pre_3), \
            #    np.array(objective_logs_pre_4), np.array(objective_logs_pre_5), np.array(objective_logs_pre_6), np.array(objective_logs_pre_7), \
            #    np.array(objective_logs_pre_8), np.array(objective_logs_pre_9), np.array(objective_logs_pre_10), np.array(objective_logs_pre_11)



    def vector_reward(self):
        vector_reward = []
        for i, tau in enumerate(self.trajectories):
            vector_reward.append(list(np.array(tau['experts']).sum(axis=0)))

        #print("objective_log : ", vector_reward)

        res = np.array(vector_reward)

        return [res[:, 0].mean(), res[:, 1].mean()]