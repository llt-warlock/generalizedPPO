import copy

from tqdm import tqdm

from adaption.auxiliary import monte_carlo_search
from envs.in_use.gym_wrapper import GymWrapper, VecEnv
from envs.in_use.v_3 import v_3_idx_to_scalar, V_3_LEFT_REGION, V_3_RIGHT_REGION
from adaption.generalization_ppo.gneralized_ppo import *
import torch

import numpy as np
import wandb
from IPython import display

def region_check(player):
    # print("player : ", player)
    for i, row in enumerate(player):
        for j, value in enumerate(row):
            if value == 1:
                # print(f"Value 1 is found in row {i} and column {j}")
                # print(" i : ", i, " j : ", j)
                return (i, j)

# def get_row_col_indices(layer):
#
#


def evaluate_ppo(ppo, preference, config):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards
    """
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    max_step = int(config.env_steps / config.n_workers)
    for t in tqdm(range(config.max)):

        states_tensor = states_tensor.unsqueeze(0)


        contexts = preference_context_generate(config.n_workers, 2, weight_vectors_tensor)
        states_augmentation = torch.cat((states_tensor, contexts), dim=1)


        actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
        next_states, reward, done, info = env.step(actions)




            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)

            obj_scalarized_reward = np.array(obj_scalarized_reward).sum()
            obj_accumulative.append(obj_scalarized_reward)

            print(count, " :   objlog : ", obj_logs, "  return : ", obj_scalarized_reward, "  transition : ", number_of_transition)

            obj_logs = []
            obj_scalarized_reward = []
            count += 1

            if item_A_left_max != 0:
                item_A_left_ratio.append(np.array(item_A_left).sum() / item_A_left_max)
                item_A_left_max = 0
            if item_A_right_max != 0:
                item_A_right_ratio.append(np.array(item_A_right).sum() / item_A_right_max)
                item_A_right_max = 0
            if item_B_left_max != 0:
                item_B_left_ratio.append(np.array(item_B_left).sum() / item_B_left_max)
                item_B_left_max = 0
            if item_B_right_max != 0:
                item_B_right_ratio.append(np.array(item_B_right).sum() / item_B_right_max)
                item_B_right_max = 0

            number_of_transition_total.append(number_of_transition)

            item_A_left = []
            item_A_right = []
            item_B_left = []
            item_B_right = []
            trajectory = []
            number_of_transition = 0

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    obj_accumulative = np.array(obj_accumulative)
    obj_accu_mean = obj_accumulative.mean()
    obj_accu_std = obj_accumulative.std()

    obj_returns = np.array(obj_returns)
    obj_means = obj_returns.mean(axis=0)
    obj_std = obj_returns.std(axis=0)

    item_A_left_ratio = np.array(item_A_left_ratio)
    item_A_left_mean = item_A_left_ratio.mean()
    item_A_left_std = item_A_left_ratio.std()

    item_A_right_ratio = np.array(item_A_right_ratio)
    item_A_right_mean = item_A_right_ratio.mean()
    item_A_right_std = item_A_right_ratio.std()

    item_B_left_ratio = np.array(item_B_left_ratio)
    item_B_left_mean = item_B_left_ratio.mean()
    item_B_left_std = item_B_left_ratio.std()

    item_B_right_ratio = np.array(item_B_right_ratio)
    item_B_right_mean = item_B_right_ratio.mean()
    item_B_right_std = item_B_right_ratio.std()

    return list(obj_means), list(obj_std), \
           item_A_left_mean, item_A_left_std, \
           item_A_right_mean, item_A_right_std, \
           item_B_left_mean, item_B_left_std, \
           item_B_right_mean, item_B_right_std, \
           np.array(number_of_transition_total).mean(), np.array(number_of_transition_total).std(), \
           obj_accu_mean, obj_accu_std, \
           preference_list


if __name__ == '__main__':
    wandb.init(project='EVALUATE', config={
        'env_id': 'v_3',
        'env_steps': 8e6,
        'batchsize_ppo': 12,
        'n_workers': 1,
        'lr_ppo': 3e-4,
        'entropy_reg': 0.02,
        'lambd': [0, 1, 0],
        'gamma': 0.99,
        'epsilon': 0.1,
        'ppo_epochs': 5,
        'GAE_lambda': 0.98,
        'mini_batch_size': 16,
        'update_frequency': 10,
        'n_objs': 2
    })
    config = wandb.config

    # Initialize Environment
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    dataset = []
    episode = {'states': [], 'actions': [], 'rewards': []}
    episode_cnt = 0

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    weight_vectors = [[0.5, 0.5]]
    weight_vectors_tensor = torch.tensor(weight_vectors)
    weight_vectors_tensor = weight_vectors_tensor.to(device)

    ppo = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    #ppo.load_state_dict(torch.load('../ppo_model/generalization/v_3_25_steps2023-12-05_01-15-26.pt'))
    ppo.load_state_dict(torch.load('../ppo_model/generalization/v_3/v_3_25_steps_2023-12-12_00-53-34_used.pt'))

    # ppo_0 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    #
    # ppo_0.load_state_dict(
    #     torch.load('../pre_trained_model/_v4_[1,0]_2023-11-20_15-41-56.pt'))

    # ppo_1 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    #
    # ppo_1.load_state_dict(
    #     torch.load('../pre_trained_model/_v4_[0,1]_2023-11-20_18-08-50.pt'))



    evaluate_ppo(ppo, weight_vectors_tensor, config, n_eval=200)

