import copy

from tqdm import tqdm

from adaption.auxiliary import preference_context_generate, preference_Detect
from envs.in_use.gym_wrapper import GymWrapper, VecEnv
from envs.in_use.v_4 import idx_to_scalar, LEFT_REGION, RIGHT_REGION, get_contexts, LEFT_REGION_1, TRANSITION_REGION_1, \
    RIGHT_REGION_1
from adaption.ppo import *
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


def evaluate_ppo(ppo, preference, config, n_eval=1000):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards
    """
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    weight_vectors = [[0.5, 0.5]]
    weight_vectors_tensor = torch.tensor(weight_vectors)
    weight_vectors_tensor = weight_vectors_tensor.to(device)
    latest_preference = preference

    obj_logs = []
    obj_returns = []
    item_A_left = []
    item_A_right = []
    item_B_left = []
    item_B_right = []
    number_of_transition = 0
    number_of_transition_total = []
    item_A_left_ratio = []
    item_A_right_ratio = []
    item_B_left_ratio = []
    item_B_right_ratio = []
    current_position = ''
    last_position = ''
    trajectories = []
    trajectory = []

    for t in range(n_eval):

        # preference detection.

        # create new environment
        position_state_env = states[1]
        env_x, env_y = region_check(position_state_env)
        position_env = idx_to_scalar(env_x, env_y)
        obj_A_row_indices, obj_A_col_indices = np.where(np.array(states[2]) == 1)
        obj_A_positions = [(row, col) for row, col in zip(obj_A_row_indices, obj_A_col_indices)]
        obj_B_row_indices, obj_B_col_indices = np.where(np.array(states[3]) == 1)
        obj_B_positions = [(row, col) for row, col in zip(obj_B_row_indices, obj_B_col_indices)]

        msg = {"item_A_pos" : obj_A_positions, "item_B_pos" : obj_B_positions, "P_pos" : (env_x, env_y)}

        simulated_env = GymWrapper(config.env_id)
        simulated_states = simulated_env.reset(msg)
        simulated_states_tensor = torch.tensor(simulated_states).float().to(device)
        estimated_trajectory = []
        for n in range(1):
            estimated_trajectory.append(simulated_states)

            simulated_states_tensor = simulated_states_tensor.unsqueeze(0)
            contexts = preference_context_generate(config.n_workers, 2, weight_vectors_tensor)

            simulated_states_augmentation = torch.cat((simulated_states_tensor, contexts), dim=1)
            actions, log_probs = ppo.act(simulated_states_augmentation, weight_vectors_tensor)
            simulated_next_states, simulated_reward, simulated_done, simulated_info = simulated_env.step(actions)

            # Prepare state input for next time step
            simulated_states = simulated_next_states.copy()
            simulated_states_tensor = torch.tensor(simulated_states).float().to(device)

        count_A = 0
        count_B = 0
        # estimate the preference
        for i, tau_states in enumerate(estimated_trajectory):
            position_state = tau_states[1]
            x, y = region_check(position_state)
            position = idx_to_scalar(x, y)
            if position in LEFT_REGION:
                count_A += 1
            elif position in RIGHT_REGION:
                count_B += 1

        total_counts = count_A + count_B
        weights_A = count_A / total_counts
        weights_B = count_B / total_counts

        weight_vectors = [[weights_A, weights_B]]
        print("weight : ", weight_vectors)
        weight_vectors_tensor = torch.tensor(weight_vectors)
        weight_vectors_tensor = weight_vectors_tensor.to(device)

        states_tensor = states_tensor.unsqueeze(0)

        if position_env in LEFT_REGION:
            weight_vectors_tensor = torch.tensor(weight_vectors).to(device)
            contexts = preference_context_generate(config.n_workers, 2, weight_vectors_tensor)
            states_augmentation = torch.cat((states_tensor, contexts), dim=1)
            actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
            next_states, reward, done, info = env.step(actions)
            #env.render(mode='human')

            item_A_left.append(info['left_item_A'])
            item_B_left.append(info['left_item_B'])
            current_position = 'LEFT'

        elif position_env in RIGHT_REGION:
            weight_vectors_tensor = torch.tensor(weight_vectors).to(device)
            contexts = preference_context_generate(config.n_workers, 2, weight_vectors_tensor)
            states_augmentation = torch.cat((states_tensor, contexts), dim=1)
            actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
            next_states, reward, done, info = env.step(actions)
            #env.render(mode='human')

            item_A_right.append(info['right_item_A'])
            item_B_right.append(info['right_item_B'])
            current_position = 'RIGHT'

        else:
            print("error")

        if current_position != last_position:
            number_of_transition += 1
            last_position = current_position

        obj_logs.append(reward)

        if done:
            trajectories.append(trajectory)

            item_A_left_max = info['item_A_left_max']
            item_A_right_max = info['item_A_right_max']

            item_B_left_max = info['item_B_left_max']
            item_B_right_max = info['item_B_right_max']

            next_states = env.reset()

            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)
            obj_logs = []

            if item_A_right_max != 0:
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
           np.array(number_of_transition_total).mean(), np.array(number_of_transition_total).std()


if __name__ == '__main__':
    wandb.init(project='EVALUATE', config={
        'env_id': 'v_4',
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
    env = VecEnv(config.env_id, config.n_workers)
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
    ppo.load_state_dict(torch.load('../ppo_model/conditional_ppo/_v4_2023-11-21_00-53-37.pt'))


    # ppo_0 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    #
    # ppo_0.load_state_dict(
    #     torch.load('../pre_trained_model/_v4_[1,0]_2023-11-20_15-41-56.pt'))

    # ppo_1 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    #
    # ppo_1.load_state_dict(
    #     torch.load('../pre_trained_model/_v4_[0,1]_2023-11-20_18-08-50.pt'))



    a, b, c, d, e, f, g, h, i, j, k, l= evaluate_ppo(ppo, weight_vectors_tensor, config, n_eval=1000)

    print("obj_means : ", a, "  obj_std : ", b)
    print("item_A_left_mean_ratio : ", c, " item_A_left_std : ", d)
    print("item_A_right_mean_ratio : ", e, " item_A_right_std : ", f)
    print("item_B_left_mean_ratio : ", g, " item_B_left_std : ", h)
    print("item_B_right_mean_ratio ： ", i, " item_B_right_std : ", j)
    print("number of transition mean : ", k, " number of transition std : ", l)
