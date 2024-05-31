import copy

from tqdm import tqdm

from adaption.auxiliary import monte_carlo_search, monte_carlo_search_baseline, find_closest_preference
from adaption.generalization_ppo.baseline.ppo_vanila_v import PPO
from envs.in_use.gym_wrapper import GymWrapper, VecEnv
from envs.in_use.v_3 import v_3_idx_to_scalar, V_3_LEFT_REGION, V_3_RIGHT_REGION
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

    count = 0

    obj_logs = []
    obj_returns = []
    item_A_left = []
    item_A_right = []
    item_B_left = []
    item_B_right = []
    number_of_transition = -1
    number_of_transition_total = []
    item_A_left_ratio = []
    item_A_right_ratio = []
    item_B_left_ratio = []
    item_B_right_ratio = []
    current_position = ''
    last_position = ''
    obj_scalarized_reward= []
    obj_accumulative = []
    preference_list = []

    current_ppo = ppo

    baseline_preferences = [[1.0, 0.0], [0.9, 0.1], [0.7, 0.3], [0.5, 0.5],
                            [0.2, 0.8], [0.0, 1.0]]

    while count < n_eval:

        # preference detection.

        # create new environment
        position_state_env = states[1]
        env_x, env_y = region_check(position_state_env)
        position_env = v_3_idx_to_scalar(env_x, env_y)
        obj_A_row_indices, obj_A_col_indices = np.where(np.array(states[2]) == 1)
        obj_A_positions = [(row, col) for row, col in zip(obj_A_row_indices, obj_A_col_indices)]
        obj_B_row_indices, obj_B_col_indices = np.where(np.array(states[3]) == 1)
        obj_B_positions = [(row, col) for row, col in zip(obj_B_row_indices, obj_B_col_indices)]

        msg = {"item_A_pos" : obj_A_positions, "item_B_pos" : obj_B_positions, "P_pos" : (env_x, env_y)}

        weight_vectors_tensor = monte_carlo_search_baseline(ppo, 5, 5, msg, config)
        states_tensor = states_tensor.unsqueeze(0)

        preference = find_closest_preference(weight_vectors_tensor.tolist()[0], baseline_preferences)

        preference_str = '_'.join(map(str, preference))  # This will turn [1, 0] into "1_0"
        filename = f"../ppo_model/baseline/v_3_25_steps/used/_v3_25_steps_{preference_str}.pt"

        current_ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
        current_ppo.load_state_dict(torch.load(filename))

        print("estimate preference : ", weight_vectors_tensor, ' closest : ', preference_str)

        if weight_vectors_tensor.tolist() not in preference_list:
            preference_list.append(weight_vectors_tensor.tolist())

        if position_env in V_3_LEFT_REGION:
            actions, log_probs = current_ppo.act(states_tensor)
            next_states, reward, done, info = env.step(actions)

            print(weight_vectors_tensor.tolist()[0], "  ", reward, "   ", preference_str)

            scalarized_rewards = sum([(weight_vectors_tensor.tolist()[0])[i] * reward[i] for i in range(len(reward))])

            obj_logs.append(reward)
            obj_scalarized_reward.append(scalarized_rewards)
            #env.render(mode='human')

            item_A_left.append(info['left_item_A'])
            item_B_left.append(info['left_item_B'])
            current_position = 'LEFT'

        elif position_env in V_3_RIGHT_REGION:
            actions, log_probs = current_ppo.act(states_tensor)
            next_states, reward, done, info = env.step(actions)

            print(weight_vectors_tensor.tolist()[0], "  ", reward, "   ", preference_str)

            scalarized_rewards = sum([(weight_vectors_tensor.tolist()[0])[i] * reward[i] for i in range(len(reward))])

            obj_logs.append(reward)
            obj_scalarized_reward.append(scalarized_rewards)
            #env.render(mode='human')

            item_A_right.append(info['right_item_A'])
            item_B_right.append(info['right_item_B'])
            current_position = 'RIGHT'

        else:
            print("error")

        if current_position != last_position:
            #print("transition change")
            number_of_transition += 1
            last_position = current_position

        if done:

            item_A_left_max = info['item_A_left_max']
            item_A_right_max = info['item_A_right_max']

            item_B_left_max = info['item_B_left_max']
            item_B_right_max = info['item_B_right_max']

            # print(item_A_left_max + item_A_right_max)
            # print(item_B_left_max + item_B_right_max)

            next_states = env.reset()

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

    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    # ppo.load_state_dict(torch.load('../ppo_model/generalization/v_3/v_3_25_steps_2023-12-12_00-53-34_used.pt'))
    #

    # ppo_0 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    #
    # ppo_0.load_state_dict(
    #     torch.load('../pre_trained_model/_v4_[1,0]_2023-11-20_15-41-56.pt'))

    # ppo_1 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    #
    # ppo_1.load_state_dict(
    #     torch.load('../pre_trained_model/_v4_[0,1]_2023-11-20_18-08-50.pt'))



    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o= evaluate_ppo(ppo, weight_vectors_tensor, config, n_eval=200)

    print("obj_means : ", a, "  obj_std : ", b)
    print("item_A_left_mean_ratio : ", c, " item_A_left_std : ", d)
    print("item_A_right_mean_ratio : ", e, " item_A_right_std : ", f)
    print("item_B_left_mean_ratio : ", g, " item_B_left_std : ", h)
    print("item_B_right_mean_ratio ： ", i, " item_B_right_std : ", j)
    print("number of transition mean : ", k, " number of transition std : ", l)
    print("return means : ", m, "  ", "return std  :", n)
    print("preference list : ", o)
