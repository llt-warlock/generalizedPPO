from tqdm import tqdm

from envs.in_use.gym_wrapper import GymWrapper, VecEnv
from envs.in_use.v_4 import idx_to_scalar, LEFT_REGION, RIGHT_REGION, get_contexts
from framework.ppo import *
import torch

import numpy as np
import wandb


def region_check(player):
    # print("player : ", player)
    for i, row in enumerate(player):
        for j, value in enumerate(row):
            if value == 1:
                # print(f"Value 1 is found in row {i} and column {j}")
                # print(" i : ", i, " j : ", j)
                return (i, j)


def evaluate_ppo(expert_A, expert_B, config, n_eval, contexts=None):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards
    """
    env = GymWrapper(config.env_id)
   # vec_env = VecEnv(config.env_id, 1)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)


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

    for t in tqdm(range(n_eval)):
        position_state = states[1]
        x, y = region_check(position_state)
        position = idx_to_scalar(x, y)

        states_tensor = states_tensor.unsqueeze(0)

        if position in LEFT_REGION:
            actions, expert_log_probs = expert_A.act(states_tensor, contexts=contexts)
            next_states, reward, done, info = env.step(actions)
            item_A_left.append(info['left_item_A'])
            item_B_left.append(info['left_item_B'])
            current_position = 'LEFT'
        elif position in RIGHT_REGION:
            actions, primary_log_probs = expert_B.act(states_tensor, contexts=contexts)
            next_states, reward, done, info = env.step(actions)
            item_A_right.append(info['right_item_A'])
            item_B_right.append(info['right_item_B'])
            current_position = 'RIGHT'
        else:
            print("error")

        if current_position != last_position and last_position != '':
            number_of_transition += 1

        obj_logs.append(reward)

        if done:
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
            number_of_transition = 0
            current_position = ''

        # Prepare state input for next time step
        last_position = current_position
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
        'env_steps': 7e6,
        'batchsize_discriminator': 512,
        'batchsize_ppo': 32,
        'n_workers': 8,
        'entropy_reg': 0,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5,
        'GAE_lambda': 0.98
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

    contexts = get_contexts()

    ppo_expert_A = PPO_CNN(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels, contexts=contexts).to(device)
    ppo_expert_B = PPO_CNN(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)

    #ppo.load_state_dict(torch.load('../ppo_agent/use/new_test_1_t_new_version_10_10_32_expert_[0, 0, 1, 1].pt'))
    # ppo_expert_A.load_state_dict(
    #     torch.load('../saved_models/td_use/airl_policy_v_4_1000_[0,1,0]_2023-10-12_11-40-40_use.pt'))
    # ppo_expert_B.load_state_dict(
    #     torch.load('../saved_models/td_use/airl_policy_v_4_1000_[0,0,1]_2023-10-12_15-25-50_use.pt'))

    ppo_expert_A.load_state_dict(
        torch.load('../../ppo_agent/td_use/tb_evaluate/'))


    a, b, c, d, e, f, g, h, i, j, k, l= evaluate_ppo(ppo_expert_A, ppo_expert_A, config, n_eval=1000, contexts=None)

    print("obj_means : ", a, "  obj_std : ", b)
    print("item_A_left_mean_ratio : ", c, " item_A_left_std : ", d)
    print("item_A_right_mean_ratio : ", e, " item_A_right_std : ", f)
    print("item_B_left_mean_ratio : ", g, " item_B_left_std : ", h)
    print("item_B_right_mean_ratio ： ", i, " item_B_right_std : ", j)
    print("number of transition mean : ", k, " number of transition std : ", l)
