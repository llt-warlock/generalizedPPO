from tqdm import tqdm

from envs.in_use.gym_wrapper import GymWrapper
from envs.in_use.v_2 import idx_to_scalar
from framework.ppo import *
import torch

import numpy as np
import wandb

def region_check(player):
    for i, row in enumerate(player):
        for j, value in enumerate(row):
            if value == 1:
                #print(f"Value 1 is found in row {i} and column {j}")
                return (i, j)



def evaluate_ppo(ppo_primary, ppo_expert, config, n_eval=1000):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards
    """
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    obj_logs = []
    obj_returns = []
    item_A_left = []
    item_A_right = []
    item_B_left = []
    item_B_right = []

    item_A_left_total = []
    item_A_right_total = []
    item_B_left_total = []
    item_B_right_total = []

    for t in range(n_eval):

        x=region_check(states[1])[0]
        y=region_check(states[1])[1]
        position = idx_to_scalar(x, y)
        if position in region_left:
            expert_actions, expert_log_probs = ppo_primary.act(states_tensor)
            next_states, reward, done, info = env.step(expert_actions)
            item_A_left.append(info['left_item_A'])
            item_B_left.append(info['left_item_B'])

        elif position in region_right:
            primary_actions, primary_log_probs = ppo_primary.act(states_tensor)
            next_states, reward, done, info = env.step(primary_actions)
            item_A_right.append(info['right_item_A'])
            item_B_right.append(info['right_item_B'])

        else:
            print("error")

        obj_logs.append(reward)


        if done:
            next_states = env.reset()

            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)
            obj_logs = []

            item_A_left_total.append(np.array(item_A_left).sum())
            item_A_right_total.append(np.array(item_A_right).sum())
            item_B_left_total.append(np.array(item_B_left).sum())
            item_B_right_total.append(np.array(item_B_right).sum())

            item_A_left = []
            item_A_right = []
            item_B_left = []
            item_B_right = []


        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    obj_returns = np.array(obj_returns)
    obj_means = obj_returns.mean(axis=0)
    obj_std = obj_returns.std(axis=0)

    item_A_left_total = np.array(item_A_left_total)
    item_A_left_mean = item_A_left_total.mean()
    item_A_left_std = item_A_left_total.std()

    item_A_right_total = np.array(item_A_right_total)
    item_A_right_mean = item_A_right_total.mean()
    item_A_right_std = item_A_right_total.std()

    item_B_left_total = np.array(item_B_left_total)
    item_B_left_mean = item_B_left_total.mean()
    item_B_left_std = item_B_left_total.std()

    item_B_right_total = np.array(item_B_right_total)
    item_B_right_mean = item_B_right_total.mean()
    item_B_right_std = item_B_right_total.std()


    return list(obj_means), list(obj_std), \
           item_A_left_mean, item_A_left_std, \
           item_A_right_mean,item_A_right_std, \
           item_B_left_mean, item_B_left_std, \
           item_B_right_mean, item_B_right_std


if __name__ == '__main__':
    wandb.init(project='EVALUATE', config={
        'env_id': 'v_2',
        'env_steps': 7e6,
        'batchsize_discriminator': 512,
        'batchsize_ppo': 32,
        'n_workers': 16,
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

    ppo_expert_A = PPO_CNN(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)
    ppo_expert_B = PPO_CNN(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)

    #ppo.load_state_dict(torch.load('../ppo_agent/use/new_test_1_t_new_version_10_10_32_expert_[0, 0, 1, 1].pt'))
    ppo_expert_A.load_state_dict(torch.load('../../saved_models/td_use/airl_policy_v_2_without_primary_2000_[1,0].pt'))
    ppo_expert_B.load_state_dict(torch.load('../../saved_models/td_use/airl_policy_v_2_without_primary_2000_[0,1].pt'))

    a,b,c,d,e,f,g,h,i,j = evaluate_ppo(ppo_expert_A, ppo_expert_B, config)

    print("obj_means : ", a, "  obj_std : ", b)
    print("item_A_left_mean : ", c, " item_A_left_std : ", d)
    print("item_A_right_mean : ", e, " item_A_right_std : ", f)
    print("item_B_left_mean : ", g, " item_B_left_std : ", h)
    print("item_B_right_mean ： ", i, " item_B_right_std : ", j)

