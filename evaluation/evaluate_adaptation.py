
from adaption.auxiliary import monte_carlo_search
from envs.in_use.gym_wrapper import GymWrapper, VecEnv
#from envs.in_use.v_3 import v_3_idx_to_scalar, V_3_LEFT_REGION, V_3_RIGHT_REGION
from adaption.generalization_ppo.gneralized_ppo import *
import torch

import numpy as np
import wandb
from IPython import display

from envs.in_use.v_5 import V_5_TOP_REGION, V_5_RIGHT_REGION, V_5_LEFT_REGION, idx_to_scalar, v_5_idx_to_scalar


def region_check(player):
    # print("player : ", player)
    for i, row in enumerate(player):
        for j, value in enumerate(row):
            if value == 1:
                # print(f"Value 1 is found in row {i} and column {j}")
                # print(" i : ", i, " j : ", j)
                return (i, j)


def evaluate_ppo(ppo, preference, config, n_eval=200):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards
    """
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    weight_vectors = [[0.33, 0.33, 0.34]]
    weight_vectors_tensor = torch.tensor(weight_vectors)
    weight_vectors_tensor = weight_vectors_tensor.to(device)
    latest_preference = preference

    count = 0

    obj_logs = []
    obj_returns = []
    item_A_left = []
    total_A_left = []
    item_A_right = []
    total_A_right = []
    item_A_top = []  ##
    total_A_top = [] ##
    item_B_left = []
    total_B_left = []
    item_B_right = []
    total_B_right = []
    item_B_top = []  ##
    total_B_top = [] ##
    item_C_left = [] ###
    total_C_left = [] ##
    item_C_right = [] ##
    total_C_right = [] ##
    item_C_top = []  ##
    total_C_top = [] ##

    number_of_transition = -1
    number_of_transition_total = []
    item_A_left_ratio = []
    item_A_right_ratio = []
    item_A_top_ratio = [] #
    item_B_left_ratio = []
    item_B_right_ratio = []
    item_B_top_ratio = [] #
    item_C_left_ratio = []
    item_C_right_ratio = []
    item_C_top_ratio = [] #
    current_position = ''
    last_position = ''
    obj_scalarized_reward= []
    obj_accumulative = []
    preference_list = []

    while count < n_eval:

        # preference detection.

        # create new environment
        position_state_env = states[1]
        env_x, env_y = region_check(position_state_env)
        #position_env = v_3_idx_to_scalar(env_x, env_y)
        position_env = v_5_idx_to_scalar(env_x, env_y)
        obj_A_row_indices, obj_A_col_indices = np.where(np.array(states[2]) == 1)
        obj_A_positions = [(row, col) for row, col in zip(obj_A_row_indices, obj_A_col_indices)]
        obj_B_row_indices, obj_B_col_indices = np.where(np.array(states[3]) == 1)
        obj_B_positions = [(row, col) for row, col in zip(obj_B_row_indices, obj_B_col_indices)]

        obj_C_row_indices, obj_C_col_indices = np.where(np.array(states[4]) == 1)
        obj_C_positions = [(row, col) for row, col in zip(obj_C_row_indices, obj_C_col_indices)]

        #msg = {"item_A_pos" : obj_A_positions, "item_B_pos" : obj_B_positions, "P_pos" : (env_x, env_y)}
        msg = {"item_A_pos": obj_A_positions, "item_B_pos": obj_B_positions, "item_C_pos": obj_C_positions, "P_pos": (env_x, env_y)}

        #print("size : ", weight_vectors_tensor.size())

        #print("weight vector : ", weight_vectors_tensor)
        weight_vectors_tensor = monte_carlo_search(ppo, 5, 1, msg, config, weight_vectors_tensor)
        states_tensor = states_tensor.unsqueeze(0)
        print("pre : ", weight_vectors_tensor)
        if weight_vectors_tensor.tolist() not in preference_list:
            preference_list.append(weight_vectors_tensor.tolist())

        if position_env in V_5_LEFT_REGION:

            contexts = preference_context_generate(config.n_workers, 3, weight_vectors_tensor)

            states_augmentation = torch.cat((states_tensor, contexts), dim=1)


            actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
            next_states, reward, done, info = env.step(actions)
            scalarized_rewards = sum([(weight_vectors_tensor.tolist()[0])[i] * reward[i] for i in range(len(reward))])

            obj_logs.append(reward)
            obj_scalarized_reward.append(scalarized_rewards)
            #env.render(mode='human')

            item_A_left.append(info['left_item_A'])
            item_B_left.append(info['left_item_B'])
            item_C_left.append(info['left_item_C'])
            current_position = 'LEFT'

        elif position_env in V_5_RIGHT_REGION:

            contexts = preference_context_generate(config.n_workers, 3, weight_vectors_tensor)
            states_augmentation = torch.cat((states_tensor, contexts), dim=1)
            actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
            next_states, reward, done, info = env.step(actions)
            scalarized_rewards = sum([(weight_vectors_tensor.tolist()[0])[i] * reward[i] for i in range(len(reward))])

            obj_logs.append(reward)
            obj_scalarized_reward.append(scalarized_rewards)
            #env.render(mode='human')

            item_A_right.append(info['right_item_A'])
            item_B_right.append(info['right_item_B'])
            item_C_right.append(info['right_item_C'])
            current_position = 'RIGHT'

        elif position_env in V_5_TOP_REGION:

            contexts = preference_context_generate(config.n_workers, 3, weight_vectors_tensor)
            states_augmentation = torch.cat((states_tensor, contexts), dim=1)
            actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
            next_states, reward, done, info = env.step(actions)
            scalarized_rewards = sum([(weight_vectors_tensor.tolist()[0])[i] * reward[i] for i in range(len(reward))])

            obj_logs.append(reward)
            obj_scalarized_reward.append(scalarized_rewards)
            #env.render(mode='human')

            item_A_top.append(info['top_item_A'])
            item_B_top.append(info['top_item_B'])
            item_C_top.append(info['top_item_C'])
            current_position = 'TOP'

        else:
            print("error")

        if current_position != last_position:
            #print("transition change")
            number_of_transition += 1
            last_position = current_position

        if done:

            item_A_left_max = info['item_A_left_max']
            item_A_right_max = info['item_A_right_max']
            item_A_top_max = info['item_A_top_max']

            item_B_left_max = info['item_B_left_max']
            item_B_right_max = info['item_B_right_max']
            item_B_top_max = info['item_B_top_max']

            item_C_left_max = info['item_C_left_max']
            item_C_right_max = info['item_C_right_max']
            item_C_top_max = info['item_C_top_max']

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
            if item_A_top_max != 0:
                item_A_top_ratio.append(np.array(item_A_top).sum() / item_A_top_max)
                item_A_top_max = 0
            if item_B_left_max != 0:
                item_B_left_ratio.append(np.array(item_B_left).sum() / item_B_left_max)
                item_B_left_max = 0
            if item_B_right_max != 0:
                item_B_right_ratio.append(np.array(item_B_right).sum() / item_B_right_max)
                item_B_right_max = 0
            if item_B_top_max != 0:
                item_B_top_ratio.append(np.array(item_B_top).sum() / item_B_top_max)
                item_B_top_max = 0
            if item_C_left_max != 0:
                item_C_left_ratio.append(np.array(item_C_left).sum() / item_C_left_max)
                item_C_left_max = 0
            if item_C_right_max != 0:
                item_C_right_ratio.append(np.array(item_C_right).sum() / item_C_right_max)
                item_C_right_max = 0
            if item_C_top_max != 0:
                item_C_top_ratio.append(np.array(item_C_top).sum() / item_C_top_max)
                item_C_top_max = 0


            total_A_left.append(np.array(item_A_left).sum())
            total_A_right.append(np.array(item_A_right).sum())
            total_A_top.append(np.array(item_A_top).sum())
            total_B_left.append(np.array(item_B_left).sum())
            total_B_right.append(np.array(item_B_right).sum())
            total_B_top.append(np.array(item_B_top).sum())

            total_C_left.append(np.array(item_C_left).sum())
            total_C_right.append(np.array(item_C_right).sum())
            total_C_top.append(np.array(item_C_top).sum())

            number_of_transition_total.append(number_of_transition)

            item_A_left = []
            item_A_right = []
            item_A_top = []
            item_B_left = []
            item_B_right = []
            item_B_top  = []
            item_C_left = []
            item_C_right = []
            item_C_top  = []
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

    item_A_top_ratio = np.array(item_A_top_ratio)
    item_A_top_mean = item_A_top_ratio.mean()
    item_A_top_std = item_A_top_ratio.std()

    item_B_left_ratio = np.array(item_B_left_ratio)
    item_B_left_mean = item_B_left_ratio.mean()
    item_B_left_std = item_B_left_ratio.std()

    item_B_right_ratio = np.array(item_B_right_ratio)
    item_B_right_mean = item_B_right_ratio.mean()
    item_B_right_std = item_B_right_ratio.std()

    item_B_top_ratio = np.array(item_B_top_ratio)
    item_B_top_mean = item_B_top_ratio.mean()
    item_B_top_std = item_B_top_ratio.std()

    item_C_left_ratio = np.array(item_C_left_ratio)
    item_C_left_mean = item_C_left_ratio.mean()
    item_C_left_std = item_C_left_ratio.std()

    item_C_right_ratio = np.array(item_C_right_ratio)
    item_C_right_mean = item_C_right_ratio.mean()
    item_C_right_std = item_C_right_ratio.std()

    item_C_top_ratio = np.array(item_C_top_ratio)
    item_C_top_mean = item_C_top_ratio.mean()
    item_C_top_std = item_C_top_ratio.std()

    total_A_left = np.array(total_A_left)
    total_A_left_mean = total_A_left.mean()
    total_A_left_std = total_A_left.std()

    total_A_right = np.array(total_A_right)
    total_A_right_mean = total_A_right.mean()
    total_A_right_std = total_A_right.std()

    total_A_top = np.array(total_A_top)
    total_A_top_mean = total_A_top.mean()
    total_A_top_std = total_A_top.std()

    total_B_left = np.array(total_B_left)
    total_B_left_mean = total_B_left.mean()
    total_B_left_std = total_B_left.std()

    total_B_right = np.array(total_B_right)
    total_B_right_mean = total_B_right.mean()
    total_B_right_std = total_B_right.std()

    total_B_top = np.array(total_B_top)
    total_B_top_mean = total_B_top.mean()
    total_B_top_std = total_B_top.std()

    total_C_left = np.array(total_C_left)
    total_C_left_mean = total_C_left.mean()
    total_C_left_std = total_C_left.std()

    total_C_right = np.array(total_C_right)
    total_C_right_mean = total_C_right.mean()
    total_C_right_std = total_C_right.std()

    total_C_top = np.array(total_C_top)
    total_C_top_mean = total_C_top.mean()
    total_C_top_std = total_C_top.std()


    return list(obj_means), list(obj_std), \
           item_A_left_mean, item_A_left_std, \
           item_A_right_mean, item_A_right_std, \
           item_A_top_mean, item_A_top_std, \
           item_B_left_mean, item_B_left_std, \
           item_B_right_mean, item_B_right_std, \
           item_B_top_mean, item_B_top_std, \
           item_C_left_mean, item_C_left_std, \
           item_C_right_mean, item_C_right_std, \
           item_C_top_mean, item_C_top_std, \
           np.array(number_of_transition_total).mean(), np.array(number_of_transition_total).std(), \
           obj_accu_mean, obj_accu_std, \
           preference_list, \
           total_A_left_mean, total_A_left_std, total_A_right_mean, total_A_right_std, total_A_top_mean, total_A_top_std, \
           total_B_left_mean, total_B_left_std, total_B_right_mean, total_B_right_std, total_B_top_mean, total_B_top_std,\
           total_C_left_mean, total_C_left_std, total_C_right_mean, total_C_right_std, total_C_top_mean, total_C_top_std



if __name__ == '__main__':
    wandb.init(project='EVALUATE', config={
        'env_id': 'v_5',
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
        'n_objs': 3
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

    weight_vectors = [[0.33, 0.33, 0.34]]
    weight_vectors_tensor = torch.tensor(weight_vectors)
    weight_vectors_tensor = weight_vectors_tensor.to(device)

    ppo = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=3, contexts=3).to(device)
    #ppo.load_state_dict(torch.load('../ppo_model/generalization/v_3_25_steps2023-12-05_01-15-26.pt'))
    ppo.load_state_dict(torch.load('../ppo_model/generalization/v_5/sample_efficiency/v_5_40_steps_2024-01-16_19-33-03_used.pt'))

    # ppo_0 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    #
    # ppo_0.load_state_dict(
    #     torch.load('../pre_trained_model/_v4_[1,0]_2023-11-20_15-41-56.pt'))

    # ppo_1 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    #
    # ppo_1.load_state_dict(
    #     torch.load('../pre_trained_model/_v4_[0,1]_2023-11-20_18-08-50.pt'))



    objmean, obj_std, \
    lA_mean, lA_std, \
    rA_mean, rA_std,\
    tA_mean, tA_std, \
    lB_mean, lB_std, \
    rB_mean, rB_std, \
    tB_mean, tB_std, \
    lC_mean, lC_std, \
    rC_mean, rC_std, \
    tC_mean, tC_std, \
    transition_mean, transition_std,\
    return_mean, return_std,\
    preference_list,\
    total_lA_mean, total_lA_std,\
        total_rA_mean, total_rA_std,\
        total_tA_mean, total_tA_std, \
    total_lB_mean, total_lB_std, \
    total_rB_mean, total_rB_std, \
    total_tB_mean, total_tB_std, \
    total_lC_mean, total_lC_std, \
    total_rC_mean, total_rC_std, \
    total_tC_mean, total_tC_std = evaluate_ppo(ppo, weight_vectors_tensor, config, n_eval=200)

    print("obj_means : ", objmean, "  obj_std : ", obj_std)
    print("item_A_left_mean_ratio : ", lA_mean, " item_A_left_std : ", lA_std)
    print("item_A_right_mean_ratio : ", rA_mean, " item_A_right_std : ", rA_std)
    print("item_A_top_mean_ratio : ", tA_mean, " item_A_top_std : ", tA_std)
    print("item_B_left_mean_ratio : ", lB_mean, " item_B_left_std : ", lB_std)
    print("item_B_right_mean_ratio ： ", rB_mean, " item_B_right_std : ", rB_std)
    print("item_B_top_mean_ratio ： ", tB_mean, " item_B_top_std : ", tB_std)
    print("item_C_left_mean_ratio : ", lC_mean, " item_C_left_std : ", lC_std)
    print("item_C_right_mean_ratio ： ", rC_mean, " item_C_right_std : ", rC_std)
    print("item_C_top_mean_ratio ： ", tC_mean, " item_C_top_std : ", tC_std)
    print("number of transition mean : ", transition_mean, " number of transition std : ", transition_std)
    print("return means : ", return_mean, "  ", "return std  :", return_std)
    print("preference list : ", len(preference_list))
    print("left A : ", total_lA_mean, " ", total_lA_std, "  right A : ",total_rA_mean, " ",total_rA_std, " top A : ", total_tA_mean, " ", total_tA_std)
    print("left B : ", total_lB_mean, " ", total_lB_std, "  right B : ", total_rB_mean, " ", total_rB_std, " top B : ", total_tB_mean, " ",total_tB_std)
    print("left C : ", total_lC_mean, " ", total_lC_std, "  right C : ", total_rC_mean, " ", total_rC_std, " top C : ", total_tC_mean, " ", total_tC_std)
