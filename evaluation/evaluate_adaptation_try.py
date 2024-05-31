import copy

from tqdm import tqdm

from adaption.auxiliary import monte_carlo_search, preference_context_generate
from adaption.generalization_ppo.generalized_ppo_train_online import training
from adaption.replay_buffer import TrajectoryDataset

from envs.in_use.gym_wrapper import GymWrapper, VecEnv
from envs.in_use.v_3 import v_3_idx_to_scalar, V_3_LEFT_REGION, V_3_RIGHT_REGION, PREFERENCE
from adaption.generalization_ppo.gneralized_ppo import PPO_CNN, update_policy
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


def evaluate_ppo(p_ppo, config, p_dataset, p_optimizer):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards
    """
    env_original = GymWrapper(config.env_id)
    states = env_original.reset()
    states_tensor = torch.tensor(states).float().to(device)
    # weight_vectors = [[0.5, 0.5]]
    # weight_vectors_tensor = torch.tensor(weight_vectors)
    # weight_vectors_tensor = weight_vectors_tensor.to(device)
    # latest_preference = preference
    ppo = p_ppo
    dataset = p_dataset
    optimizer = p_optimizer
    number = 0
    max_step = int(config.env_steps / config.n_workers)

    weight_vectors = [[0.5, 0.5]]
    weight_vectors_tensor = torch.tensor(weight_vectors)
    weight_vectors_tensor = weight_vectors_tensor.to(device)

    preference_set = []


    for t in tqdm(range(max_step)):
        print(t)
        lr_a_now = config.lr_ppo * (1 - t / max_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_a_now

        # create new environment
        position_state_env = states[1]
        env_x, env_y = region_check(position_state_env)
        #position_env = v_3_idx_to_scalar(env_x, env_y)

        obj_A_row_indices, obj_A_col_indices = np.where(np.array(states[2]) == 1)
        obj_A_positions = [(row, col) for row, col in zip(obj_A_row_indices, obj_A_col_indices)]
        obj_B_row_indices, obj_B_col_indices = np.where(np.array(states[3]) == 1)
        obj_B_positions = [(row, col) for row, col in zip(obj_B_row_indices, obj_B_col_indices)]

        msg = {"item_A_pos" : obj_A_positions, "item_B_pos" : obj_B_positions, "P_pos" : (env_x, env_y)}

        print(" msg : ", msg)
        print("weight : ", weight_vectors_tensor)
        weight_vectors_tensor = monte_carlo_search(ppo, 5, 5, msg, config, weight_vectors_tensor)

        if weight_vectors_tensor.tolist()[0] not in preference_set:
            preference_set.insert(0, weight_vectors_tensor.tolist()[0])

        print("update preference set : ", preference_set)
        states_tensor = states_tensor.unsqueeze(0)

        contexts = preference_context_generate(1, 2, weight_vectors_tensor)
        states_augmentation = torch.cat((states_tensor, contexts), dim=1)

        print("weight vector : ", weight_vectors_tensor.tolist())
        ppo, dataset, optimizer = training(ppo, preference_set, dataset, optimizer, config, t)


        actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)

        next_states, reward, done, info = env_original.step(actions)

        if done:
            env_original.reset()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)


if __name__ == '__main__':
    wandb.init(project='EVALUATE', config={
        'env_id': 'v_3',
        'env_steps': 8e6,
        'batchsize_ppo': 32,
        'n_workers': 1,
        'env_worker': 16,
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

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    ppo = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo, eps=1e-5)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o= evaluate_ppo(ppo, config, dataset, optimizer)

