import numpy as np
from tqdm import tqdm
from ppo import *
import torch
from envs.gym_wrapper import *
import wandb
import argparse


# Use GPU if available
from framework.ppo import update_policy, TrajectoryDataset, PPO

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # Fetch ratio args
    #parser = argparse.ArgumentParser(description='Preference Lambda.')
    #parser.add_argument('--lambd', nargs='+', type=float)
    #args = parser.parse_args()

    # Init WandB & Parameters
    wandb.init(project='PPO', config={
        'env_id': 'warehouse_0',
        'env_steps': 1e6,
        #'env_steps': 75,
        'batchsize_ppo': 12,
        'n_workers': 2,
        'lr_ppo': 3e-4,
        'entropy_reg': 0.05,
        'lambd': [1., 0.],
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config

    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):

        temp_states = states_tensor
        # prediction
        imaginations = []

        for _ in range(5):
            actions, log_probs = ppo.act(temp_states)
            imag_next_states, rewards, done, info = vec_env.step(actions)
            imaginations.append(info)
            temp_states = torch.tensor(imag_next_states).float().to(device)

        #print("imagination : ", imaginations)

        ###################### analysis ################

        regions = [[0, 0] for _ in range(config.n_workers)] # [ [0,0], [0,0] ]

        for i in imaginations: # [ [[worker]， [worker]]1,  [[worker]， [worker]]2,  [[worker]， [worker]]3, [[worker]， [worker]]4, [[worker]， [worker]]5]

            for j in range(len(i)):  # number of workers

                position = i[j]['P_pos']
                #print("position : ", position)
                if position in i[j]['$']:
                    regions[j][0] += 1
                elif position in i[j]['&']:
                    regions[j][1] += 1


        #print("regions : ", regions)

        weights = [[0., 0.] for _ in range(config.n_workers)]

        for i in range(len(weights)):
            weights[i][0] = regions[i][0]/ 5
            weights[i][1] = regions[i][1] / 5


        ################################################
        #print("weights : ", weights) # [ [0,0], [0,0] ]


        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)
        scalarized_rewards = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]
        #print("scalarized_rewards : ", scalarized_rewards)
        #scalarized_rewards = [sum([weights[r][i] * rewards[r][i] for i in range(len(rewards[r]))]) for r in range(len(rewards))] # number of workers


        train_ready = dataset.write_tuple(states, actions, scalarized_rewards, done, log_probs, rewards)


        if train_ready:
            #print("in train ready")
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            objective_logs = dataset.log_objectives()
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    #vec_env.close()
    torch.save(ppo.state_dict(), 'ppo_v2_' + str(config.lambd) + '.pt')
