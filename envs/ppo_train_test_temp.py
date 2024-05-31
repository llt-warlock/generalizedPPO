from tqdm import tqdm
from ppo import *
import torch
from envs.gym_wrapper import *
import wandb
import argparse


# Use GPU if available
from framework.ppo import TrajectoryDataset, PPO, update_policy

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
        'batchsize_ppo': 12,
        'n_workers': 12,
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


    ini_pos = []
    for i in range(config.n_workers):
        player_initia_pos_i = np.where(states[i][1] == 1)
        row_i = player_initia_pos_i[0][0]
        col_i = player_initia_pos_i[1][0]
        pos_i = (row_i, col_i)
        print("position :", pos_i)
        ini_pos.append(pos_i)

    print("ini_pos : ", ini_pos)


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

        actions, log_probs = ppo.act(states_tensor)

        next_states, rewards, done, info = vec_env.step(actions)
        #print("state : ", next_states[0], " p : ", info[0]['P_pos'])
        scalarized_rewards = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]


        #print("reward : ", scalarized_rewards)
        train_ready = dataset.write_tuple(states, actions, scalarized_rewards, done, log_probs, rewards)
        #print("train_ready : ", train_ready)

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

    vec_env.close()
    torch.save(ppo.state_dict(), 'ppo_warehouse_0_' + str(config.lambd) + '.pt')
