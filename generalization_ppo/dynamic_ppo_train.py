# Use GPU if available
import numpy as np
import torch
import datetime

from adaption.auxiliary import preference_context_generate
from adaption.generalization_ppo.evaluation import evaluate_two_obj_baseline, evaluate_three_obj_baseline
from adaption.generalization_ppo.find_position import find_agent_positions, convert_positions_to_scalars, \
    convert_to_region
from envs.in_use import gym_wrapper
import wandb
from adaption.generalization_ppo.gneralized_ppo import PPO_CNN, update_policy
from adaption.generalization_ppo.replay_buffer import TrajectoryDataset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Use GPU if available
from envs.in_use.gym_wrapper import VecEnv
from tqdm import tqdm

def region_check(player):
    # print("player : ", player)
    for i, row in enumerate(player):
        for j, value in enumerate(row):
            if value == 1:
                # print(f"Value 1 is found in row {i} and column {j}")
                # print(" i : ", i, " j : ", j)
                return (i, j)


if __name__ == '__main__':

    # Fetch ratio args
    #parser = argparse.ArgumentParser(description='Preference Lambda.')
    #parser.add_argument('--lambd', nargs='+', type=float)
    #args = parser.parse_args()

    # Init WandB & Parameters
    wandb.init(project='PPO', config={
        'env_id': 'v_3',
        'env_steps': 7e6,
        'batchsize_ppo': 32,
        'n_workers': 2,
        'lr_ppo': 3e-4,
        'GAE_lambda': 0.999,
        'entropy_reg': 0.1,
        'lambd': [0, 0, 1, 1],
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


    print("state shape : ", state_shape, " in channel : ", in_channels)

    # Initialize Models
    ppo = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=2, contexts=2).to(device)

    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo, eps=1e-5)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    max_step = int(config.env_steps / config.n_workers)
    try:

        print("device : ", device)

        for t in tqdm(range(max_step)):

            lr_a_now = config.lr_ppo * (1 - t / max_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_a_now

            # create new environment
            # position_state_env = states[1]
            #
            # print("position : ", states)

            print("state : ", states)

            position = find_agent_positions(states_tensor)

            print("position : ", position)

            prediction_scalar = convert_positions_to_scalars(position)

            print("position scalar : ", prediction_scalar)

            preference = convert_to_region(prediction_scalar)
            print("preference : ", preference)

            weight_vectors_tensor = torch.tensor(preference).float().to(device)
            # weight_vectors = np.repeat(weight_vectors, 4, axis=

            contexts = preference_context_generate(config.n_workers, 2, weight_vectors_tensor)

            states_augmentation = torch.cat((states_tensor, contexts), dim=1)

            actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)

            next_states, rewards, done, info = vec_env.step(actions)
            scalarized_rewards = [sum([preference[i] * r[i] for i in range(len(r))]) for r in rewards]


            train_ready = dataset.write_tuple(states, next_states, actions, scalarized_rewards, done, log_probs, rewards, info, weight_vectors_tensor)

            if train_ready:
                update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                              entropy_reg=config.entropy_reg, GAE_lambda=config.GAE_lambda)

                # objective_logs = dataset.log_objectives()

                # for i in range(objective_logs.shape[1]):
                #     wandb.log({preference + 'Obj_' + str(i): objective_logs[:, i].mean()})
                # for ret in dataset.log_returns():
                #     wandb.log({preference + 'Returns': ret})
                dataset.reset_trajectories()

            if t % 1000 == 0:
                evaluate_two_obj_baseline(ppo, config)
                #evaluate_three_obj_baseline(ppo, config)

            # Prepare state input for next time step
            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # torch.save(ppo.state_dict(), '../../ppo_model/baseline/v_3_25_steps/_v3_25_steps_' + preference + '_' + timestamp + '.pt')
        # torch.save(ppo.state_dict(),
        #            '../../ppo_model/baseline/v_5_40_steps/_v5_40_steps_' + preference + '_' + timestamp + '.pt')

    except KeyboardInterrupt:
        print("Manual interruption detected...")
        # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # torch.save(ppo.state_dict(),
        #            '../../ppo_model/baseline/v_3_25_steps/_v3_25_steps_' + preference + '_' + timestamp + '.pt')
        # torch.save(ppo.state_dict(),
        #            '../../ppo_model/baseline/v_5_40_steps/_v5_40_steps_' + preference + '_' + timestamp + '.pt')
