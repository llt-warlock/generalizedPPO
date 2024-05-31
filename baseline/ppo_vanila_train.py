# Use GPU if available
import torch
import datetime

from adaption.generalization_ppo.evaluation import evaluate_two_obj_baseline, evaluate_three_obj_baseline
from envs.in_use import gym_wrapper
import wandb
from ppo_vanila_v import PPO, update_policy, TrajectoryDataset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Use GPU if available
from envs.in_use.gym_wrapper import VecEnv
from tqdm import tqdm

def training(preference, weight_vector):

    # Fetch ratio args
    #parser = argparse.ArgumentParser(description='Preference Lambda.')
    #parser.add_argument('--lambd', nargs='+', type=float)
    #args = parser.parse_args()

    # Init WandB & Parameters
    wandb.init(project='PPO', config={
        'env_id': 'v_5',
        'env_steps': 9e6,
        'batchsize_ppo': 32,
        'n_workers': 16,
        'lr_ppo': 3e-4,
        'GAE_lambda': 0.98,
        'entropy_reg': 0.01,
        'lambd': [0, 0, 1, 1],
        'gamma': 0.995,
        'epsilon': 0.1,
        'update_frequency': 1000,
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

    save_count = 0

    print("state shape : ", state_shape, " in channel : ", in_channels)

    # Initialize Models
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo, eps=1e-5)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    max_step = int(config.env_steps / config.n_workers)
    try:


        for t in tqdm(range(max_step)):

            lr_a_now = config.lr_ppo * (1 - t / max_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_a_now

            actions, log_probs = ppo.act(states_tensor)
            next_states, rewards, done, info = vec_env.step(actions)
            #print("reward : ", rewards)

            #print("done : ", done, "  info : ", info)

            scalarized_rewards = [sum([weight_vector[i] * r[i] for i in range(len(r))]) for r in rewards]
            # print("weight : ", weight_vector)
            # print("reward : ",rewards)
            # print("scalarized : ", scalarized_rewards)

            train_ready = dataset.write_tuple(states, next_states, actions, scalarized_rewards, done, log_probs, rewards, info)

            if train_ready:
                update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                              entropy_reg=config.entropy_reg, GAE_lambda=config.GAE_lambda)


                # save_count += 1
                # print("save count : ", save_count)
                # objective_logs = dataset.log_objectives()
                #
                # for i in range(objective_logs.shape[1]):
                #     wandb.log({preference + 'Obj_' + str(i): objective_logs[:, i].mean()})
                # for ret in dataset.log_returns():
                #     wandb.log({preference + 'Returns': ret})

                dataset.reset_trajectories()

            if t % config.update_frequency == 0:
                #evaluate_two_obj_baseline(ppo, config, weight_vector)
                evaluate_three_obj_baseline(ppo, config, weight_vector)

            #if save_count % 100 == 0:
                # torch.save(ppo.state_dict(),
                #            '../ppo_model/generalization/v_5/sample_efficiency/v_5_40_steps_' + str(save_count) + '.pt')
                #timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                # torch.save(ppo.state_dict(),
                #            '../../ppo_model/baseline/v_5_40_steps/final/v_5_40_steps_' + str(save_count) + '.pt')

            # Prepare state input for next time step
            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        torch.save(ppo.state_dict(),
                   '../../ppo_model/baseline/v_5_40_steps/_v5_40_steps_' + preference + '_' + timestamp + '.pt')

    except KeyboardInterrupt:
        print("Manual interruption detected...")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        torch.save(ppo.state_dict(),
                   '../../ppo_model/baseline/v_5_40_steps/_v5_40_steps_' + preference + '_' + timestamp + '.pt')
