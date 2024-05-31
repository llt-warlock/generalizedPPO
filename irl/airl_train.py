import datetime

from tqdm import tqdm
from adaption.generalization_ppo.baseline.ppo_vanila_v import PPO, update_policy, TrajectoryDataset
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.in_use.gym_wrapper import VecEnv
from irl.airl import *
import torch
import numpy as np
import pickle
import wandb

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Load demonstrations
    expert_trajectories = pickle.load(open('../adaption/demonstration/_v3_[0.2, 0.8]_1000.pk', 'rb'))

    # Init WandB & Parameters
    wandb.init(project='AIRL', config={
        'env_id': 'v_3',
        'env_steps': 6e6,
        'batchsize_discriminator': 512,
        'batchsize_ppo': 32,
        'n_workers': 16,
        'entropy_reg': 0.01,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 10,
        'GAE_lambda': 0.98,
        'disc_epochs': 10,
        'mini_batch_size': 32,
        'lr_ppo': 3e-4,
        'lr_disc': 3e-4
    })
    config = wandb.config

    # Create Environment
    #vec_env = SubprocVecEnv([make_env(config.env_id, i) for i in range(config.n_workers)])
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
    discriminator = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=config.lr_disc)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    policy_trajectory_buffer = []

    # # Logging
    # objective_logs = []
    try:
        max_step = int(config.env_steps / config.n_workers)

        for t in tqdm(range((int(config.env_steps/config.n_workers)))):

            # Act
            actions, log_probs = ppo.act(states_tensor)
            next_states, rewards, done, info = vec_env.step(actions)

            # # Log Objectives
            # objective_logs.append(rewards)

            # learning rate decrease
            lr_a_now = config.lr_ppo * (1 - t / max_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_a_now
            for param_group in optimizer_discriminator.param_groups:
                param_group['lr'] = lr_a_now


            # Calculate (vectorized) AIRL reward
            airl_state = torch.tensor(states).to(device).float()
            airl_next_state = torch.tensor(next_states).to(device).float()
            airl_action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
            airl_rewards = discriminator.predict_reward(airl_state, airl_next_state, config.gamma, airl_action_prob)
            airl_rewards = list(airl_rewards.detach().cpu().numpy() * [0 if i else 1 for i in done])

            # Save Trajectory
            #train_ready = dataset.write_tuple(states, actions, airl_rewards, done, log_probs)
            # train_ready = dataset.write_tuple(states, actions, next_states, done, log_probs,
            #                                   logs=rewards, info=info, rewards=airl_rewards)
            train_ready = dataset.write_tuple(states, next_states, actions, airl_rewards, done, log_probs,
                                              rewards, info)

            if train_ready:
                update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                              config.entropy_reg, config.GAE_lambda)


                d_loss, fake_acc, real_acc = update_discriminator(discriminator=discriminator,
                                                                  optimizer=optimizer_discriminator,
                                                                  gamma=config.gamma,
                                                                  expert_trajectories=expert_trajectories,
                                                                  policy_trajectories=dataset.trajectories.copy(), ppo=ppo,
                                                                  batch_size=config.batchsize_discriminator,
                                                                  epcohes=config.disc_epochs)

                # Log Loss Statsitics
                wandb.log({'Discriminator Loss': d_loss,
                           'Fake Accuracy': fake_acc,
                           'Real Accuracy': real_acc})

                objective_logs = dataset.log_objectives()
                # print("objective_logs : ", objective_logs)
                for i in range(objective_logs.shape[1]):
                    wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})

                for ret in dataset.log_returns():
                    wandb.log({'Returns': ret})
                dataset.reset_trajectories()

            # Prepare state input for next time step
            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        #vec_env.close()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        torch.save(discriminator.state_dict(), '../adaption/irl_model/discriminatorn_v_3_1000_[0.2,0.8]_' + timestamp + '.pt')
        torch.save(ppo.state_dict(),
                   '../adaption/irl_model/airl_policy_v_3_1000_[0.2,0.8]_' + timestamp + '.pt')

    except KeyboardInterrupt:
        print("Manual interruption detected...")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        torch.save(discriminator.state_dict(), '../adaption/irl_model/discriminatorn_v_3_1000_[0.2,0.8]_' + timestamp + '.pt')
        torch.save(ppo.state_dict(),
                   '../adaption/irl_model/airl_policy_v_3_1000_[0.2,0.8]_' + timestamp + '.pt')



