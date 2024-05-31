import random

import numpy as np

import torch

from adaption.auxiliary import preference_context_generate
from adaption.generalization_ppo.evaluation import evaluate_two_obj
from adaption.generalization_ppo.gneralized_ppo import update_policy

# Use GPU if available
# from envs.in_use.gym_wrapper import VecEnv
# from adaption.ppo import PPO_CNN, update_policy
# from adaption.replay_buffer import TrajectoryDataset
from envs.in_use.gym_wrapper import VecEnv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def training(ppo, preference_set, dataset, optimizer, config, number):
    # Init WandB & Parameters


    # # 32 env parallel ->    2, 4, 26
    # fixed_samples = generate_fixed_samples_three(1)
    # custom_samples = generate_custom_samples_three(1)
    # random_samples = generate_samples_three(14, 3)
    # weight_vectors = np.concatenate((fixed_samples, custom_samples, random_samples), axis=0)

    # weight_vectors = [preference] * config.env_worker
    # weight_vectors_tensor = torch.tensor(weight_vectors).float().to(device)
    # Create Environmen

    # number of env, number of obj
    #print("preference before : ", preference_set)
    print("ini : ", preference_set)
    primary_preference = preference_set[0]
    remaining_preferences = preference_set[1:]
    #print("preference after : ", preference_set)
    print("primary : ", primary_preference)
    print("remaining : ",remaining_preferences)
    if len(remaining_preferences) != 0:
        active_sample = random.choices(remaining_preferences, k=8)
        primary_weight_vectors = [primary_preference] * 8

        print("active sample  :", active_sample)
        print("primary weight : ", primary_weight_vectors)

        weight_vectors = np.concatenate((primary_weight_vectors, active_sample), axis=0)
    else:
        weight_vectors = [primary_preference] * config.env_worker

    print("weight vector : ",weight_vectors)
    #weight_vectors = primary_preference * config.env_worker
    weight_vectors_tensor = torch.tensor(weight_vectors).float().to(device)
    contexts = preference_context_generate(config.env_worker, config.n_objs, weight_vectors_tensor)

    # state spaces
    vec_env = VecEnv(config.env_id, config.env_worker)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    print(" obs shape : ", obs_shape, "  ", state_shape,  "   ", in_channels)

    max_step = int(config.env_steps / config.env_worker)
    policy_update = False
    has_evaluate = False

    print("env id : ", config.env_id)
    n = 0
    while not policy_update:
        states_augmentation = torch.cat((states_tensor, contexts), dim=1)

        actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)

        next_states, rewards, done, info = vec_env.step(actions)
        scalarized_rewards = [np.dot(reward, weight) for reward, weight in zip(rewards, weight_vectors)]

        train_ready = dataset.write_tuple(states, actions, next_states, done, log_probs,
                                          logs=rewards, info=info, rewards=scalarized_rewards,
                                          weights=weight_vectors_tensor)

        if train_ready:
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          config.entropy_reg, config.GAE_lambda)

            dataset.reset_trajectories()
            policy_update = True
            return ppo, dataset, optimizer


        if number % 10 == 0 and not has_evaluate:
            has_evaluate = True
            evaluate_two_obj(ppo, config)
            print("evaluate")


        n+=1
        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
