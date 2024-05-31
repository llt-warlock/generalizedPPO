import numpy as np
from tqdm import tqdm

import torch

from adaption.auxiliary import preference_context_generate
from adaption.generalization_ppo.evaluation import evaluate_two_obj
from adaption.generalization_ppo.gneralized_ppo import PPO_CNN, update_policy
from adaption.generalization_ppo.replay_buffer_steps import TrajectoryDataset
from adaption.generalization_ppo.samping.preference_sampling import generate_samples_two, generate_fixed_samples_two
import wandb

# Use GPU if available
# from envs.in_use.gym_wrapper import VecEnv
# from adaption.ppo import PPO_CNN, update_policy
# from adaption.replay_buffer import TrajectoryDataset
from envs.in_use.gym_wrapper import VecEnv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # Init WandB & Parameters
    wandb.init(project='PPO', config={
        'env_id': 'v_3v',
        'env_steps': 6e6,
        'batchsize_ppo': 32,
        'n_workers': 16,
        'lr_ppo': 3e-4,
        'entropy_reg': 0.01,
        'lambd': [0, 1, 0],
        'gamma': 0.995,
        'epsilon': 0.1,
        'ppo_epochs': 5,
        'GAE_lambda': 0.98,
        'mini_batch_size': 16,
        'update_frequency': 1000,
        'n_objs': 2
    })
    config = wandb.config

    # # 32 env parallel ->    2, 4, 26
    # fixed_samples = generate_fixed_samples_three(1)
    # custom_samples = generate_custom_samples_three(1)
    # random_samples = generate_samples_three(14, 3)
    # weight_vectors = np.concatenate((fixed_samples, custom_samples, random_samples), axis=0)

    fixed_samples = generate_fixed_samples_two(1)
    random_samples = generate_samples_two(15, 2)
    weight_vectors = np.concatenate((fixed_samples, random_samples), axis=0)


    # weight_vectors = [[1,0], [1,0], [1,0], [1,0], [1,0], [1,0],
    #                   [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    #                   [0.7, 0.3], [0.3, 0.7], [0.5, 0.5], [0.5, 0.5]]

    weight_vectors_tensor = torch.tensor(weight_vectors).float().to(device)

    # Create Environmen

    # number of env, number of obj
    contexts = preference_context_generate(config.n_workers, config.n_objs, weight_vectors_tensor)

    # state spaces
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    print(" obs shape : ", obs_shape, "  ", state_shape,  "   ", in_channels)

    # Initialize Models
    ppo = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=config.n_objs, contexts=config.n_objs).to(device)
    # ppo.load_state_dict(
    #     torch.load('./meta_ppo/_v4_meta_ppo_2023-11-22_17-53-11_.pt'))
    # ppo = PPO_CNN(state_shape=obs_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo, eps=1e-5)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
    max_step = int(config.env_steps / config.n_workers)

    re_sample = False


    try:
        for t in tqdm(range(max_step)):

            # learning rate decrease
            lr_a_now = config.lr_ppo * (1 - t / max_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_a_now
            # for param_group in icm_optimizer.param_groups:
            #     param_group['lr'] = lr_a_now

            if re_sample:
                # fixed_samples = generate_fixed_samples_three(1)
                # custom_samples = generate_custom_samples_three(1)
                # random_samples = generate_samples_three(14, 3)
                # weight_vectors = np.concatenate((fixed_samples, custom_samples, random_samples), axis=0)

                fixed_samples = generate_fixed_samples_two(1)
                random_samples = generate_samples_two(15, 2)
                weight_vectors = np.concatenate((fixed_samples, random_samples), axis=0)

                # weight_vectors = [[1,0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
                #                   [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                #                   [0.7, 0.3], [0.3, 0.7], [0.5, 0.5], [0.5, 0.5]]


                weight_vectors_tensor = torch.tensor(weight_vectors).float().to(device)
                #weight_vectors = np.repeat(weight_vectors, 4, axis=
                print("weight vector tensor : ", weight_vectors_tensor)

                contexts = preference_context_generate(config.n_workers, config.n_objs, weight_vectors_tensor)
                re_sample = False

                print("preference : ", weight_vectors)

            states_augmentation = torch.cat((states_tensor, contexts), dim=1)

            actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)

            next_states, rewards, done, info = vec_env.step(actions)

            scalarized_rewards = [np.dot(reward, weight) for reward, weight in zip(rewards, weight_vectors)]

            train_ready = dataset.write_tuple(states, actions, next_states, done, log_probs,
                                              logs=rewards, info=info, rewards=scalarized_rewards,
                                              weights=weight_vectors_tensor)

            if train_ready:
                print("data : ", len(dataset.trajectories))
                update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                              config.entropy_reg, config.GAE_lambda)

                re_sample = True

                dataset.reset_trajectories()

            # Prepare state input for next time step
            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

            if t % config.update_frequency == 0:
                evaluate_two_obj(ppo, config)
                #evaluate_three_obj(ppo, config)

        # vec_env.close()
        # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # torch.save(ppo.state_dict(), '../ppo_model/generalization/v_3/v_3_25_steps_' + timestamp + '.pt' )
        #

    except KeyboardInterrupt:
        print("Manual interruption detected...")
        # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # torch.save(ppo.state_dict(), '../ppo_model/generalization/v_3/v_3_25_steps_' + timestamp + '.pt' )
        #

# def create_weight_tensor(weight, num_vectors):
#     """
#     Create a tensor of weight vectors.
#
#     Args:
#     - weight (list): A list containing the weights, e.g., [1.0, 0.0].
#     - num_vectors (int): The number of weight vectors to create in the tensor.
#
#     Returns:
#     - torch.Tensor: A tensor with the specified number of weight vectors.
#     """
#     weight_vector = torch.tensor(weight)  # Create a tensor from the weight list
#     weight_tensor = weight_vector.repeat(num_vectors, 1)  # Repeat the weight vector num_vectors times
#     return weight_tensor
#
#
