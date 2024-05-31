import os

from tqdm import tqdm

from adaption.auxiliary import preference_context_generate
from envs.in_use.gym_wrapper import GymWrapper, VecEnv
from adaption.generalization_ppo.baseline.ppo_vanila_v import PPO
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np
import wandb


def evaluate_ppo(preference_sequences, config, n_eval, changing_frequency):
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
    obj_scalarized_reward= []
    obj_accumulative = []


    time_episode_reward = []
    time_episode_scalarized_reward = []

    next = False
    current_ppo = None

    ppo_list = []
    # ppo pools
    for i in range(len(preference_sequences)):
        true_preference = preference_sequences[i]
        preference_str = '_'.join(map(str, true_preference))  # This will turn [1, 0] into "1_0"
        filename = f"../ppo_model/baseline/v_5_40_steps/v_5_40_steps_{preference_str}.pt"

        print("file name : ", filename)

        ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
        ppo.load_state_dict(torch.load(filename))

        ppo_list.append(ppo)



    count = 0
    while count < n_eval:
        change_count = 0
        t = 0
        current_reward = [0.0, 0.0, 0.0]
        time_reward = []
        current_scalarized_reward = 0
        time_scalarized_reward = []
        next = False
        while not next:
            if t % changing_frequency == 0:

                index = change_count% len(preference_sequences)
                true_preference = preference_sequences[index]
                weight_vectors = [preference_sequences[index]]
                change_count+=1
                weight_vectors_tensor = torch.tensor(weight_vectors)
                weight_vectors_tensor = weight_vectors_tensor.float().to(device)

                print(t, " change, ", weight_vectors)

                current_ppo = ppo_list[index]

            states_tensor = states_tensor.unsqueeze(0)

            actions, log_probs = current_ppo.act(states_tensor)
            next_states, reward, done, info = env.step(actions)
            obj_logs.append(reward)
            scalarized_rewards = sum([true_preference[i] * reward[i] for i in range(len(reward))])
            obj_scalarized_reward.append(scalarized_rewards)

            #print("reward : ", reward)
            current_reward += reward
            time_reward.append(current_reward.copy())


            current_scalarized_reward += scalarized_rewards
            time_scalarized_reward.append(current_scalarized_reward)

            if done:
                print("DONE")
                next_states = env.reset()
                obj_logs = np.array(obj_logs).sum(axis=0)
                obj_returns.append(obj_logs)
                obj_logs = []
                obj_scalarized_reward = np.array(obj_scalarized_reward).sum()
                obj_accumulative.append(obj_scalarized_reward)
                obj_scalarized_reward = []


                time_episode_reward.append(time_reward)

                time_episode_scalarized_reward.append(time_scalarized_reward)

                current_reward = [0.0, 0.0, 0.0]
                time_reward = []

                current_scalarized_reward = 0
                time_scalarized_reward = []
                next = True

                count += 1

            t += 1
            # Prepare state input for next time step
            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        # print("episode reward : ", time_episode_reward)
        # print("episode return : ", time_episode_scalarized_reward)

    obj_accumulative = np.array(obj_accumulative)
    obj_accu_mean = obj_accumulative.mean()
    obj_accu_std = obj_accumulative.std()

    obj_returns = np.array(obj_returns)
    obj_means = obj_returns.mean(axis=0)
    obj_std = obj_returns.std(axis=0)

    time_episode_reward = np.array(time_episode_reward)
    time_episode_reward_mean = time_episode_reward.mean(axis=0)

    time_episode_scalarized_reward = np.array(time_episode_scalarized_reward)
    time_episode_scalarized_reward_mean = time_episode_scalarized_reward.mean(axis=0)

    return obj_accu_mean, obj_accu_std, list(obj_means), list(obj_std), list(time_episode_reward_mean), list(time_episode_scalarized_reward_mean)


def generate_all_preferences_three(step_size):
    preferences = []
    for w1 in range(0, 101, int(step_size * 100)):
        for w2 in range(0, 101 - w1, int(step_size * 100)):
            w1_scaled = w1 / 100.0
            w2_scaled = w2 / 100.0
            w3_scaled = 1 - w1_scaled - w2_scaled  # w3 is determined by w1 and w2
            if w3_scaled >= 0:  # Ensure non-negative weights
                preferences.append([w1_scaled, w2_scaled, w3_scaled])

    return preferences


def write_preferences_to_file(preferences, file_path):
    with open(file_path, 'w') as file:
        for pref in preferences:
            file.write(f"{pref}\n")


if __name__ == '__main__':
    wandb.init(project='EVALUATE', config={
        'env_id': 'v_5',
        'env_steps': 7e6,
        'batchsize_discriminator': 512,
        'batchsize_ppo': 32,
        'n_workers': 1,
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

    # weight_vectors = [[1.0, 0.0]]
    # weight_vectors_tensor = torch.tensor(weight_vectors)
    # weight_vectors_tensor = weight_vectors_tensor.to(device)

    #print("state shape : ", state_shape, " channel : ", in_channels, " action : ", n_actions)

    # ppo.load_state_dict(
    #     torch.load('../ppo_model/generalization/v_3/v_3_25_steps_2023-12-12_00-53-34_used.pt'))
    #


    # ppo.load_state_dict(
    #     torch.load('../ppo_model/generalization/v_5/sample_efficiency/v_5_40_steps_2024-01-16_19-33-03_used.pt'))
    # ppo.load_state_dict(
    #     torch.load('../ppo_model/generalization/v_5/sample_0.1/v_5_40_steps_0.1_2024-01-18_01-47-55_used.pt'))

    # test here
    #weight_vectors = generate_beta_preferences(1, 1)
    # weight_vectors = [[1, 0, 0]]
    # weight_vectors_tensor = torch.tensor(weight_vectors)
    # weight_vectors_tensor = weight_vectors_tensor.to(device)

    #preferences = generate_all_preferences_three(0.1)

    preferences = [[0.66,0.32,0.02],[0.04,0.92,0.04],[0.25,0.5,0.25],[0.66,0.32,0.02],[0.0,0.25,0.75]]
    #preferences =[[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.25, 0.75]]


    # for i in range(len(preferences)):
    #     weight_vectors = [preferences[i]]
    #     weight_vectors_tensor = torch.tensor(weight_vectors)
    #     weight_vectors_tensor = weight_vectors_tensor.float().to(device)


    a, b, c, d, e, f= evaluate_ppo(preferences, config, n_eval=100, changing_frequency=8)

    print(e)

    print("###########")

    print(f)

    # Store data into a txt file
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find('project')] + 'project'
    result_path = os.path.join(root_path, 'results', 'time', 'v_5')

    print("result path ; ", result_path)

    file_path  = os.path.join(result_path, 'v_5_baseline_objective_data_time_[0.66,0.32,0.02],[0.04,0.92,0.04],[0.25,0.5,0.25],[0.66,0.32,0.02],[0.0,0.25,0.75].txt')

    with open(file_path, 'w') as fi:
        for point in e:
            fi.write(f"{point[0]}, {point[1]}, {point[2]}\n")

    file_path_return = os.path.join(result_path, 'v_5_baseline_return_data_time_[0.66,0.32,0.02],[0.04,0.92,0.04],[0.25,0.5,0.25],[0.66,0.32,0.02],[0.0,0.25,0.75].txt')
    write_preferences_to_file(f, file_path_return)











