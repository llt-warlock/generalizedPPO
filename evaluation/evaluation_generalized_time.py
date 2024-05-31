import os

from tqdm import tqdm

from adaption.auxiliary import preference_context_generate
from envs.in_use.gym_wrapper import GymWrapper, VecEnv
from adaption.generalization_ppo.gneralized_ppo import PPO_CNN
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np
import wandb



def generate_beta_preferences(alpha, beta):
    """
    Generate a preference vector from a Beta distribution.

    Args:
        alpha (float): Alpha parameter for the Beta distribution.
        beta (float): Beta parameter for the Beta distribution.

    Returns:
        numpy.ndarray: A 2D preference vector.
    """
    omega_1 = np.random.beta(alpha, beta)
    omega_2 = 1 - omega_1  # Ensure that they sum to 1
    return np.array([omega_1, omega_2])




def estimate_gradient(ppo, config, reward_function, current_preference, epsilon=0.01):
    """
    Estimate the gradient of the expected reward with respect to the preference vector.

    Args:
        reward_function (function): A function that returns the scalarized reward given a preference vector.
        current_preference (numpy.ndarray): The current preference vector.
        epsilon (float): The perturbation for finite differences.

    Returns:
        numpy.ndarray: Estimated gradient of the reward with respect to the preference vector.
    """
    grad = np.zeros_like(current_preference)
    for i in range(len(current_preference)):
        # Create perturbed vectors
        preference_plus = current_preference.copy()
        preference_minus = current_preference.copy()

        print("preference plus ", preference_plus, "  ", preference_minus)

        preference_plus[i] += epsilon  # Perturb one element positively
        preference_minus[i] -= epsilon  # Perturb one element negatively

        # Ensure they are still valid preference vectors
        preference_plus /= np.sum(preference_plus)
        preference_minus /= np.sum(preference_minus)

        # # Calculate the finite difference
        # reward_plus = reward_function(preference_plus)
        # reward_minus = reward_function(preference_minus)

        reward_plus, _ , _ , _  = evaluate_ppo(ppo, preference_plus, torch.tensor([preference_plus]).float().to(device), [0,1], config, n_eval=50)
        reward_minus, _, _, _= evaluate_ppo(ppo, preference_plus, torch.tensor([preference_minus]).float().to(device), [0,1], config, n_eval=50)

        print("reward plus : ", reward_plus, "  reward minus : ", reward_minus)

        grad[i] = (reward_plus - reward_minus) / (2 * epsilon)

    return grad


def update_preferences_with_gradient(current_preference, grad, learning_rate=0.1):
    """
    Update the preference vector using the estimated gradient.

    Args:
        current_preference (numpy.ndarray): The current preference vector.
        grad (numpy.ndarray): Estimated gradient of the reward with respect to the preference vector.
        learning_rate (float): Learning rate for the gradient update.

    Returns:
        numpy.ndarray: Updated preference vector.
    """
    # Update preferences in the direction of the gradient
    new_preference = current_preference + learning_rate * grad

    # Ensure the updated preferences are still valid (sum to 1 and each element in [0, 1])
    new_preference = np.clip(new_preference, 0, 1)
    new_preference /= new_preference.sum()

    return new_preference





def evaluate_ppo(ppo, preference_sequences, config, n_eval, changing_frequency):
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
                contexts = preference_context_generate(config.n_workers, 3, weight_vectors_tensor)

                print(t, " change, ", weight_vectors)

            states_tensor = states_tensor.unsqueeze(0)
            states_augmentation = torch.cat((states_tensor, contexts), dim=1)

            #print(states_augmentation.size(), "  ", weight_vectors_tensor.size())

            actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
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

    ppo = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions, weight=3, contexts=3).to(device)
    #
    # ppo.load_state_dict(
    #     torch.load('../ppo_model/generalization/v_3/sample_efficiency/v_3_25_steps_2024-01-19_06-08-40_0.1_used.pt'))
    # ppo.load_state_dict(
    #     torch.load('../ppo_model/generalization/v_3/v_3_25_steps_2023-12-12_00-53-34_used.pt'))
    #


    ppo.load_state_dict(
        torch.load('../ppo_model/generalization/v_5/sample_efficiency/v_5_40_steps_2024-01-16_19-33-03_used.pt'))
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


    a, b, c, d, e, f= evaluate_ppo(ppo, preferences, config, n_eval=100, changing_frequency=8)

    print(e)

    print("###########")

    print(f)

    # Store data into a txt file
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find('project')] + 'project'
    result_path = os.path.join(root_path, 'results', 'time', 'v_5')

    print("result path ; ", result_path)

    file_path  = os.path.join(result_path, 'v_5_objective_data_time_[0.66,0.32,0.02],[0.04,0.92,0.04],[0.25,0.5,0.25],[0.66,0.32,0.02],[0.0,0.25,0.75].txt')

    with open(file_path, 'w') as fi:
        for point in e:
            fi.write(f"{point[0]}, {point[1]}, {point[2]}\n")

    file_path_return = os.path.join(result_path, 'v_5_return_data_time_[0.66,0.32,0.02],[0.04,0.92,0.04],[0.25,0.5,0.25],[0.66,0.32,0.02],[0.0,0.25,0.75].txt')
    write_preferences_to_file(f, file_path_return)




















        # print(a, " ", b, " ", c, " ", d)
        #
        # print("#############")
        # print(" len : ", len(e), " ", len(f))
        # print( " ", e, " ",f)

        # grad = estimate_gradient(ppo, config, a, weight_vectors[0], epsilon=0.01)
        #
        # print("grad : ", grad)
        #
        # new_preference = update_preferences_with_gradient(weight_vectors[0], grad, learning_rate=0.01)
        # print("new preference vector : ", new_preference)
        #
        # weight_vectors = [new_preference]

