import numpy as np
import torch
import wandb

from adaption.auxiliary import preference_context_generate
# Use GPU if available
from envs.in_use.gym_wrapper import VecEnv, GymWrapper

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate_two_obj_baseline(ppo, config, preference):
    weight = preference
    # Create Environmen

    # state spaces
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    obj_logs = []
    obj_returns = []
    obj_scalarized_reward= []
    obj_accumulative = []

    for _ in range(1000):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = env.step(actions)

        # print("rewards : ", rewards)

        obj_logs.append(rewards)
        scalarized_rewards = sum([weight[i] * rewards[i] for i in range(len(rewards))])

        # print("scalar reward : ", scalarized_rewards)

        obj_scalarized_reward.append(scalarized_rewards)

        if done:
            next_states = env.reset()
            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)
            obj_logs = []
            obj_scalarized_reward = np.array(obj_scalarized_reward).sum()
            obj_accumulative.append(obj_scalarized_reward)
            obj_scalarized_reward = []

        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    # print("reward vector : ", obj_returns)
    # print("retrun : ", obj_accumulative)

    for i in range(np.array(obj_returns).shape[1]):
        wandb.log({'Evaluation Pre_' + '[0.33, 0.67]' + ': obj_' + str(i): (np.array(obj_returns)[:, i]).mean()})

    for ret in obj_accumulative:
        wandb.log({'Returns': ret})


def evaluate_two_obj_baseline_steps(ppo, config):
    weight = [1.0, 0.0]
    # Create Environmen

    # state spaces
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    obj_logs = []
    obj_returns = []
    obj_scalarized_reward= []
    obj_accumulative = []

    for _ in range(400):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = env.step(actions)

        # print("rewards : ", rewards)

        obj_logs.append(rewards)
        scalarized_rewards = sum([weight[i] * rewards[i] for i in range(len(rewards))])

        # print("scalar reward : ", scalarized_rewards)

        obj_scalarized_reward.append(scalarized_rewards)

        if done:
            next_states = env.reset()
            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)
            obj_logs = []
            obj_scalarized_reward = np.array(obj_scalarized_reward).sum()
            obj_accumulative.append(obj_scalarized_reward)
            obj_scalarized_reward = []

        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    # print("reward vector : ", obj_returns)
    # print("retrun : ", obj_accumulative)

    for i in range(np.array(obj_returns).shape[1]):
        wandb.log({'Evaluation Pre_' + '[1.0, 0.0, 0.0]' + ': obj_' + str(i): (np.array(obj_returns)[:, i]).mean()})

    for ret in obj_accumulative:
        wandb.log({'Returns': ret})



def evaluate_three_obj_baseline(ppo, config, preference):
    weight = preference
    # Create Environmen

    # state spaces
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    obj_logs = []
    obj_returns = []
    obj_scalarized_reward= []
    obj_accumulative = []

    for _ in range(1000):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = env.step(actions)

        # print("rewards : ", rewards)

        obj_logs.append(rewards)
        scalarized_rewards = sum([weight[i] * rewards[i] for i in range(len(rewards))])
        # print("weight : ", weight)
        # print("reward : ", rewards)
        # print("scalarized : ", scalarized_rewards)

        # print("scalar reward : ", scalarized_rewards)

        obj_scalarized_reward.append(scalarized_rewards)

        if done:
            next_states = env.reset()
            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)
            obj_logs = []
            obj_scalarized_reward = np.array(obj_scalarized_reward).sum()
            obj_accumulative.append(obj_scalarized_reward)
            obj_scalarized_reward = []

        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    # print("reward vector : ", obj_returns)
    # print("retrun : ", obj_accumulative)

    for i in range(np.array(obj_returns).shape[1]):
        wandb.log({'Evaluation Pre_' + str(preference) + ': obj_' + str(i): (np.array(obj_returns)[:, i]).mean()})

    for ret in obj_accumulative:
        wandb.log({'Returns': ret})




def evaluate_two_obj(ppo, config):
    weight_vectors = [[1.0, 0.0],
                      [0.9, 0.1],
                      [0.8, 0.2],
                      [0.7, 0.3],
                      [0.6, 0.4],
                      [0.5, 0.5],
                      [0.4, 0.6],
                      [0.3, 0.7],
                      [0.2, 0.8],
                      [0.1, 0.9],
                      [0.0, 1.0]]

    weight_vectors_tensor = torch.tensor(weight_vectors)
    weight_vectors_tensor = weight_vectors_tensor.to(device)
    # Create Environmen

    # number of env, number of obj
    contexts = preference_context_generate(11, 2, weight_vectors_tensor)

    # state spaces
    vec_env = VecEnv(config.env_id, 11)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    env_pre = {'0': '[1,0]', '1': '[0.9, 0.1]', '2': '[0.8, 0.2]', '3': '[0.7, 0.3]', '4': '[0.6, 0.4]', '5': '[0.5, 0.5]',
               '6': '[0.4, 0.6]', '7': '[0.3, 0.7]', '8': '[0.2, 0.8]', '9': '[0.1, 0.9]', '10': '[0, 1]'}

    env_dict = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[]}

    env_trajectory = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[]}

    for _ in range(1000):
        states_augmentation = torch.cat((states_tensor, contexts), dim=1)
        actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
        next_states, rewards, done, info = vec_env.step(actions)

        for i in range(11):
            env_dict[str(i)].append(rewards[i])
            if done[i]:
                temp = np.array(env_dict[str(i)]).sum(axis=0)
                env_trajectory[str(i)].append(temp)
                env_dict[str(i)] = []

        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    for key, value in env_trajectory.items():
        if len(value) != 0:
            v = np.array(value)
            for i in range(v.shape[1]):
                wandb.log({'Pre_' + env_pre[str(key)] + ': obj_' + str(i): (np.array(value)[:, i]).mean()})



def evaluate_three_obj(ppo, config):
    weight_vectors = [
        [1.0, 0.0, 0.0],
        [0.75, 0.25, 0.0],
        [0.75, 0.0, 0.25],
        [0.5, 0.5, 0.0],
        [0.5, 0.25, 0.25],
        [0.5, 0.0, 0.5],
        [0.25, 0.75, 0.0],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
        [0.25, 0.0, 0.75],
        [0.0, 1.0, 0.0],
        [0.0, 0.75, 0.25],
        [0.0, 0.5, 0.5],
        [0.0, 0.25, 0.75],
        [0.0, 0.0, 1.0]
    ]

    weight_vectors_tensor = torch.tensor(weight_vectors)
    weight_vectors_tensor = weight_vectors_tensor.to(device)
    # Create Environmen

    # number of env, number of obj
    contexts = preference_context_generate(15, 3, weight_vectors_tensor)

    # state spaces
    vec_env = VecEnv(config.env_id, 15)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    env_pre = {'0': '[1.0, 0.0, 0.0]', '1': '[0.75, 0.25, 0.0]', '2': '[0.75, 0.0, 0.25]', '3': '[0.5, 0.5, 0.0]', '4': ' [0.5, 0.25, 0.25]', '5': '[0.5, 0.0, 0.5]',
               '6': '[0.25, 0.75, 0.0]', '7': '[0.25, 0.5, 0.25]', '8': ' [0.25, 0.25, 0.5]', '9': '[0.25, 0.0, 0.75]', '10': '[0.0, 1.0, 0.0]',
               '11': '[0.0, 0.75, 0.25]', '12': '[0.0, 0.5, 0.5]', '13': '[0.0, 0.25, 0.75]', '14':'[0.0, 0.0, 1.0]'}

    env_dict = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[], '14':[]}

    env_trajectory = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [],
                '11': [], '12': [], '13': [], '14': []}

    for _ in range(1000):
        states_augmentation = torch.cat((states_tensor, contexts), dim=1)
        actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
        next_states, rewards, done, info = vec_env.step(actions)

        for i in range(15):
            env_dict[str(i)].append(rewards[i])
            if done[i]:
                temp = np.array(env_dict[str(i)]).sum(axis=0)
                env_trajectory[str(i)].append(temp)
                env_dict[str(i)] = []

        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    for key, value in env_trajectory.items():
        if len(value) != 0:
            v = np.array(value)
            for i in range(v.shape[1]):
                wandb.log({'Pre_' + env_pre[str(key)] + ': obj_' + str(i): (np.array(value)[:, i]).mean()})


def evaluate_four_obj(ppo, config):
    weight_vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0, 0.5],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5, 0.5],
        [0.33, 0.33, 0.34, 0.0],
        [0.33, 0.34, 0.0, 0.33],
        [0.34, 0.0, 0.33, 0.33],
        [0.0, 0.33, 0.33, 0.34],
        [0.25, 0.25, 0.25, 0.25]
    ]

    weight_vectors_tensor = torch.tensor(weight_vectors)
    weight_vectors_tensor = weight_vectors_tensor.to(device)
    # Create Environmen

    # number of env, number of obj
    contexts = preference_context_generate(15, 4, weight_vectors_tensor)

    # state spaces
    vec_env = VecEnv(config.env_id, 15)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    env_pre = {'0': '[1.0, 0.0, 0.0, 0.0]', '1': ' [0.0, 1.0, 0.0, 0.0]', '2': '[0.0, 0.0, 1.0, 0.0]', '3': '[0.0, 0.0, 0.0, 1.0]',
               '4': ' [0.5, 0.5, 0.0, 0.0]', '5': '[0.5, 0.0, 0.5, 0.0]', '6': '[0.5, 0.0, 0.0, 0.5]',
               '7': ' [0.0, 0.5, 0.5, 0.0]', '8': '[0.0, 0.5, 0.0, 0.5]', '9': '[0.0, 0.0, 0.5, 0.5]',
               '10': '[0.33, 0.33, 0.34, 0.0]', '11': '[0.33, 0.34, 0.0, 0.33]', '12': '[0.34, 0.0, 0.33, 0.33]', '13':'[0.0, 0.33, 0.33, 0.34]',
               '14': '[0.25, 0.25, 0.25, 0.25]'}

    env_dict = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[], '14':[]}

    env_trajectory = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [],
                '11': [], '12': [], '13': [], '14': []}

    for _ in range(1000):
        states_augmentation = torch.cat((states_tensor, contexts), dim=1)
        actions, log_probs = ppo.act(states_augmentation, weight_vectors_tensor)
        next_states, rewards, done, info = vec_env.step(actions)

        for i in range(15):
            env_dict[str(i)].append(rewards[i])
            if done[i]:
                temp = np.array(env_dict[str(i)]).sum(axis=0)
                env_trajectory[str(i)].append(temp)
                env_dict[str(i)] = []

        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    for key, value in env_trajectory.items():
        if len(value) != 0:
            v = np.array(value)
            for i in range(v.shape[1]):
                wandb.log({'Pre_' + env_pre[str(key)] + ': obj_' + str(i): (np.array(value)[:, i]).mean()})
