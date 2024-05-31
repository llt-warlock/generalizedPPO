import numpy as np
import torch

from adaption.auxiliary import preference_context_generate
from envs.in_use.gym_wrapper import GymWrapper, VecEnv
from adaption.generalization_ppo.baseline.ppo_vanila_v import PPO
import torch

import numpy as np
import wandb
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate_ppo(ppo, config, weight, n_eval=1000):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards
    """
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    print("state tensor: ", states_tensor.size())

    obj_logs = []
    obj_returns = []
    obj_scalarized_reward= []
    obj_accumulative = []

    count = 0

    while count < n_eval:
        actions, log_probs = ppo.act(states_tensor)
        next_states, reward, done, info = env.step(actions)
        obj_logs.append(reward)
        scalarized_rewards = sum([weight[i] * reward[i] for i in range(len(reward))])
        obj_scalarized_reward.append(scalarized_rewards)

        if done:
            print(count)
            next_states = env.reset()
            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)
            obj_logs = []
            obj_scalarized_reward = np.array(obj_scalarized_reward).sum()
            obj_accumulative.append(obj_scalarized_reward)
            obj_scalarized_reward = []
            count += 1

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    obj_accumulative = np.array(obj_accumulative)
    obj_accu_mean = obj_accumulative.mean()
    obj_accu_std = obj_accumulative.std()

    obj_returns = np.array(obj_returns)
    obj_means = obj_returns.mean(axis=0)
    obj_std = obj_returns.std(axis=0)


    print("obj_mean ", obj_means)
    print("obj std : ", obj_std)

    return obj_accu_mean, obj_accu_std, list(obj_means), list(obj_std)


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

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)


    # ppo.load_state_dict(
    #     torch.load('../ppo_model/baseline/v_3_25_steps/used/_v3_25_steps_0.45_0.55.pt'))


    ppo.load_state_dict(
        torch.load('../ppo_model/baseline/v_5_40_steps/v_5_40_steps_0.33_0.33_0.34.pt'))

    a, b, c, d= evaluate_ppo(ppo, config, [0.33, 0.33, 0.34], 1000)

    print(a, " ", b, " ", c, " ", d)