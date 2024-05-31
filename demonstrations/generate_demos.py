import numpy as np
from tqdm import tqdm
import torch
from envs.in_use import v_3
import pickle

# Use GPU if available
from envs.in_use.gym_wrapper import GymWrapper

from adaption.generalization_ppo.baseline.ppo_vanila_v import PPO, update_policy, TrajectoryDataset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# Select Environment
env_id = 'v_3'

# Select n demos (need to set timer separately)
n_demos = 1000
max_steps = 25

# Initialize Environment
env = GymWrapper(env_id)
states = env.reset()
states_tensor = torch.tensor(states).float().to(device)
dataset = []
episode = {'states': [], 'actions': []}
episode_cnt = 0

# Fetch Shapes
n_actions = env.action_space.n
obs_shape = env.observation_space.shape
state_shape = obs_shape[:-1]
in_channels = obs_shape[-1]

# Load Pretrained PPO
ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
#ppo.load_state_dict(torch.load('../framework/warehouse_0_[0.0, 1.0].pt'))
#ppo.load_state_dict(torch.load('../saved_models/ppo_airl_v3_[0,1,0,1].pt'))
ppo.load_state_dict(torch.load('../adaption/ppo_model/baseline/v_3_25_steps/used/_v3_25_steps_0.9_0.1.pt'))

obj_logs = []
obj_returns = []

for t in tqdm(range((max_steps-1)*n_demos)):
    actions, log_probs = ppo.act(states_tensor)
    next_states, rewards, done, info = env.step(actions)
    episode['states'].append(states)
    # Note: Actions currently append as arrays and not integers!
    episode['actions'].append(actions)

    obj_logs.append(rewards)

    if done:
        #print("len of epsideo : ", len(episode))
        next_states = env.reset()
        dataset.append(episode)
        #print("episode : ", episode)


        obj_logs = np.array(obj_logs).sum(axis=0)
        obj_returns.append(obj_logs)
        obj_logs = []

        episode = {'states': [], 'actions': []}

    # Prepare state input for next time step
    states = next_states.copy()
    states_tensor = torch.tensor(states).float().to(device)

obj_returns = np.array(obj_returns)
obj_means = obj_returns.mean(axis=0)
obj_std = obj_returns.std(axis=0)

print("obj mean : ",obj_means)
print("obj_std : ", obj_std)

pickle.dump(dataset, open('../adaption/demonstration/_v3_[0.9, 0.1]_1000' + '.pk', 'wb'))