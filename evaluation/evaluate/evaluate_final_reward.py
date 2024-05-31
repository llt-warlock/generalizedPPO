import torch

from irl.airl import Discriminator

from tqdm import tqdm
from envs.in_use import v_1
from envs.gym_wrapper import *
import wandb
import argparse
from envs.new_test_1 import idx_to_scalar, board_position, board_position_below
# Use GPU if available
from framework.ppo import PPO_CNN, TrajectoryDataset, update_policy


# WEIGHTS = np.linspace(0, 10, 10)
# THRESHOLD = 1e-3
# MAX_ITERATIONS = len(WEIGHTS)

WEIGHTS = [0.01]

evaluation_results = []

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def extract_expert_weight(convex_dict):
    # Find the maximum second value
    max_second_value = max([value[1] for value in convex_dict.values()])

    # Filter key-value pairs with max second value
    filtered_dict = {key: value for key, value in convex_dict.items() if value[1] == max_second_value}

    # Find the maximum first value among filtered key-value pairs
    max_first_value = max([value[0] for value in filtered_dict.values()])

    # Filter key-value pairs with max second value and max first value
    filtered_dict = {key: value for key, value in filtered_dict.items() if value[0] == max_first_value}

    # Extract the minimal value of the key (w_e)
    min_we = min(map(float, filtered_dict.keys()))

    return min_we

if __name__ == '__main__':

    # Fetch ratio args for automatic preferences
    parser = argparse.ArgumentParser(description='Preference Ratio.')
    parser.add_argument('--ratio', nargs='+', type=int)
    args = parser.parse_args()

    # Config
    wandb.init(project='Policy_Iteration', config={
        'env_id': 'v_1',
        'env_steps': 9e6,
        'batchsize_ppo': 8,
        'n_workers': 8,
        'lr_ppo': 3e-4,
        'entropy_reg': 0.01,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5,
        'GAE_lambda': 0.98
    })
    config = wandb.config

    # Create Environment

    # state spaces
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    convex_dict = {}
    current_reward_vector = []
    try:

        for w_e in WEIGHTS:



            next_weight = False

            # Initialize Models
            print('Initializing and Normalizing Rewards...')
            ppo = PPO_CNN(state_shape=obs_shape, in_channels=in_channels, n_actions=n_actions).to(device)
            optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo, eps=1e-5)
            dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
            ppo.load_state_dict(torch.load('../../ppo_agent/meta/v_1_meta_policy_50_iteration.pt'))

            # Expert 1
            discriminator_0 = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)

            discriminator_0.load_state_dict(torch.load(
                '../../saved_models/meta_use/discriminatorn_v_1_9_3_1000_[0,1].pt'))
            ppo_0 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
            ppo_0.load_state_dict(torch.load('../../saved_models/meta_use/airl_policy_v_1_9_3_1000_[0,1].pt'))
            utop_0 = discriminator_0.estimate_utopia(ppo_0, config)
            print(f'Reward Normalization 1: {utop_0}')
            discriminator_0.set_eval()

            max_step = int(config.env_steps / config.n_workers)

            # max step allowed
            for t in tqdm(range(int(config.env_steps / config.n_workers))):

                lr_a_now = config.lr_ppo * (1 - t / max_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_a_now

                actions, log_probs = ppo.act(states_tensor)
                next_states, rewards, done, info = vec_env.step(actions)

                # Fetch AIRL rewards
                airl_state = torch.tensor(states).to(device).float()
                airl_next_state = torch.tensor(next_states).to(device).float()
                airl_rewards_0 = discriminator_0.forward(airl_state, airl_next_state, config.gamma).squeeze(1)
                airl_rewards_0 = airl_rewards_0.detach().cpu().numpy() * [0 if i else 1 for i in done]

                vectorized_rewards = [[r[0], airl_rewards_0[i]] for i, r in enumerate(rewards)]
                scalarized_rewards = [np.dot([0,1], r) for r in vectorized_rewards]

                train_ready = dataset.write_tuple(states, actions, next_states, scalarized_rewards, done, log_probs,
                                                  rewards, info)


                if train_ready:
                    update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                                                 config.entropy_reg, config.GAE_lambda, is_maml=False)

                    objective_logs = dataset.log_objectives()
                    for i in range(objective_logs.shape[1]):
                        wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
                    for ret in dataset.log_returns():
                        wandb.log({'Returns': ret})


                    dataset.reset_trajectories()


                states = next_states.copy()
                states_tensor = torch.tensor(states).float().to(device)


        print("current_reward_vector : ", current_reward_vector)
        # extract policy
        #minimal_w_e = extract_expert_weight(convex_dict)

    except KeyboardInterrupt:
        print("keyboard interrupt")