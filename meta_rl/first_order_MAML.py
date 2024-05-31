from tqdm import tqdm

from envs.gym_wrapper import *
import wandb

# Use GPU if available

from framework.ppo import PPO_CNN, TrajectoryDataset, update_policy
from irl.airl import Discriminator
from utils.evaluate_policy import evaluate_ppo

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    # Init WandB & Parameters
    wandb.init(project='MAML', config={
        'env_id': 'v_1',
        'env_steps': 15000,
        'batchsize_ppo': 32,
        'n_workers': 8,
        'lr_ppo': 3e-4,
        'entropy_reg': 0.01,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5,
        'GAE_lambda': 0.98,
    })
    config = wandb.config

    meta_iteration = 50

    # Create Environment
    # state spaces
    vec_env = VecEnv(config.env_id, config.n_workers)
    # states = vec_env.reset()
    # states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]


    # reset to meta param after reach specific task
    # Initialize Models
    ppo = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo, eps=1e-5)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
    # this doesn't need to be reinitialized
    # Expert 0
    discriminator_0 = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator_0.load_state_dict(torch.load('../saved_models/meta_use/discriminatorn_v_1_9_3_1000_[0,1].pt'))
    ppo_0 = PPO_CNN(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    ppo_0.load_state_dict(torch.load('../saved_models/meta_use/airl_policy_v_1_9_3_1000_[0,1].pt'))
    utop_0 = discriminator_0.estimate_utopia(ppo_0, config)
    print(f'Reward Normalization 1: {utop_0}')
    discriminator_0.set_eval()

    #max_step = int(config.env_steps / config.n_workers)
    max_step = int(config.env_steps)


    def sample_weights(n_samples=1):
        """Sample n_samples weight vectors such that w1 + w2 = 1 and both are in [0,1]."""
        w1_samples = np.random.rand(n_samples)
        w2_samples = 1.0 - w1_samples

        # Reshape to get an array of shape (n_samples, 2)
        return np.column_stack((w1_samples, w2_samples))

    counter = 0
    all_rewards = []
    all_weights = []
    all_meta_gradients = []

    try:
        for iter in range(meta_iteration):
            print("meta_iteration : ", iter)
            # update paras
            # meta_params = [param.clone() for param in ppo.parameters()]
            meta_gradients = []

            #sample tasks
            tasks = sample_weights(5)
            print("Tasks : ", tasks)
            # iterate each task
            for idx, w_v in enumerate(tasks):
                print("Tasks weight : ", w_v)
                # 策略副本
                ppo_copy = copy.deepcopy(ppo)
                optimizer_copy = torch.optim.Adam(ppo_copy.parameters(), lr=config.lr_ppo, eps=1e-5)

                if counter != 0:
                    vec_env = VecEnv(config.env_id, config.n_workers)

                states = vec_env.reset()
                states_tensor = torch.tensor(states).float().to(device)


                # Inner loop, iterate each task in sampled tasks
                for t in tqdm(range(max_step)):

                    # learning rate linearly decreases
                    lr_a_now = config.lr_ppo * (1 - t / max_step)
                    for param_group in optimizer_copy.param_groups:
                        param_group['lr'] = lr_a_now

                    actions, log_probs = ppo.act(states_tensor)
                    next_states, rewards, done, info = vec_env.step(actions)

                    # Fetch AIRL rewards
                    airl_state = torch.tensor(states).to(device).float()
                    airl_next_state = torch.tensor(next_states).to(device).float()
                    airl_rewards_0 = discriminator_0.forward(airl_state, airl_next_state, config.gamma).squeeze(1)
                    airl_rewards_0 = airl_rewards_0.detach().cpu().numpy() * [0 if i else 1 for i in done]

                    vectorized_rewards = [[r[0], airl_rewards_0[i]] for i, r in enumerate(rewards)]
                    scalarized_rewards = [np.dot(w_v, r) for r in vectorized_rewards]

                    train_ready = dataset.write_tuple(states, actions, next_states, scalarized_rewards, done, log_probs,
                                                      rewards, info)

                    # update policy
                    if train_ready:
                        task_grads = update_policy(ppo_copy, dataset, optimizer_copy, config.gamma, config.epsilon, config.ppo_epochs,
                                      config.entropy_reg, config.GAE_lambda, is_maml=True)
                        meta_gradients.append(task_grads)

                        objective_logs = dataset.log_objectives()
                        for i in range(objective_logs.shape[1]):
                            wandb.log({'Task_' + str(idx) + '_Obj_' + str(i): objective_logs[:, i].mean()})
                        for ret in dataset.log_returns():
                            wandb.log({'Task_' + str(idx) + '_Returns': ret})

                        dataset.reset_trajectories()

                    # Prepare state input for next time step
                    states = next_states.copy()
                    states_tensor = torch.tensor(states).float().to(device)

                torch.save(ppo_copy.state_dict(), '../ppo_agent/meta/v_1_' + str(w_v) + '.pt')

                # evaluate policy
                reward_vector = evaluate_ppo(ppo_copy, config)

                #store results
                all_rewards.append(reward_vector)
                all_weights.append(w_v.copy())

                counter += 1


            # 开始元更新准备

            # 计算这些任务的平均梯度
            avg_meta_gradient = [sum(grad) / len(meta_gradients) for grad in zip(*meta_gradients)]

            optimizer.zero_grad()

            # Use the averaged meta-gradients to update the original PPO model
            for param, grad in zip(ppo.parameters(), avg_meta_gradient):
                param.grad = grad

            optimizer.step()

        print("all reward : ", all_rewards)
        print("all weight : ", all_weights)

        torch.save(ppo.state_dict(), '../ppo_agent/meta/v_1_meta_policy_50_iteration' + '.pt')

    except KeyboardInterrupt:
        print("Manual interruption detected...")
        #torch.save(ppo.state_dict(), '../ppo_agent/use/new_test_1_6_6_' + str(config.lambd) + '.pt')
        torch.save(ppo.state_dict(), '../ppo_agent/meta/v_1_meta_policy_50_iteration' + '.pt')

