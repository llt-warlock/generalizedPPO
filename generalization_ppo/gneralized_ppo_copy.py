import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import torch.nn.init as init
# Use GPU if available
from torch.utils.data import DataLoader

from adaption.auxiliary import preference_context_generate
from adaption.generalization_ppo.samping.preference_sampling import generate_samples_two
from envs.in_use.v_4 import LEFT_REGION, RIGHT_REGION, REGION_0, REGION_1, REGION_2, REGION_3
from utils.region import region_check, rolling_window_sum
from envs import driving
from envs.in_use.v_4 import idx_to_scalar, LEFT_REGION, RIGHT_REGION
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class PPO_CNN(nn.Module):
    def __init__(self, state_shape, in_channels, n_actions, weight, contexts):
        super(PPO_CNN, self).__init__()
        #print("state shape : ", state_shape)
        # General Parameters
        self.state_shape = state_shape
        self.in_channels = in_channels + contexts

        weight_dim = weight

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the flattened CNN output
        cnn_output_size = 128 * (state_shape[0]) * (state_shape[1])

        # Preference conditioning layer
        self.pref_fc = nn.Linear(weight_dim, 16)

        # Shared FC Layer (adjust input size to account for weight processing)
        self.shared_fc_1 = nn.Linear(cnn_output_size, 128)
        #self.shared_fc_2 = nn.Linear(256, 128)


        # actor network
        self.policy_fc = nn.Linear(in_features=128 + 16, out_features=64)
        self.policy_head = nn.Linear(in_features=64, out_features=n_actions)

        # critic network
        self.value_fc = nn.Linear(in_features=128 + 16, out_features=64)
        self.value_head = nn.Linear(in_features=64, out_features=2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        orthogonal_init(self.conv1)
        orthogonal_init(self.conv2)
        orthogonal_init(self.conv3)
        orthogonal_init(self.pref_fc)
        orthogonal_init(self.shared_fc_1)
        #orthogonal_init(self.shared_fc_2)
        orthogonal_init(self.policy_fc)
        orthogonal_init(self.policy_head)
        orthogonal_init(self.value_fc)
        orthogonal_init(self.value_head)

    def forward(self, x, weights):
        x = x.view(-1, self.in_channels, self.state_shape[0], self.state_shape[1])
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.shape[0], -1)
        x = self.tanh(self.shared_fc_1(x))
        #x = self.tanh(self.shared_fc_2(x))

        #print("weight : ", weights.size())

        weights = self.tanh(self.pref_fc(weights))

        x_with_weights = torch.cat((x, weights), dim=1).float().to(device)

        # actor
        policy = self.tanh(self.policy_fc(x_with_weights))
        policy = self.softmax(self.policy_head(policy))

        # critic
        value = self.tanh(self.value_fc(x_with_weights))
        value = self.value_head(value)

        return policy, value

    def act(self, state, weights, contexts=None):

        action_probabilities, _ = self.forward(state, weights)

        m = Categorical(action_probabilities)
        action = m.sample()
        return action.detach().cpu().numpy(), m.log_prob(action).detach().cpu().numpy()

    def evaluate_trajectory(self, tau, weights):
        trajectory_states = torch.tensor(np.array(tau['states'])).float().to(device)
        trajectory_actions = torch.tensor(tau['actions']).to(device)
        trajectory_next_states = torch.tensor(np.array(tau['next_states'])).float().to(device)
        # print("trajectory state length: ", trajectory_states.shape)
        # print("trajectory next state length: ", trajectory_next_states.shape)
        # use network

        contexts = preference_context_generate(weights.size(0), weights.size(1), weights)
        states_augmentation = torch.cat((trajectory_states, contexts), dim=1)
        states_augmentation_ = torch.cat((trajectory_next_states, contexts), dim=1)

        action_probabilities, critic_values = self.forward(states_augmentation, weights)
        _, critic_values_ = self.forward(states_augmentation_, weights)

        dist = Categorical(action_probabilities)

        action_entropy = dist.entropy().mean()

        action_log_probabilities = dist.log_prob(trajectory_actions)

        return action_log_probabilities, torch.squeeze(critic_values), torch.squeeze(critic_values_), action_entropy


def g_clip(epsilon, A):
    return torch.tensor([1 + epsilon if i else 1 - epsilon for i in A >= 0]).to(device) * A

def n_step_return_fourR(states, reward_vectors, n):
    """Calculate n-step returns"""

    positions = [idx_to_scalar(*region_check(state[1])) for state in states]

    # Create indicators for region 0, 1, 2 ,3
    is_0 = torch.tensor([1.0 if pos in REGION_0 else 0.0 for pos in positions], device=device)
    is_1 = torch.tensor([1.0 if pos in REGION_1 else 0.0 for pos in positions], device=device)
    is_2 = torch.tensor([1.0 if pos in REGION_2 else 0.0 for pos in positions], device=device)
    is_3 = torch.tensor([1.0 if pos in REGION_3 else 0.0 for pos in positions], device=device)

    # Rolling window sum for n steps
    w_0_values = rolling_window_sum(is_0, n)
    w_1_values = rolling_window_sum(is_1, n)
    w_2_values = rolling_window_sum(is_2, n)
    w_3_values = rolling_window_sum(is_3, n)

    # Compute the sum of values from both A and B at each timestep
    sum_values = w_0_values + w_1_values + w_2_values + w_3_values

    # Compute the weights
    weights_0 = w_0_values / sum_values
    weights_1 = w_1_values / sum_values
    weights_2 = w_2_values / sum_values
    weights_3 = w_3_values / sum_values

    # Combine the weights to get the final tensor
    combined_weights = torch.stack((weights_0, weights_1, weights_2, weights_3), dim=1)
    expanded_weights = torch.cat((torch.zeros((len(combined_weights), 1), device=device), combined_weights), dim=1)

    dot_product_result = (expanded_weights * reward_vectors).sum(dim=1)

    #print("dot product : ", dot_product_result)
    return dot_product_result

def n_step_return(states, reward_vectors, n):
    """Calculate n-step returns"""

    positions = [idx_to_scalar(*region_check(state[1])) for state in states]

    #Create indicators for region A and B
    is_A = torch.tensor([1.0 if pos in LEFT_REGION else 0.0 for pos in positions], device=device)
    is_B = torch.tensor([1.0 if pos in RIGHT_REGION else 0.0 for pos in positions], device=device)

    # print("is A ", is_A)
    # print("is B ", is_B)

    # Rolling window sum for n steps
    w_A_values = rolling_window_sum(is_A, n)
    w_B_values = rolling_window_sum(is_B, n)

    # Compute the sum of values from both A and B at each timestep
    sum_values = w_A_values + w_B_values

    # Compute the weights
    weights_A = w_A_values / sum_values
    weights_B = w_B_values / sum_values


    # Combine the weights to get the final tensor
    combined_weights = torch.stack((weights_A, weights_B), dim=1)
    #expanded_weights = torch.cat((torch.zeros((len(combined_weights), 1), device=device), combined_weights), dim=1)


    dot_product_result = (combined_weights * reward_vectors).sum(dim=1)

    #print("dot product : ", dot_product_result)
    return dot_product_result


def update_policy(ppo, dataset, optimizer, gamma, epsilon, n_epochs, entropy_reg, GAE_lambda):

    for epoch in range(n_epochs):
        batch_loss = 0
        value_loss = 0

        for i, tau in enumerate(dataset.trajectories):

            for j in range(2)：

                weight_vector = generate_samples_two(1, 2)
                weight_vectors_tensor = torch.tensor(weight_vector).float().to(device)

                states = torch.tensor(np.array(tau['states'])).detach().to(device)
                #raw_rewards = torch.tensor(tau['rewards']).detach().to(device)

                dw = torch.tensor(tau['dw']).detach().to(device)
                done = torch.tensor(tau['do']).detach().to(device)
                weights = torch.stack(tau['weights']).detach().to(device)
                reward_vectors_raw = torch.tensor(np.array(tau['logs'])).to(device)


                # reward of different objectives for two obj
                rewards = reward_vectors_raw[:, 0] * weights[:, 0] + reward_vectors_raw[:, 1] * weights[:, 1]

                # for three objs
                # rewards = reward_vectors_raw[:, 0] * weights[:, 0] + reward_vectors_raw[:, 1] * weights[:, 1] + reward_vectors_raw[:, 2] * weights[:, 2]

                normalized_reward = ((rewards - rewards.mean()) / (rewards.std() + 1e-5))

                # print("reward vector : ", reward_vectors_obj)
                # print("weight : ", weights)

                # GAE advantage calculation
                adv = []
                gae = 0

                action_log_probabilities, critic_values, critic_values_, action_entropy = ppo.evaluate_trajectory(tau, weights)

                likelihood_ratios = torch.exp(action_log_probabilities - torch.tensor(tau['log_probs']).detach().to(device))

                # for two objectives
                weighted_values = weights[:, 0] * critic_values[:, 0] + weights[:, 1] * critic_values[:, 1]
                weighted_values_ = weights[:, 0] * critic_values_[:, 0] + weights[:, 1] * critic_values_[:, 1]

                # for three objectives
                # weighted_values = weights[:, 0] * critic_values[:, 0] + weights[:, 1] * critic_values[:, 1] + weights[:, 2] * critic_values[:, 2]
                # weighted_values_ = weights[:, 0] * critic_values_[:, 0] + weights[:, 1] * critic_values_[:, 1] + weights[:, 2] * critic_values_[:, 2]

                deltas = normalized_reward + gamma * (1.0 - dw) * weighted_values_ - weighted_values
                for delta, d in zip(reversed(deltas.flatten()), reversed(done.flatten())):
                    gae = delta + gamma * GAE_lambda * gae * (1.0 - d)
                    adv.insert(0, gae)
                # compute the advantages
                adv = torch.tensor(adv, dtype=torch.float).to(device)
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
                v_target = adv + weighted_values

                # clipped losses
                clipped_losses = -torch.min(likelihood_ratios * adv, g_clip(epsilon, adv))

                # value losses
                value_loss += torch.mean((v_target - weighted_values) ** 2)
                # batch losses
                batch_loss += torch.mean(clipped_losses) - entropy_reg * action_entropy

        overall_loss = (batch_loss + value_loss) / dataset.batch_size

        optimizer.zero_grad()
        overall_loss.backward()

        # Gradient clip
        torch.nn.utils.clip_grad_norm_(ppo.parameters(), 0.5)
        optimizer.step()




# def update_policy(ppo, dataset, optimizer, gamma, epsilon, n_epochs, entropy_reg, GAE_lambda, n_step=None, contexts=None, is_maml=None):
#     #accumulated_gradients = [torch.zeros_like(param) for param in ppo.parameters()]
#
#     #n_step_returns = []
#
#     for epoch in range(n_epochs):
#         batch_loss = 0
#         value_loss = 0
#
#         for i, tau in enumerate(dataset.trajectories):
#             print("**************************************")
#             states = torch.tensor(np.array(tau['states'])).detach().to(device)
#             #raw_rewards = torch.tensor(tau['rewards']).detach().to(device)
#
#             dw = torch.tensor(tau['dw']).detach().to(device)
#             done = torch.tensor(tau['do']).detach().to(device)
#             weights = torch.stack(tau['weights']).detach().to(device)
#             objective_reward = torch.tensor(tau['logs']).detach().to(device)
#
#             print("objective_reward : ", objective_reward.size(), "  ", objective_reward)
#             tensor_without_first_column = objective_reward[:, 1:]
#             print("after : ", tensor_without_first_column)
#
#             if n_step is None:
#                 rewards = torch.tensor(tau['rewards']).detach().to(device)
#             elif n_step is not None:
#                 reward_vectors = torch.tensor(np.array(tau['experts'])).detach().to(device)
#                 #reward_vectors = torch.tensor(np.array(tau['logs'])).detach().to(device)
#                 rewards = n_step_return(states, reward_vectors, n_step)
#                 dataset.write_n_step_returns(rewards.sum().item())
#
#             normalized_reward = ((rewards - rewards.mean()) / (rewards.std() + 1e-5))
#
#             # GAE advantage calculation
#             adv = []
#             gae = 0
#
#             if contexts is not None:
#                 action_log_probabilities, critic_values_1, critic_values_2, critic_values_1_, critic_values_2_, action_entropy = ppo.evaluate_trajectory(tau, weights, contexts)
#             else:
#                 action_log_probabilities, critic_values_1, critic_values_2, critic_values_1_, critic_values_2_, action_entropy = ppo.evaluate_trajectory(tau, weights)
#
#             likelihood_ratios = torch.exp(action_log_probabilities - torch.tensor(tau['log_probs']).detach().to(device))
#
#             critic_value_tensor = torch.stack((critic_values_1, critic_values_2), dim=1)
#             weighted_critic_value = torch.sum(critic_value_tensor * weights, dim=1)
#
#             critic_value_tensor_ = torch.stack((critic_values_1_, critic_values_2_), dim=1)
#             weighted_critic_value_ = torch.sum(critic_value_tensor_ * weights, dim=1)
#
#             print("weighted critic value size : ", critic_value_tensor.size(), "  ", weights.size())
#             print("weighted critic value : ", critic_value_tensor, "  ", weights)
#             print("final critic value : ", weighted_critic_value)
#
#             print("weighted critic next value size : ", critic_value_tensor_.size(), "  ", weights.size())
#             print("weighted critic next value : ", critic_value_tensor_, "  ", weights)
#             print("final critic value next : ", weighted_critic_value_)
#
#             deltas = normalized_reward + gamma * (1.0 - dw) * weighted_critic_value_ - weighted_critic_value
#             for delta, d in zip(reversed(deltas.flatten()), reversed(done.flatten())):
#                 gae = delta + gamma * GAE_lambda * gae * (1.0 - d)
#                 adv.insert(0, gae)
#             # compute the advantages
#             adv = torch.tensor(adv, dtype=torch.float).to(device)
#             adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
#             weighted_v_target = adv + weighted_critic_value
#
#             clipped_losses = -torch.min(likelihood_ratios * adv, g_clip(epsilon, adv))
#             # clipped losses
#
#             batch_loss += torch.mean(clipped_losses) - entropy_reg * action_entropy
#             value_loss += torch.mean((weighted_v_target - weighted_critic_value) ** 2)
#
#         overall_loss = (batch_loss + value_loss) / dataset.batch_size
#
#         optimizer.zero_grad()
#         overall_loss.backward()
#
#         # Gradient clip
#         torch.nn.utils.clip_grad_norm_(ppo.parameters(), 0.5)
#         optimizer.step()
#


