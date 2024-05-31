import numpy as np
import torch

from envs.in_use.v_3 import v_3_idx_to_scalar, V_3_LEFT_REGION, V_3_RIGHT_REGION


def find_agent_positions(env_matrix):
    """
    Get the positions of the agent in all parallel environments.

    Args:
    env_matrix (torch.Tensor): Input tensor of shape [n_env, n_layers, grid_width, grid_height].

    Returns:
    torch.Tensor: A tensor containing the agent positions for all environments, with shape [n_env, 2].
    """
    # Reshape the input tensor for easier indexing
    n_env, n_layers, grid_width, grid_height = env_matrix.shape
    reshaped_matrix = env_matrix.view(n_env, n_layers, -1)  # Reshape to [n_env, n_layers, grid_width * grid_height]

    # Use advanced indexing to extract agent positions
    agent_positions = reshaped_matrix[:, 1, :].argmax(dim=1)

    # Convert the flat positions to (x, y) coordinates
    agent_x = agent_positions // grid_width
    agent_y = agent_positions % grid_width

    # Combine x and y coordinates into a single tensor
    agent_positions = torch.stack((agent_x, agent_y), dim=1)

    return agent_positions

def convert_positions_to_scalars(agent_positions):
    """Converts a tensor of agent positions to scalar values using v_3_idx_to_scalar"""
    # Assuming agent_positions is a tensor of shape [n_env, 2] where each row is [row, col]
    scalar_positions = [v_3_idx_to_scalar(pos[0], pos[1]) for pos in agent_positions]
    return torch.tensor(scalar_positions)


def convert_to_region(scalar_position):
    regions = np.zeros((len(scalar_position), 2), dtype=int)

    #scalar_position.detach().cpu()

    for i, pos in enumerate(scalar_position):
        # Move tensor to CPU and convert it to a Python int
        pos_cpu = pos.cpu().item()
        print("pos : ", pos)
        # Determine the region for each position using numpy operations
        if pos_cpu in V_3_LEFT_REGION:
            regions[i] = [1, 0]
        elif pos_cpu in V_3_RIGHT_REGION:
            regions[i] = [0, 1]

    return regions
