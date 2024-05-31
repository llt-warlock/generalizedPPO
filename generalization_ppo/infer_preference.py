import numpy as np
import torch

from adaption.evaluation.base_evaluation_generalized import evaluate_ppo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

        reward_plus = evaluate_ppo(ppo, preference_plus, torch.tensor(preference_plus).flost().to(device), [1,0], config, n_eval=100)
        reward_minus = evaluate_ppo(ppo, preference_plus, torch.tensor(preference_minus).flost().to(device), [1,0], config, n_eval=100)

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


