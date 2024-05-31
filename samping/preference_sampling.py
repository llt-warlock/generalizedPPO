import random

import numpy as np


def generate_fixed_samples_two(n_samples):
    """
    Generates samples from a fixed set of vectors:
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], and [0.0, 0.0, 1.0].

    :param n_samples: Number of samples to generate.
    :return: A numpy ndarray with shape (n_samples, 3).
    """
    # Define the fixed vectors
    fixed_vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    # Randomly select from the fixed vectors
    indices = np.random.choice(len(fixed_vectors), n_samples)
    samples = fixed_vectors[indices]

    return samples


def generate_samples(n_samples, n_elements):
    """
    Generates samples from a uniform distribution where each sample sums up to 1.

    :param n_samples: Number of samples to generate.
    :param n_elements: Number of elements in each sample.
    :return: A numpy ndarray with shape (n_samples, n_elements).
    """
    # Generate random samples
    samples = np.random.rand(n_samples, n_elements)

    # Normalize each sample so that the sum of elements equals to 1
    samples /= samples.sum(axis=1)[:, np.newaxis]

    return samples



def generate_samples_with_granularity(n_samples, n_elements, granularity):
    """
    Generate samples with a specified granularity that sum up to 1.

    :param n_samples: Number of samples to generate.
    :param n_elements: Number of elements in each sample.
    :param granularity: The granularity of each sample.
    :return: A numpy ndarray with shape (n_samples, n_elements).
    """
    assert granularity > 0, "Granularity must be a positive number."

    # Calculate the total number of intervals
    intervals = int(1 / granularity) + 1

    # Generate all possible combinations
    all_combinations = np.array(np.meshgrid(*[np.arange(intervals)] * n_elements)).T.reshape(-1, n_elements)
    all_combinations = granularity * all_combinations
    valid_combinations = all_combinations[np.isclose(all_combinations.sum(axis=1), 1)]

    # Randomly select the specified number of samples
    indices = np.random.choice(len(valid_combinations), size=n_samples, replace=True)
    samples = valid_combinations[indices]

    return samples