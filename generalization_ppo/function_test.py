import numpy as np

def generate_custom_samples(n_samples):
    """
    Generates samples from a uniform distribution where each sample sums up to 1,
    with a predefined 0.0 in each sample at varying positions.

    :param n_samples: Number of samples to generate.
    :return: A numpy ndarray with shape (n_samples, 3).
    """
    # Initialize array
    samples = np.zeros((n_samples, 3))

    for i in range(n_samples):
        # Randomly generate two elements since one is always 0.0
        random_elements = np.random.rand(2)

        # Decide the position of 0.0 in each sample
        zero_position = i % 3

        # Fill the non-zero elements
        samples[i, :] = np.insert(random_elements, zero_position, 0.0)

        # Normalize so that sum equals 1
        samples[i, :] /= samples[i, :].sum()

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

# Example usage
n_samples = 5   # Number of samples
n_elements = 3  # Number of elements in each sample

generated_samples = generate_samples(n_samples, n_elements)

print(generated_samples)

print(generate_custom_samples(10))

