import numpy as np

def generate_uniform_samples_two(n_samples):
    """
    Generates samples where one randomly chosen objective has a fixed preference of 0.0,
    and the preferences for the other two objectives are sampled uniformly such that they sum to 1.0.

    :param n_samples: Number of samples to generate.
    :return: A numpy ndarray with shape (n_samples, 3).
    """
    samples = np.zeros((n_samples, 3))

    for i in range(n_samples):
        # Randomly choose an index to fix at 0.0
        fixed_index = np.random.choice(3)

        # Sample a value for one of the remaining objectives
        value = np.random.uniform(0, 1)

        # Assign the values to the two non-fixed indices
        non_fixed_indices = [idx for idx in range(3) if idx != fixed_index]
        samples[i, non_fixed_indices] = [value, 1 - value]

        # Shuffle the non-fixed indices to ensure random assignment
        np.random.shuffle(samples[i, non_fixed_indices])

    return samples

# Example usage:
# Generate 5 samples with a random objective fixed at 0.0
samples = generate_uniform_samples_two(5)
print(samples)
