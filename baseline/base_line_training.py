import numpy as np

from adaption.generalization_ppo.baseline.ppo_vanila_train import training

if __name__ == '__main__':

    preference_spaces = [[0.66, 0.32, 0.02]]

    for i, preference in enumerate(preference_spaces):
        #weight_vectors = [preference]
        weight_vectors = preference
        #weight_vectors = np.repeat(weight_vectors, 16, axis=0)

        training(str(preference), weight_vectors)

    #training('[0.9, 0.1]', [[0.9, 0.1]])
