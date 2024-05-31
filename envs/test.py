import numpy as np

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):

        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):

        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)


        x = x / (self.running_ms.std + 1e-8)


        return x

    def reset(self):
        self.R = np.zeros(self.shape)


reward = [5,3,2,4]

reward_scaler = RewardScaling(shape=1, gamma=0.99)


print(reward_scaler(reward))

