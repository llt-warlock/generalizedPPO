import random

import numpy as np
accumulated_gradients = [2,4,6]



average_gradients = [acc_grad / 2 for acc_grad in accumulated_gradients]

print(average_gradients)