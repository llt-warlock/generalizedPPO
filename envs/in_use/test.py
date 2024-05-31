import numpy as np

REGION_0 = np.concatenate(
    [np.arange(11, 15), np.arange(21, 25), np.arange(31, 35), np.arange(41, 45)])

REGION_1 = np.concatenate(
    [np.arange(15, 19), np.arange(25, 29), np.arange(35, 39), np.arange(45, 49)])

REGION_2 = np.concatenate(
    [np.arange(51, 55), np.arange(61, 65), np.arange(71, 75), np.arange(81, 85)])

REGION_3 = np.concatenate(
    [np.arange(55, 59), np.arange(65, 69), np.arange(75, 79), np.arange(85, 89)])


print(REGION_0)
print(REGION_1)
print(REGION_2)
print(REGION_3)

print(np.concatenate([REGION_0, REGION_1]))