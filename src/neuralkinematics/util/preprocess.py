import numpy as np
SEED = 5
rst = np.random.RandomState(SEED)
def resample(IMU, VIC):
    target = VIC.shape[0]
    perm = rst.permutation(IMU.shape[0])
    perm = perm[:target]
    ixs = np.sort(perm)

    return IMU[ixs], VIC