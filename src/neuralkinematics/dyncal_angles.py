from util.read_from_mat import get_quat, get_vicon_FCF
from util.read_from_mat import get_data_low_pass as get_data, get_calibration
import numpy as np
from scipy.spatial.transform import Rotation
from util.preprocess import  resample
import matplotlib.pyplot as plt
from models.fully_connected import Net
from torch import nn
import torch
from pyquaternion import Quaternion
from models.calibration_based_gan import MVTSGAN

def get_quat_and_conj(p_path, sub, speed, loc):
    X = get_quat(p_path, sub, speed, loc)
    X_conj = np.array([Quaternion(x).conjugate.elements for x in X])
    return np.append(X, X_conj, axis=1)


def get_RAW_IMU_data(p_path, sub, speed):
    X = get_data(p_path, sub, speed, 'RSH')
    X = np.append(X, get_data(p_path, sub, speed, 'RTH'), axis=1)
    X = np.append(X, get_data(p_path, sub, speed, 'RFO'), axis=1)
    X = np.append(X, get_data(p_path, sub, speed, 'LSH'), axis=1)
    X = np.append(X, get_data(p_path, sub, speed, 'LTH'), axis=1)
    X = np.append(X, get_data(p_path, sub, speed, 'LFO'), axis=1)

    return X.astype(np.float32)


loss_fn = nn.MSELoss()
# net = Net(input_size=54, output_size=3)
# print(net.fc1.weight)


p_path = '/home/student.unimelb.edu.au/senanayaked/data/treadmill/3_IMU_algorithm_valiation_data(1)/3 IMU algorithm valiation data/IMUData'

subjects = [2, 3,4, 5, 6, 7, 8, 10, 11]

for lo in (subjects):
    leaveout = lo

    training = np.setdiff1d(subjects, [leaveout])
    net = MVTSGAN(54, output_streams=3, epochs=500, lr=0.000001, batch_len=200, saturation=False, rand_noise=True,
                  verbose=1)  # Net(12, 3)
    # optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)

    for i, s in enumerate(training):

        if not i:
            X = get_RAW_IMU_data(p_path, s, 1)
            cal = get_calibration(p_path, s)
            Y, _ = get_vicon_FCF(p_path, s, 1)
            Y -= Y.mean(axis = 0)
            # Y /= Y.max(axis=0)

            X, Y = resample(X, Y)
            cal = np.repeat(cal, X.shape[0], axis=0)

        else:
            X_ = get_RAW_IMU_data(p_path, s, 1)
            Y_, _ = get_vicon_FCF(p_path, s, 1)
            cal_ = get_calibration(p_path, s)

            X_, Y_ = resample(X_, Y_)
            cal_ = np.repeat(cal_, X_.shape[0], axis=0)

            Y_ -= Y_.mean(axis=0)
            # Y_ /= Y_.max(axis=0)
            X = np.append(X, X_, axis=0)
            cal = np.append(cal, cal_, axis=0)

            Y = np.append(Y, Y_, axis=0)


    net.fit(X, Y, cal)
    # for i in range(200):
    #
    #     batch_size = 50
    #
    #     n_batches = X.shape[0] // batch_size + 1
    #     # if i and not i %2000:
    #     #     batch_size //= 2
    #     loss = 0
    #     for b in range(n_batches):
    #         Xb = X[b * batch_size: (b + 1) * batch_size]
    #         loss += net.train_loop(Xb.reshape([1, Xb.shape[0], Xb.shape[1]]), Y[b * batch_size: (b + 1) * batch_size],
    #                               loss_fn, optimizer)
    #         # net.fc1(torch.Tensor(Xb))
    #     print(f'loss = {loss:.4f}')

    X = get_RAW_IMU_data(p_path, lo, 1)

    Y, _ = get_vicon_FCF(p_path, lo, 1)
    Y -= Y.mean(axis=0)
    # Y /= Y.max(axis=0)

    X, Y = resample(X, Y)
    cal = get_calibration(p_path, lo)
    cal = np.repeat(cal, X.shape[0], axis=0)

    # Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())
    Y_ = net.predict(X.reshape((1, X.shape[0], X.shape[1])), cal)#.generator.predict(X.reshape((1, X.shape[0], X.shape[1]))).reshape((Y.shape))
    # Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())

    np.savetxt(f'results/csv/cal_gan_s_{lo:02d}_1_0.csv', Y_, delimiter=',')

    fig, axes = plt.subplots(3, sharey=True)

    for i in range(3):
        ax = axes.flatten()[i]
        ax.plot(Y.T[i][:200])
        ax.plot(Y_.T[i][:200])

    plt.savefig(f'results/png/cal_gan_s_{lo:02d}_1_0.png')
    plt.show()
    plt.close()
    break

