from util.read_from_mat import get_quat, get_knee_mocap
from util.read_from_mat import get_data_low_pass as get_data, get_calibration
import numpy as np
from scipy.spatial.transform import Rotation
from util.preprocess import  resample
import matplotlib.pyplot as plt
from models.fully_connected import Net
from torch import nn
import torch
from models.mvtsgan import MVTSGAN
from scipy.signal import sosfilt, butter
from util.visualizer import vis_imu

def mean_sample(X, rate = 10, fs = 100):
    out = []
    n_sets = len(X)//rate

    for i in range(n_sets):
        st = i * rate
        et = (i+1) * rate
        cand = X[st:min(et, len(X))].mean(axis=0)


        out.append(cand)
    out = np.array(out)
    res = []

    for i in range(X.shape[1]):

        cand = out[:, i]
        cand -= cand.min()
        cand /= cand.max()
        sos = butter(4, 5, fs=fs, btype='lowpass', output='sos')
        cand = sosfilt(sos, cand)

        res.append(cand)

    return np.array(res).T
def get_RAW_IMU_data(p_path, sub, speed):
    X = get_data(p_path, sub, speed, 'RSH')
    X = np.append(X, get_data(p_path, sub, speed, 'RTH'), axis=1)
    # X = np.append(X, get_data(p_path, sub, speed, 'RFO'), axis=1)
    # X = np.append(X, get_data(p_path, sub, speed, 'LSH'), axis=1)
    # # X = np.append(X, get_data(p_path, sub, speed, 'LTH'), axis=1)
    # X = np.append(X, get_data(p_path, sub, speed, 'LFO'), axis=1)

    return X.astype(np.float32)


loss_fn = nn.MSELoss()
# net = Net(input_size=54, output_size=3)
# print(net.fc1.weight)


p_path = '/home/unimelb.edu.au/damithasanka/data/treadmill/3_IMU_algorithm_valiation_data(1)/3 IMU algorithm valiation data/IMUData'

subjects = [2, 3, 6]# 7, 8, 10, 11]
test_subs = [2]
for lo in (test_subs):
    leaveout = lo

    training =  np.setdiff1d(subjects, [leaveout])
    net = MVTSGAN(18, output_streams=3, epochs=2000, lr=0.000001, batch_len=500, saturation=False, rand_noise=False,
                  verbose=1)  # Net(12, 3)
    # optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)

    for i, s in enumerate(training):

        if not i:
            X = get_RAW_IMU_data(p_path, s, 1)
            X -= X.mean(axis=0)
            Y = get_knee_mocap(s, 1)
            Y -= Y.mean(axis = 0)
            # Y /= Y.max(axis=0)

            X, Y = resample(X, Y)
        else:
            X_ = get_RAW_IMU_data(p_path, s, 1)
            X_ -= X_.mean(axis=0)
            Y_ = get_knee_mocap(s, 1)

            X_, Y_ = resample(X_, Y_)
            Y_ -= Y_.mean(axis=0)
            # Y_ /= Y_.max(axis=0)
            X = np.append(X, X_, axis=0)


            Y = np.append(Y, Y_, axis=0)

    # fig, axes = plt.subplots(6, 3, sharey=True)
    # for i in range(X.shape[1]):
    #     axes.flatten()[i].plot(X.T[i][:200])
    # plt.show()

    net.fit(X, Y)

    IMU = np.loadtxt('data/alex_imu_no_mag_s2.csv', delimiter=',')
    IMU = mean_sample(IMU, rate=1, fs=300)

    VIC = np.loadtxt('data/gait05_knee_IK_s2.csv', delimiter=',')

    X, Y = resample(IMU, VIC)
    print(IMU.shape)
    print(VIC.shape)
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    X_te = X  # [tr_sz:]
    Y_te = Y  # [tr_sz:]
    # Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())
    Y_ = net.generator.predict(X.reshape((1, X.shape[0], X.shape[1]))).reshape((Y.shape))
    # Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())

    np.savetxt(f'results/csv/knee_gan_s_{lo:02d}_1_0.csv', Y_, delimiter=',')

    fig, axes = plt.subplots(3, sharey=True)

    for i in range(3):
        ax = axes.flatten()[i]
        ax.plot(Y.T[i][:400])
        ax.plot(Y_.T[i][:400])

    plt.savefig(f'results/png/knee_gan_s_{lo:02d}_1_0.png')
    plt.show()
    plt.close()

