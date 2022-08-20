from util.read_from_mat import get_quat, get_vicon_FCF
from util.read_from_mat import get_data_low_pass as get_data, calibration
import numpy as np
from scipy.spatial.transform import Rotation
from util.preprocess import  resample
import matplotlib.pyplot as plt
from models.fully_connected import Net
from torch import nn
import torch
from models.conv_autoencoder import DenseAutoencoder
import tensorflow as tf
from keras import backend as K



def get_RAW_IMU_data(p_path, sub, speed):
    RSH = get_data(p_path, sub, speed, 'RSH')

    RFO = get_data(p_path, sub, speed, 'RFO')

    RTH = get_data(p_path, sub, speed, 'RTH')
    # corrected_acc = calibration(p_path, sub, RSH, RFO)
    # corrected_acc = corrected_acc.reshape((corrected_acc.shape[0], 1))

    # X = np.append(X, get_data(p_path, sub, speed, 'RTH'), axis=1)
    X =np.append(RSH, RFO, axis=1)
    # X = np.append(X, RTH, axis = 1)
    # X = np.append(X, corrected_acc, axis=1)
    # X = np.append(X, get_data(p_path, sub, speed, 'LSH'), axis=1)
    # # X = np.append(X, get_data(p_path, sub, speed, 'LTH'), axis=1)
    # X = np.append(X, get_data(p_path, sub, speed, 'LFO'), axis=1)

    return X.astype(np.float32)


p_path = '/home/unimelb.edu.au/damithasanka/data/treadmill/3_IMU_algorithm_valiation_data(1)/3 IMU algorithm valiation data/IMUData'

subjects = [ 2, 3, 4, 5, 6, 7, 8, 10, 11]
test_subs = subjects#[6]
for lo in (test_subs):
    leaveout = lo

    training =  [lo]#np.setdiff1d(subjects, [leaveout])

    # optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)

    for i, s in enumerate(training):

        if not i:
            X = get_RAW_IMU_data(p_path, s, 1)
            # X -= X.mean(axis=0)
            Y, _ = get_vicon_FCF(p_path, s, 1)
            Y -= Y.mean(axis = 0)
            # Y /= Y.max(axis=0)

            X, Y = resample(X, Y)
        else:
            X_ = get_RAW_IMU_data(p_path, s, 1)
            # X_ -= X_.mean(axis=0)
            Y_, _ = get_vicon_FCF(p_path, s, 1)

            X_, Y_ = resample(X_, Y_)
            Y_ -= Y_.mean(axis=0)
            # Y_ /= Y_.max(axis=0)
            X = np.append(X, X_, axis=0)


            Y = np.append(Y, Y_, axis=0)

    fig, axes = plt.subplots(6, 3, sharey=True)
    for i in range(X.shape[1]):
        axes.flatten()[i].plot(X.T[i][:200])

    net = DenseAutoencoder(X.shape[1])
    losses = net.fit(X)



    X = get_RAW_IMU_data(p_path, lo, 1)
    # X-= X.mean(axis=0)
    Y, _ = get_vicon_FCF(p_path, lo, 1)
    Y -= Y.mean(axis=0)
    # Y /= Y.max(axis=0)

    X, Y = resample(X, Y)
    # Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())
    X_al = net.encoder.predict(X.reshape(1, X.shape[0], X.shape[1]))
    Y_ = net.decoder.predict(X_al).reshape(X.shape)
    # Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())


    for i in range(X.shape[1]):
        axes.flatten()[i].plot(Y_.T[i][:200])
    plt.show()

    # fig, axes = plt.subplots(3, sharey=True)

    plt.figure()
    plt.plot(losses, label='loss')
    plt.legend()
    plt.title(f'subject {lo}')
    plt.show()

