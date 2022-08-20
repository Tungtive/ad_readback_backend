from util.read_from_mat import get_quat, get_vicon_FCF
from util.read_from_mat import get_data_low_pass as get_data, calibration
import numpy as np
from scipy.spatial.transform import Rotation
from util.preprocess import  resample
import matplotlib.pyplot as plt
from models.fully_connected import Net
from torch import nn
import torch
from models.mvtsgan import MVTSGAN
import tensorflow as tf
from keras import backend as K

def reset_keras():
    sess = tf.keras.backend.get_session()
    tf.keras.backend.clear_session()
    sess.close()
    sess = tf.keras.backend.get_session()
    np.random.seed(1)
    tf.set_random_seed(2)


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
'''really "bad" subjects, 7, 10, 11'''
subjects = [ 2, 3, 4, 5, 6, 7, 8, 10, 11]
test_subs = subjects
for lo in (test_subs):
    leaveout = lo

    training =  np.setdiff1d(subjects, [leaveout])

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

    # fig, axes = plt.subplots(6, 3, sharey=True)
    # for i in range(X.shape[1]):
    #     axes.flatten()[i].plot(X.T[i][:200])
    # plt.show()

    net = MVTSGAN(X.shape[1], output_streams=3, epochs=1, lr=0.00005, batch_len=1000, saturation=False, rand_noise=False,
                  verbose=1)
    g_losses, d_losses = net.fit(X, Y)

    plt.figure()
    plt.plot(g_losses, label = 'gen loss')
    plt.plot(d_losses, label = 'dis loss')
    plt.legend()
    plt.show()


    X = get_RAW_IMU_data(p_path, lo, 1)
    # X-= X.mean(axis=0)
    Y, _ = get_vicon_FCF(p_path, lo, 1)
    Y -= Y.mean(axis=0)
    # Y /= Y.max(axis=0)

    X, Y = resample(X, Y)
    X = X[38:]
    Y = Y[38:]
    # Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())
    Y_ = net.generator.predict(X.reshape((1, X.shape[0], X.shape[1]))).reshape((Y.shape))
    # Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())

    np.savetxt(f'results/csv/gan_s_{lo:02d}_1_0.csv', Y_, delimiter=',')

    fig, axes = plt.subplots(3, sharey=True)

    for i in range(3):
        ax = axes.flatten()[i]
        ax.plot(Y.T[i][:200])
        ax.plot(Y_.T[i][:200 ])
    plt.title(f'subject {lo}')
    plt.savefig(f'results/png/cal2_gan_s_{lo:02d}_1_0.png')
    plt.show()
    plt.close()

