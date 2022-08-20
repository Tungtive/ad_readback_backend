import pandas as pd
import numpy as np
from util.preprocess import  resample
import matplotlib.pyplot as plt
from models.mvtsgan import MVTSGAN
from scipy.signal import butter,sosfilt
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


IMU = np.loadtxt('data/alex_imu_no_mag_s1.csv', delimiter=',')
IMU = mean_sample(IMU)


VIC = np.loadtxt('data/gait05_knee_IK_s1.csv', delimiter=',')
print(IMU.shape)
print(VIC.shape)

tr_rat = 0.8


X, Y = resample(IMU, VIC)
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)
tr_sz = int(X.shape[0] * tr_rat)
X_tr = X#[:tr_sz]
Y_tr = Y#[:tr_sz]


IMU = np.loadtxt('data/alex_imu_no_mag_s2.csv', delimiter=',')
IMU = mean_sample(IMU, rate=1, fs=300)


VIC = np.loadtxt('data/gait05_knee_IK_s2.csv', delimiter=',')


X, Y = resample(IMU, VIC)
print(IMU.shape)
print(VIC.shape)
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)

X_te = X#[tr_sz:]
Y_te = Y#[tr_sz:]

fig, axes = plt.subplots(3, sharey=True)
net = MVTSGAN(18, output_streams=3, epochs=2000, lr=0.00001, batch_len=500, saturation=False, rand_noise=False,
                  verbose=1)

net.fit(X_tr, Y_tr)
X_ft = X[:1000]
Y_ft = Y[:1000]

net.fit(X_ft, Y_ft, fine_tune=True)

Y_pr = net.generator.predict(X_te.reshape((1, X_te.shape[0], X_te.shape[1]))).reshape((Y_te.shape))

for i in range(3):
    ax = axes.flatten()[i]
    ax.plot(Y_te.T[i][:400])
    ax.plot(Y_pr.T[i][:400])

plt.savefig(f'results/png/alex_knee.png')
plt.show()
plt.close()
