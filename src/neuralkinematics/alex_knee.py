import pandas as pd
import numpy as np
from util.preprocess import  resample
import matplotlib.pyplot as plt
from models.mvtsgan import MVTSGAN
from scipy.signal import butter,sosfilt
from util.visualizer import vis_imu

def mean_sample(X, rate = 10):
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
        sos = butter(4, 5, fs=100, btype='lowpass', output='sos')
        cand = sosfilt(sos, cand)

        res.append(cand)

    return np.array(res).T


IMU = np.loadtxt('data/alex_imu_no_mag.csv', delimiter=',')
IMU = mean_sample(IMU)

vis_imu(IMU)

VIC = np.loadtxt('data/gait05_knee_IK.csv', delimiter=',')
# IMUF = np.zeros(IMU.shape)
# for v in range(IMU.shape[0]):
#     for l in range(IMU.shape[1]):
#         try:
#             IMUF[v, l] = float(IMU[v, l])
#         except ValueError:
#             print(v,',', l, ',' , IMU[v,l])

print(IMU.shape)
print(VIC.shape)

tr_rat = 0.8


X, Y = resample(IMU, VIC)
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)
tr_sz = int(X.shape[0] * tr_rat)
X_tr = X[:tr_sz]
Y_tr = Y[:tr_sz]
X_te = X[tr_sz:]
Y_te = Y[tr_sz:]

print(X.shape)
print(Y.shape)

fig, axes = plt.subplots(3, sharey=True)
net = MVTSGAN(18, output_streams=3, epochs=2000, lr=0.00001, batch_len=500, saturation=False, rand_noise=False,
                  verbose=1)

net.fit(X_tr, Y_tr)

Y_pr = net.generator.predict(X_te.reshape((1, X_te.shape[0], X_te.shape[1]))).reshape((Y_te.shape))

for i in range(3):
    ax = axes.flatten()[i]
    ax.plot(Y_te.T[i][:400])
    ax.plot(Y_pr.T[i][:400])

plt.savefig(f'results/png/alex_knee.png')
plt.show()
plt.close()
