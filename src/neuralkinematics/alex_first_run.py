from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.preprocess import resample
from models.mvtsgan import MVTSGAN
from scipy.signal import butter, sosfilt

m = loadmat('/home/unimelb.edu.au/damithasanka/data/alex/gait_all.mat')

'''load raw IMU data'''
''' 
'TS_01344_Accel', 'Var64', 'Var65',
'TS_01344_Gyro', Var67', 'Var68',
'TS_01344_Mag', 'Var70', 'Var71', 

'TS_01356_Accel', 'Var79', 'Var80',
'TS_01356_Gyro', Var82', 'Var83',
'TS_01356_Mag', 'Var84', 'Var85',

'TS_01368_Accel', 'Var94', 'Var95',
'TS_01368_Gyro', Var97', 'Var98',
'TS_01368_Mag', 'Var100', 'Var101',

'''

raw_data_cols = ['TS_01344_Accel', 'Var64', 'Var65',
'TS_01344_Gyro', 'Var67', 'Var68',
'TS_01344_Mag', 'Var70', 'Var71',

'TS_01356_Accel', 'Var79', 'Var80',
'TS_01356_Gyro', 'Var82', 'Var83',
'TS_01356_Mag', 'Var85', 'Var86',

'TS_01368_Accel', 'Var94', 'Var95',
'TS_01368_Gyro', 'Var97', 'Var98',
'TS_01368_Mag', 'Var100', 'Var101']

raw_IMU = pd.read_csv('/home/unimelb.edu.au/damithasanka/data/alex/raw_gait05.csv', usecols=raw_data_cols)

print(raw_IMU.head())
IMU_data = np.array(raw_IMU)
vic_data = m['gait05'][0][0]
vic_angles = []

fig, axes = plt.subplots(3, sharey=True)



for i in range(3):
    vic_angles.append(vic_data[6+i][:,1])
vic_angles = np.array(vic_angles).T

X_tr, Y_tr = resample(IMU_data, vic_angles)
Y_tr -= Y_tr.mean(axis=0)

sos = butter(1, 5, fs = 100, btype = 'lowpass', output='sos')
Y_f = sosfilt(sos, Y_tr.T)
X_tr = sosfilt(sos, X_tr.T).T
X_tr -= X_tr.mean(axis=0)
X_tr -= X_tr.min(axis=0)
X_tr /= X_tr.max(axis=0)
for i in range(3):
    ax = axes.flatten()[i]
    ax.plot(Y_tr.T[i][:250])
    ax.plot(Y_f[i][:250])

# plt.title(f'subject {lo}')
# plt.savefig(f'results/png/cal2_gan_s_{lo:02d}_1_0.png')
plt.show()
Y_tr = Y_f.T
print(Y_tr.shape)
X_te = X_tr[100: 500]
Y_te = Y_tr[100: 500]

net = MVTSGAN(X_tr.shape[1], output_streams=3, epochs=1500, lr=0.00001, batch_len=500, saturation=False, rand_noise=False,
              verbose=1)
g_losses, d_losses = net.fit(X_tr, Y_tr)

plt.figure()
plt.plot(g_losses, label='gen loss')
plt.plot(d_losses, label='dis loss')
plt.legend()
plt.show()


# Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())
Y_ = net.generator.predict(X_te.reshape((1, X_te.shape[0], X_te.shape[1]))).reshape((Y_te.shape))
# Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())


fig, axes = plt.subplots(3, sharey=True)

for i in range(3):
    ax = axes.flatten()[i]
    ax.plot(Y_te.T[i])
    ax.plot(Y_.T[i])
# plt.title(f'subject {lo}')
# plt.savefig(f'results/png/cal2_gan_s_{lo:02d}_1_0.png')
plt.show()
plt.close()