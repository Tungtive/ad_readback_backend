from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.preprocess import resample
from models.mvtsgan import MVTSGAN
from scipy.signal import butter, sosfilt
from util.read_from_mat import get_data_low_pass as get_data, get_vicon_FCF


p_path = '/home/unimelb.edu.au/damithasanka/data/treadmill/3_IMU_algorithm_valiation_data(1)/3 IMU algorithm valiation data/IMUData'


def get_RAW_IMU_data(p_path, sub, speed):
    RSH = get_data(p_path, sub, speed, 'RSH')

    RFO = get_data(p_path, sub, speed, 'RFO')

    RTH = get_data(p_path, sub, speed, 'RTH')
    # corrected_acc = calibration(p_path, sub, RSH, RFO)
    # corrected_acc = corrected_acc.reshape((corrected_acc.shape[0], 1))

    # X = np.append(X, get_data(p_path, sub, speed, 'RTH'), axis=1)
    X =np.append(RTH, RSH, axis=1)
    X = np.append(X, RFO, axis = 1)
    # X = np.append(X, RTH, axis = 1)
    # X = np.append(X, corrected_acc, axis=1)
    # X = np.append(X, get_data(p_path, sub, speed, 'LSH'), axis=1)
    # # X = np.append(X, get_data(p_path, sub, speed, 'LTH'), axis=1)
    # X = np.append(X, get_data(p_path, sub, speed, 'LFO'), axis=1)

    return X.astype(np.float32)

def get_gait_data():

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

    # print(raw_IMU.head())
    IMU_data = np.array(raw_IMU)
    vic_data = m['gait05'][0][0]
    vic_angles = []




    for i in range(3):
        vic_angles.append(vic_data[6+i][:,1])
    vic_angles = np.array(vic_angles).T

    X_tr, Y_tr = resample(IMU_data, vic_angles)
    Y_tr -= Y_tr.mean(axis=0)
    mag_rows = [6,7,8,15,16,17,24,25,26]
    sos = butter(1, 5, fs = 100, btype = 'lowpass', output='sos')
    Y_f = sosfilt(sos, Y_tr.T)
    X_tr = sosfilt(sos, X_tr.T).T
    X_tr -= X_tr.mean(axis=0)
    X_tr -= X_tr.min(axis=0)
    X_tr /= X_tr.max(axis=0)
    return X_tr, Y_f.T

subjects = [ 2, 3, 4, 5, 6, 7, 8, 10, 11]

# leaveout = lo

training = subjects# np.setdiff1d(subjects, [leaveout])

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
print(X.shape)

net = MVTSGAN(X.shape[1], output_streams=3, epochs=500, lr=0.00005, batch_len=1000, saturation=False, rand_noise=False,
              verbose=1)
g_losses, d_losses = net.fit(X, Y)

plt.figure()
plt.plot(g_losses, label = 'gen loss')
plt.plot(d_losses, label = 'dis loss')
plt.legend()
plt.show()


X, Y = get_gait_data()
print(X.shape)
print(Y.shape)
# X = X[38:]
# Y = Y[38:]
# Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())
Y_ = net.generator.predict(X.reshape((1, X.shape[0], X.shape[1]))).reshape((Y.shape))
# Y_ = (net.forward(torch.Tensor(X).float()).detach().numpy())

# np.savetxt(f'results/csv/gan_s_{lo:02d}_1_0.csv', Y_, delimiter=',')

fig, axes = plt.subplots(3, sharey=True)

for i in range(3):
    ax = axes.flatten()[i]
    ax.plot(Y.T[i][:500])
    ax.plot(Y_.T[i][:500 ])

# plt.savefig(f'results/png/cal2_gan_s_{lo:02d}_1_0.png')
plt.show()
plt.close()