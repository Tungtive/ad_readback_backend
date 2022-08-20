from scipy.io import loadmat
import pandas as pd
import scipy.io as scio
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
SPEEDS = pd.read_csv('/home/unimelb.edu.au/damithasanka/PycharmProjects/NeuralKinematics/util/speeds.csv')
import matplotlib.pyplot as plt

def calibration(Ppath, subject, RSH, RFO):
    file_str = 'P{}/static_S{}.mat'

    fname = file_str.format(subject, subject)

    m = loadmat(f'{Ppath}/{fname}')
    # rkf = m['rKF']
    # raf = m['rAF']
    # a_RSH = RSH[:, 3:6]
    # a_RFO = RFO[:, 3:6]

    fig, axes = plt.subplots(3, 1)
    for i in range(3):
        ax = axes.flatten()[i]
        ax.plot(m['ankAngle'][i])
    plt.show()

    Len = len(RSH)
    ang_acc = np.zeros(( Len))
    j1 = -rkf.flatten()
    j2 = -raf.flatten()
    c = np.array([1, 1, 1])
    c = c / np.linalg.norm(c)
    x1 = np.cross(j1, c)
    y1 = np.cross(j1, x1)
    x2 = np.cross(j2, c)
    y2 = np.cross(j2, x2)
    x1 = x1 / np.linalg.norm(x1)
    y1 = y1 / np.linalg.norm(y1)
    x2 = x2 / np.linalg.norm(x2)
    y2 = y2 / np.linalg.norm(y2)
    x1 = x1 / np.linalg.norm(x1)
    y1 = y1 / np.linalg.norm(y1)
    x2 = x2 / np.linalg.norm(x2)
    y2 = y2 / np.linalg.norm(y2)

    for i in range(Len):
        v1 = [np.dot(a_RSH[i], x1), np.dot(a_RSH[ i], y1), np.dot(a_RSH[ i], j1)]
        v2 = [np.dot(a_RFO[ i], x2), np.dot(a_RFO[ i], y2), np.dot(a_RFO[ i], j2)]
        ang_acc[i] = np.arctan2([np.linalg.norm(np.cross(v1, v2))], [np.dot(v1, v2)])

    return ang_acc


def get_calibration(Ppath, subject):
    file_str = 'P{}/DynCal_S{}.mat'

    fname = file_str.format(subject, subject)

    m = loadmat(f'{Ppath}/{fname}')
    rkf = m['rKF']
    raf = m['rAF']
    return np.append(rkf, raf, axis=1).reshape((1, 6))


def get_data(Ppath, subject, speed, loc):

    if subject < 4:
        file_str = 'P{}/S{}_Walk_{}.mat'
    else:
        file_str = 'P{}/Walk_S{}_{}.mat'

    fname = file_str.format(subject, subject, SPEEDS[f'S{subject}'][speed])
    m = loadmat(f'{Ppath}/{fname}')
    X = m[loc]['Acc'][0][0]
    X = np.append(X, m[loc]['Gyro'][0][0], axis=0)
    X = np.append(X, m[loc]['Mag'][0][0], axis=0)

    return X.T


def get_data_low_pass(Ppath, subject, speed, loc):
    sos = butter(1, 5, fs = 148, btype = 'lowpass', output='sos')
    if subject < 4:
        file_str = 'P{}/S{}_Walk_{}.mat'
    else:
        file_str = 'P{}/Walk_S{}_{}.mat'

    fname = file_str.format(subject, subject, SPEEDS[f'S{subject}'][speed])
    m = loadmat(f'{Ppath}/{fname}')
    stream = m[loc]['Acc'][0][0]
    # stream -= stream[:,0].reshape((3,1))
    X = sosfilt(sos, stream)

    stream = m[loc]['Gyro'][0][0]
    # stream -= stream[:,0].reshape((3,1))
    X = np.append(X, sosfilt(sos, stream), axis=0)
    X -= X.min(axis = 1).reshape((X.shape[0], 1))
    X /= X.max(axis=1).reshape((X.shape[0],1))
    stream = m[loc]['Mag'][0][0]
    stream -= stream.min()
    stream /= stream.max()
    # stream -= stream.mean(axis=1).reshape((3, 1))
    X = np.append(X, stream, axis=0)

    return X.T

def get_knee_mocap(subject, speed):
    file_str = '/home/unimelb.edu.au/damithasanka/data/treadmill/knee/S{}_{}.mat'
    fname = file_str.format(subject, SPEEDS[f'S{subject}'][speed])
    m = loadmat(fname)
    return (m['ViconRKnee'].T)

def get_quat(Ppath, subject, speed, loc):
    if subject < 4:
        file_str = 'P{}/S{}_Walk_{}.mat'
    else:
        file_str = 'P{}/Walk_S{}_{}.mat'

    fname = file_str.format(subject, subject, SPEEDS[f'S{subject}'][speed])
    m = loadmat(f'{Ppath}/{fname}')
    X = m[loc]['quat'][0][0]
    return X.T



def get_vicon_FCF(P_path, sub, speed):
    speed_str = SPEEDS[f'S{sub}'][speed]
    rt_str = f'{P_path}/P{sub}/Walk_S{sub}_{speed_str}_rt.mat'
    m = scio.loadmat(rt_str)
    IMU = m['ankAngle'].T
    VIC = m['viconRANK'].T

    return VIC.astype(np.float32), IMU.astype(np.float32)

def get_vicon_FCF_offline(P_path, sub, speed_str):

    rt_str = f'{P_path}/P{sub}/Walk_S{sub}_{speed_str}_offline.mat'
    m = scio.loadmat(rt_str)
    IMU = m['ankAngle'].T
    VIC = m['viconRANK'].T

    return VIC.astype(np.float32), IMU.astype(np.float32)

def get_gait_events(P_path, sub, speed_str):

    rt_str = f'{P_path}/P{sub}/Walk_S{sub}_{speed_str}_rt.mat'
    m = scio.loadmat(rt_str)
    'round(all_data.HS(3:end - 5)*100);'
    HS = np.round(m['HS'][1:-5]*100).astype(int).flatten()
    HO = np.round(m['HO'][1:-5]*100).astype(int).flatten()
    TO = np.round(m['TO'][1:-5]*100).astype(int).flatten()

    return HS, HO, TO

def get_speeds_ordered(P_path, sub):
    flist = glob(f'{P_path}/P{sub}/*.mat')

    speeds = []
    speed_nums = []
    for fen in flist:
        fname = fen.split('/')[-1]
        speed = fname.split(f'S{sub}')[-1].split('_rt')[0][1:].split('_')
        speed_str = '_'.join(speed)
        speed_num = float('.'.join(speed))

        speeds.append(speed_str)
        speed_nums.append(speed_num)

    return np.array(speeds)[list(np.argsort(speed_nums))]