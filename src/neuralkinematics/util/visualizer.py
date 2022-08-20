import matplotlib.pyplot as plt

def vis_imu(X):
    fig, axes = plt.subplots(3, 6, figsize= (10,10))
    for i in range(X.shape[1]):
        ax = axes.flatten()[i]
        ax.plot(X[:200, i])

    plt.show()