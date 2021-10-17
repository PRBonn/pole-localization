import matplotlib.pyplot as plt
import numpy as np

def plot_global_map(globalmapfile):
    data = np.load(globalmapfile)
    x, y = data['poleparams'][:, :2].T
    plt.clf()
    plt.scatter(x, y, s=1, c='b', marker='.')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(globalmapfile[:-4] + '.svg')

plot_global_map('data/pole-dataset/NCLT/globalmap_gt_cluster.npz')