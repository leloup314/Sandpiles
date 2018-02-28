#! usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from mpl_toolkits.mplot3d import Axes3D

def plot_avalanches2d(avalanches):
    """
    Plots all 2d avalanches in avalanches iteratively. Each plot needs to be closed to show the next one.
    
    :param avalanches: np.array of avalanches
    """
    for i in range(avalanches.shape[0]):
        plt.imshow(avalanches[i], cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.title('Avalanche formation on %i x %i sandbox' % (avalanches.shape[1], avalanches.shape[2]))
        plt.xlabel('x')
        plt.ylabel('y')
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, avalanches.shape[1], 2), minor=True)
        ax.set_yticks(np.arange(-.5, avalanches.shape[2], 2), minor=True)
        ax.grid(which='minor', color='k', ls='--', lw=0.25)
        avalanche_patch = mp.Patch(color='k', label='Avalanche sites')
        plt.legend(handles=[avalanche_patch])
        plt.show()
        
        
def plot_avalanches3d(avalanches):
    """
    Plots all 3d avalanches in avalanches iteratively. Each plot needs to be closed to show the next one.
    
    :param avalanches: np.array of avalanches
    """
    for i in range(avalanches.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        c_a = avalanches[i]
        p = np.column_stack(np.where(c_a))
        x, y, z = p[:, 0], p[:, 1], p[:,2]
        ax.set_xlim(0, c_a.shape[0])
        ax.set_ylim(0, c_a.shape[1])
        ax.set_zlim(0, c_a.shape[2])
        ax.scatter(x,y,z, label='Avalanche sites', marker='s', c='k', edgecolors='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Avalanche formation on %i x %i x %i sandbox' % c_a.shape)
        ax.legend(loc='upper right', bbox_to_anchor=(1,1.1))
        plt.show()



