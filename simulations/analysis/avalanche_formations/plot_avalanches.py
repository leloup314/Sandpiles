#! usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

def plot_avalanches(avalanches):
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

