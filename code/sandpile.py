"""Implementation of the Bak-Tang-Wiesenfeld approach for cellular automation of sanpile dynamics"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from mpl_toolkits.mplot3d import Axes3D  # Is used in background for 3D plotting
from numba import jit, njit, types  # use numbas njit for speed-up

sys.setrecursionlimit(10000)  # increase recursion depth to allow recursively add_sand for large lattice sites n


def init_sandbox(n, dim, state=None, crit_pile=4):
    """
    Initialises an empty (0s) square NxN sandbox lattice on which the simulation is done.

    :param n: int of lattice sites of sandbox
    :param dim: int of dimension like n^dim
    :param state: str some state of the initial sandbox
    :param crit_pile: int critical height of pile of sand grains

    """

    if state == 'one':
        res = np.ones(shape=(n, ) * dim, dtype=np.int16)
    elif state == 'crit':
        res = np.full(shape=(n, ) * dim, fill_value=crit_pile - 2, dtype=np.int16)
    elif state == 'over_crit':
        res = np.full(shape=(n, ) * dim, fill_value=crit_pile * 2, dtype=np.int16)
    else:  # None, ground state
        res = np.zeros(shape=(n, ) * dim, dtype=np.int16)

    return res


@njit
def off_boundary(s, point):
    """
    Checks whether s[x][y] is beyond the boundary of the sandbox.

    :param s: np.array
    :param point: tuple of coordinates of grain drop-off
    """

    for i, dim in enumerate(point):
        if dim > s.shape[i] - 1 or dim < 0:
            return True
    return False


#@njit
def get_neighbours(point):
    """
    Finds all nearest neighbours of point with Euclidean distance == 1 and returns them
    :param point: tuple of coordinates for which neighbours are returned
    :return: tuples with coordinates of neighbours
    """

    nn = []
    point = list(point)

    for i, coordinate in enumerate(point):
        for shift in (-1, 1):
            nn.append(tuple(point[0: i] + [coordinate + shift] + point[i+1:]))

    # For @njit use input np.array for neighbours
    # for i in range(len(point)*2):
        # for j, coordinate in enumerate(point):
                # neighbour_array[i, j] = coordinate if i % len(point) == j else coordinate + 1 if i/len(point) < j else coordinate - 1

    return nn


#@njit
def add_sand(s, point, crit_pile):
    """
    Adds grain of sand at s[x][y] and checks criticality and boundary conditions.

    :param s: np.array
    :param point: tuple of coordinates of grain drop-off
    :param crit_pile: int critical height of pile of sand grains
    """

    # If off-boundary, 'drop' the grain from the sandbox and do nothing
    if not off_boundary(s, point):

        # Add grain to sandbox; use numpys multi-indexing with tuples e.g. s[1][1] == s[(1, 1)] == s[1, 1]
        s[point] += 1

        # Check if sandpile at s[x][y] is at critical level (or above, needed for over critical init)
        if s[point] >= crit_pile:

            nearest_neighbours = get_neighbours(point)  # Get nearest neighbours of point
            s[point] -= len(nearest_neighbours)   # Remove as many grains as neighbours and redistribute

            # Redistribute grains by recursively calling add_sand for all neighbours of point
            for neighbour in nearest_neighbours:
                add_sand(s, neighbour, crit_pile)


def add_sand_random(s, crit_pile):
    """
    Adds one grain of sand at a random place in the sandbox.

    :param s: np.array
    :param crit_pile: int critical height of pile of sand grains
    """

    # Make random point
    point = []
    for dim in range(len(s.shape)):
        point.append(np.random.randint(low=0, high=s.shape[dim]))
    point = tuple(point)

    add_sand(s, point, crit_pile)


def plot3d(s, iterations, crit_pile):
    """
    Plots evolution over time of sandpiles in 3D bar plot. Very slow, only suitable for N <= 20
    :param s: np.array
    :param iterations: number of grain drops
    :param crit_pile: critical pile heigh
    """

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xedges = np.arange(s.shape[0])
    yedges = np.arange(s.shape[1])
    xpos, ypos = np.meshgrid(xedges + 0.25, yedges + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    dx=0.5*np.ones_like(zpos)
    dy=dx.copy()

    for i in range(iterations):
        add_sand_random(s, crit_pile=crit_pile)
        dz = s.flatten()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
        ax.set_zlim3d(0, crit_pile-1)
        plt.pause(0.10001)
        if i != iterations-1:
            ax.cla()

    plt.ioff()
    plt.show()


def plot2d(s, iterations, crit_pile):
    """
    Plots evolution over time of sandpiles in 2D heat map plot.
    :param s: np.array
    :param iterations: number of grain drops
    :param crit_pile: critical pile heigh
    """

    plt.ion()  # interactive plotting
    img = plt.imshow(s, cmap='jet', vmin=0, vmax=crit_pile)  # make image with colormap BlueGreenRed
    plt.colorbar(img)  # add colorbar

    for i in range(iterations):
        add_sand_random(s, crit_pile)
        img.set_data(s)
        plt.pause(.000002)

    plt.ioff()
    plt.show()


def main():

    # Init variables and sandbox
    iterations = 100
    crit_pile = 6
    dim = 2
    n = 5
    sandbox = init_sandbox(n, dim=dim, state='ground', crit_pile=crit_pile)
    # neighbour_array = np.zeros(shape=(dim*2, dim), dtype=np.int16)

    plot3d(sandbox, iterations, crit_pile)

if __name__ == '__main__':
    main()
