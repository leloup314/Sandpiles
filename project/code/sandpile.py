import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # Is used in background for 3D plotting
from numba import njit  # use numbas njit for speed-up

# Check abs(slope z)
# Make function that returns slope
# Look at slope in between sandpiles. If slope over z_crit, redistribute 1 grain to x,y with "highest" slope
# Distribute randomly if ALL slopes are equal:
# Drop only 1 grain vs drop until slope is < slope_crit

# Markus git : flooklab


def init_sandbox(n, state=None, crit_pile=4):
    """
    Initialises an empty (0s) square NxN sandbox lattice on which the simulation is done.

    :param n: int dimension of sandbox
    :param state: str some state of the initial sandbox
    :param crit_pile: int critical height of pile of sand grains

    """

    if state == 'one':
        res = np.ones(shape=(n, n), dtype=np.uint8)
    elif state == 'crit':
        res = np.full(shape=(n, n), fill_value=crit_pile - 1, dtype=np.uint8)
    elif state == 'over_crit':
        res = np.full(shape=(n, n), fill_value=crit_pile + 1, dtype=np.uint8)
    else:  # None, ground state
        res = np.zeros(shape=(n, n), dtype=np.uint8)

    return res


@njit
def off_boundary(s, x, y):
    """
    Checks whether s[x][y] is beyond the boundary of the sandbox.

    :param s: np.array
    :param x: int x coordinate of grain drop-off
    :param y: int y coordinate of grain drop-off
    """
    dims = (x, y)  # Non-mutable tuple increases speed-up ([500ns] vs (300ns))/loop
    for i, dim in enumerate(dims):
        if dim > s.shape[i] - 1 or dim < 0:
            return True
    return False


@njit
def get_neighbours(x, y):
    """
    Finds all neighbours of [x][y] and returns them
    :param s: np.array
    :param x: int x coordinate of grain drop-off
    :param y: int y coordinate of grain drop-off
    :return: tuples with coordinates of neighbours
    """
    return (x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)  # FIXME: hardcoded neighbours


@njit
def add_sand(s, x, y, crit_pile):
    """
    Adds grain of sand at s[x][y] and checks criticality and boundary conditions.

    :param s: np.array
    :param x: int x coordinate of grain drop-off
    :param y: int y coordinate of grain drop-off
    :param crit_pile: int critical height of pile of sand grains
    """

    # If off-boundary, 'drop' the grain from the sandbox and do nothing
    if not off_boundary(s, x, y):

        # Add grain to sandbox
        s[x][y] += 1

        # Check if sandpile at s[x][y] is at critical level (or above, needed for over critical init)
        if s[x][y] >= crit_pile:

            actual_pile = s[x][y]  # Store actual height of pile to remove (can be > crit_pile for over_crit init)
            s[x][y] = 0  # Remove all grains of sand from [x][y]
            nearest_neighbours = get_neighbours(x, y)  # Get nearest neighbours of [x][y]

            # Redistribute grains in actual_pile by recursively calling add_sand for all neighbours of [x][y]
            while actual_pile:
                x_tmp, y_tmp = nearest_neighbours[actual_pile % len(nearest_neighbours)]
                add_sand(s, x_tmp, y_tmp, crit_pile)
                actual_pile -= 1


@njit
def add_sand_random(s, crit_pile):
    """
    Adds one grain of sand at a random place in the sandbox.

    :param s: np.array
    :param crit_pile: int critical height of pile of sand grains
    """

    x_rand = np.random.randint(low=0, high=s.shape[0])
    y_rand = np.random.randint(low=0, high=s.shape[1])
    add_sand(s, x_rand, y_rand, crit_pile)


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

    for _ in range(iterations):
        add_sand_random(s, crit_pile)
        img.set_data(s)  # update image
        plt.pause(0.00001)  # pause to allow interactive plotting

    plt.ioff()
    plt.show()




def main():

    # Init variables and sandbox
    iterations = 100
    crit_pile = 6
    sandbox = init_sandbox(5, state='ground', crit_pile=crit_pile)

    plot2d(sandbox, iterations, crit_pile)

if __name__ == '__main__':
    main()
