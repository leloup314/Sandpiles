import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # Is used in background for 3D plotting
from numba import njit  # use numbas njit for speed-up


# Drop only 1 grain vs drop until slope is < slope_crit


def init_sandbox(dim, length, state=None, crit_slope=4):
    """
    Initialises an empty (0s) square NxN sandbox lattice on which the simulation is done.

    :param length: int dimension of sandbox
    :param state: str some state of the initial sandbox
    :param crit_slope: int critical local slope of sandpile

    """

    #TODO change 'crit' and 'over_crit'; does not make sense here anymore
    if state == 'one':
        res = np.ones(shape=np.full(dim, fill_value=length), dtype=np.uint8)
    elif state == 'crit':
        res = np.full(shape=np.full(dim, fill_value=length), fill_value=crit_slope - 1, dtype=np.uint8)
    elif state == 'over_crit':
        res = np.full(shape=np.full(dim, fill_value=length), fill_value=crit_slope + 1, dtype=np.uint8)
    else:  # None, ground state
        res = np.zeros(shape=np.full(dim, fill_value=length), dtype=np.uint8)

    return res


@njit
def off_boundary(s, x):
    """
    Checks whether point x is beyond the boundary of the sandbox.

    :param s: np.array
    :param x: int position of grain drop-off
    """

    #           Any x_i larger than sandbox size shape_i         or     any x_i smaller zero?
    return ( (len(np.where( (x-np.array(s.shape)) >= 0)[0]) > 0) or (len(np.where(x < 0)[0]) > 0) )


#@njit
def get_neighbours(x):
    """
    Finds all nearest neighbours of x and returns them.
    :param x: int position of grain drop-off
    :return: array of coordinates of neighbours
    """

    # Dimension of the sandbox
    dim = x.shape[0]

    # Return nearest neighbours as np.array like
    # [[x1-1, x2, ...][x1, x2-1, ...] ... [x1+1, x2, ...][x1, x2+1, ...] ...]
    return (    np.array( ((np.ones(shape=(dim,1), dtype=np.uint8)*x - np.identity(dim)),
                           (np.ones(shape=(dim,1), dtype=np.uint8)*x + np.identity(dim))), dtype=np.int8
                        ).reshape((2*dim,dim))
           )


#@njit
def get_neighbouringSlopes(s, x, neighbours):
    """
    Returns slopes (pile height differences) to all nearest neighbours of x with respect to x.
    Each slope value corresponds to a column vector in the neighbours-array (see also get_neighbours(x))
    :param x: int position of grain drop-off
    :return: array of slopes
    """

    # Dimension of the sandbox
    dim = x.shape[0]

    # Set all slopes to 0 initially (closed boundary conditions)
    retSlope = np.zeros(shape=2*dim, dtype=np.int8)
    #TODO other boundary conditions?

    # Determine slopes to each neighbour from pile height differences
    # (if neighbour is off-boundary, just don't overwrite boundary conditions from above)
    for i in range(2*dim):
        if not off_boundary(s, neighbours[i]):
            retSlope[i] = (s[tuple(x)] - s[tuple(neighbours[i])])

    return retSlope


#@njit
def do_relaxation(s, x_array, crit_slope, recLevel=0):
    """
    Performs the avalanche relaxation mechanism recursively until all slopes are non-critical anymore.
    :param s: np.array
    :param x_array: int position of grain drop-off or an array of positions for multiple simultaneous relaxations
    :param crit_slope: int critical height of pile of sand grains
    """

##    print("Recursion level:", recLevel) # DEBUG information

    # Reshape x_array if it is only a single position, such that the loop below can be used in all cases
    if x_array.ndim == 1:
        x_array = x_array.reshape((1,x_array.shape[0]))

    # To emulate simultaneous relaxations, do them successively for each member (position)
    # of x_array using the same sandbox s=const for slope determination.
    # The simultaneous relaxations are meanwhile accumulated in sandbox sPrime
    sPrime = np.copy(s)

    # Note at which positions/iterations relaxation events happen
    relaxEvents = np.array([], dtype=np.uint8)

    # Loop through positions in x_array
    for it in range(x_array.shape[0]):
        x = x_array[it]

        # Dont try to relax if x is off-boundary
        if off_boundary(s, x):
            continue

        ###-- Choose random nearest neighbour with maximum (and critical) slope. --###
        ###-- If no slope is critical, do nothing further.                       --###

        # Get all nearest neighbours around position x and determine slopes between x and neighbours
        neighbours = get_neighbours(x)
        slopes = get_neighbouringSlopes(s, x, neighbours)

        # Find neightbours with at least critical slope and list their corresponding slopes
        crit_slopes_idx = np.where(slopes >= crit_slope)[0]
        crit_slopes = slopes[crit_slopes_idx]
        crit_neighbours = neighbours[crit_slopes_idx]

        # Continue loop if no slope is critical at position x
        if len(crit_slopes_idx) == 0:
            continue
        else:
            # Bookkeeping: actual relaxation event will happen at this recursion level
            relaxEvents = np.append(arr=relaxEvents, values=[it], axis=0)

        # Sort slope heights and corresponding indices
        sort_idx = np.argsort(crit_slopes)
        crit_slopes_idx_sorted = crit_slopes_idx[sort_idx]
        crit_slopes_sorted = crit_slopes[sort_idx]
        crit_neighbours_sorted = crit_neighbours[sort_idx]

        # Reverse sorting
        crit_slopes_idx_sorted = np.flip(m=crit_slopes_idx_sorted, axis=0)
        crit_slopes_sorted = np.flip(m=crit_slopes_sorted, axis=0)
        crit_neighbours_sorted = np.flip(m=crit_neighbours_sorted, axis=0)

        # Find neighbours with maximum slope
        current_max_slope = crit_slopes_sorted[0]
        max_slopes_idx = np.where(crit_slopes_sorted == current_max_slope)[0]

        # Choose random neighbour out of those with maximum slope
        tRnd = np.random.choice(a=max_slopes_idx, size=1)[0]
        rnd_crit_neighbour = crit_neighbours_sorted[tRnd]

        ##- Drop grains to the chosen neighbour     -##
        ##- until slope becomes non-critical again. -##
        toDrop = np.ceil((current_max_slope - crit_slope + 1) / 2.0)

        sPrime[tuple(x)]                  -= toDrop
        sPrime[tuple(rnd_crit_neighbour)] += toDrop

    # If no relaxation actually happened the avalanche stops at this recursion level
    if len(relaxEvents) == 0:
        return s

    # Now after simultaneous relaxations at positions in x_array
    # relax all neighbours of actually relaxed positions in x_array simultaneously
    x_array_neighbours = get_neighbours(x_array[relaxEvents[0]])
    for it in relaxEvents[1:]:  #Skip first event as this is initial content of x_array_neighbours
        x_array_neighbours = np.append(arr=x_array_neighbours, values=get_neighbours(x_array[it]), axis=0)
        #TODO do not add duplicate neighbours here

    # Use sPrime as updated sandbox for next relaxation step
    returnSandbox = do_relaxation(s=sPrime, x_array=x_array_neighbours, crit_slope=crit_slope, recLevel=recLevel+1)

    return returnSandbox


#@njit
def add_sand(s, x, crit_slope):
    """
    Adds grain of sand at x and initiates relaxation mechanism.

    :param s: np.array
    :param x: int position of grain drop-off
    :param crit_slope: int critical height of pile of sand grains
    """

    # If off-boundary, 'drop' the grain from the sandbox and do nothing
    if off_boundary(s, x):
        return s

    # Add grain to sandbox
    s[tuple(x)] += 1

    # Initiate relaxation of the sandpile
    s = do_relaxation(s, x, crit_slope)

    # Return fully relaxed sandpile
    return s


#@njit
def add_sand_random(s, crit_slope):
    """
    Adds one grain of sand at a random place in the sandbox.

    :param s: np.array
    :param crit_slope: int critical height of pile of sand grains
    """

    # Generate random position
    x_rand = np.zeros(shape=(s.ndim), dtype=np.uint8)
    for i in range(s.ndim):
        x_rand[i] = np.random.randint(low=0, high=s.shape[i])

    # Add grain of sand at this position
    s = add_sand(s, x_rand, crit_slope)

    # Return relaxed sandpile with one more grain of sand on it
    return s


#def plot3d(s, iterations, crit_slope):
#    """
#    Plots evolution over time of sandpiles in 3D bar plot. Very slow, only suitable for N <= 20
#    :param s: np.array
#    :param iterations: number of grain drops
#    :param crit_slope: critical pile heigh
#    """
#
#    plt.ion()
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    xedges = np.arange(s.shape[0])
#    yedges = np.arange(s.shape[1])
#    xpos, ypos = np.meshgrid(xedges + 0.25, yedges + 0.25)
#    xpos = xpos.flatten('F')
#    ypos = ypos.flatten('F')
#    zpos = np.zeros_like(xpos)
#    dx=0.5*np.ones_like(zpos)
#    dy=dx.copy()
#
#    for i in range(iterations):
#        add_sand_random(s, crit_slope=crit_slope)
#        dz = s.flatten()
#        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
#        ax.set_zlim3d(0, crit_slope-1)
#        plt.pause(0.10001)
#        if i != iterations-1:
#            ax.cla()
#
#    plt.ioff()
#    plt.show()
#
#
#def plot2d(s, iterations, crit_slope):
#    """
#    Plots evolution over time of sandpiles in 2D heat map plot.
#    :param s: np.array
#    :param iterations: number of grain drops
#    :param crit_slope: critical pile heigh
#    """
#
#    plt.ion()  # interactive plotting
#    img = plt.imshow(s, cmap='jet', vmin=0, vmax=crit_slope)  # make image with colormap BlueGreenRed
#    plt.colorbar(img)  # add colorbar
#
#    for _ in range(iterations):
#        add_sand_random(s, crit_slope)
#        img.set_data(s)  # update image
#        plt.pause(0.00001)  # pause to allow interactive plotting
#
#    plt.ioff()
#    plt.show()




def main():

    # Init variables and sandbox
    iterations = 10000
    crit_slope = 5
    sandbox = init_sandbox(dim=2, length=10, state='one', crit_slope=crit_slope)

    for _ in range(iterations):
        sandbox = add_sand_random(sandbox, crit_slope)

    print(sandbox)

#    plot2d(sandbox, iterations, crit_slope)

if __name__ == '__main__':
    main()
