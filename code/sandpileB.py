import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # Is used in background for 3D plotting
from numba import njit  # use numbas njit for speed-up


# TODO Drop only 1 grain vs drop until slope is < slope_crit ?


def init_sandbox(dim, length, state='empty'):
    """
    Initialises a sandbox lattice (hypercube of dimension dim) on which the simulation is done.

    :param length: int dimension of sandbox
    :param state: str some state of the initial sandbox
    :return: np.array sandbox of dimension dim
    """

    if state == 'empty':
        res = np.zeros(shape=(length,)*dim, dtype=np.uint32)
    else:  # None, ground state
        res = np.zeros(shape=(length,)*dim, dtype=np.uint32)

    return res


@njit
def off_boundary(s, x):
    """
    Checks whether point x is beyond the boundary of the sandbox.

    :param s: np.array
    :param x: int position of grain drop-off
    """

    for i, x_i in enumerate(x):
        if (x_i >= s.shape[i]) or (x_i < 0):
            return True
    return False


@njit
def at_open_edge(s, x, open_bounds):
    """
    Checks whether point x is right at an edge of the sandbox.
    Only consider edges exhibiting open boundary conditions in the query,
    such that 'closed edges' are hidden from open boundary handling.

    :param s: np.array
    :param x: int position of grain drop-off
    :param open_bounds: tuple (of 2 times sandbox's dimension) of booleans specifying
                        open[True]/closed[False] boundary conditions for respective edges
    """

    # Cannot be at the edge AND off boundary
    if off_boundary(s, x):
        return False

    for i, x_i in enumerate(x):
        if (open_bounds[2*i] == True) and (x_i == 0):                   # x_i at i'th open lower edge?
            return True
        if (open_bounds[2*i+1] == True) and (x_i == s.shape[i] - 1):    # x_i at i'th open upper edge?
            return True
    return False


#@njit
def get_neighbours(x):
    """
    Finds all nearest neighbours of x and returns them.

    :param x: int position of grain drop-off
    :return: array of coordinates of neighbours
    """

    nn = []
    x = list(x)

    for i, x_i in enumerate(x):
        for shift in (-1, 1):
            nn.append(tuple(x[0:i] + [x_i + shift] + x[i+1:]))

    return np.array(nn)


#@njit
def get_neighbouringSlopes(s, x, neighbours):
    """
    Returns slopes (pile height differences) to all given neighbours of x with respect to x.
    Each slope value corresponds to a column vector in the neighbours-array (see also get_neighbours(x))

    :param s: np.array sandbox
    :param x: int position of grain drop-off
    :param neighbours: array of coordinates of neighbours
    :return: array of slopes
    """

    # Number of neighbours
    num = neighbours.shape[0]

    # Set all slopes to 0 initially (closed boundary conditions)
    retSlope = np.zeros(shape=num, dtype=np.int32)

    # Determine slopes to each neighbour from pile height differences
    # (if neighbour is off-boundary, just don't overwrite boundary conditions from above)
    for i in range(num):
        if not off_boundary(s, neighbours[i]):
            retSlope[i] = (s[tuple(x)] - s[tuple(neighbours[i])])

    return retSlope


#@njit
def get_unique_rows(array):
    """
    Returns array with removed duplicate rows.

    :param array: 2-dim np.array
    :return: 2-dim np.array with duplicate rows removed
    """

    # Perform lex sort on array
    sorted_idx = np.lexsort(array.T)
    sorted_array = array[sorted_idx]

    # Get unique row mask
    row_mask = np.append([True], np.any(a=np.diff(sorted_array,axis=0), axis=1))

    # Return unique rows
    return sorted_array[row_mask]


#@njit
def do_relaxation(s, x_array, crit_slope, open_bounds, avalanche_stats, recLevel=0):
    """
    Performs the avalanche relaxation mechanism recursively until all slopes are non-critical anymore.

    :param s: np.array sandbox
    :param x_array: int position of grain drop-off or an array of positions for multiple simultaneous relaxations
    :param crit_slope: int critical height of pile of sand grains
    :param open_bounds: boolean array of boundary conditions (open/closed)
    :param avalanche_stats: dict used for gathering avalanche statistics
    :param recLevel: int avalanche's recursion depth
    :return: np.array changed sandbox after relaxation process
    """

    # Dimension of the sandbox
    dim = s.ndim

    # Use current recursion depth as avalanche's time extent
    avalanche_stats["time"] = recLevel

##    if recLevel > 6: print("Recursion level:", recLevel) # DEBUG information

    # Reshape x_array if it is only a single position, such that the loop below can be used in all cases
    if x_array.ndim == 1:
        x_array = x_array.reshape((1,x_array.shape[0]))

    # To emulate simultaneous relaxations, do them successively for each member (position)
    # of x_array using the same sandbox s=const for slope determination.
    # The simultaneous relaxations are meanwhile accumulated in sandbox sPrime
    sPrime = np.copy(s)

    # Note at which positions/iterations relaxation events happen
    relaxEvents = np.array([], dtype=np.uint32)

    # Loop through positions in x_array
    for it in range(x_array.shape[0]):
        x = x_array[it]

        # Dont try to relax if x is off-boundary
        if off_boundary(s, x):
            continue

        # If x is right at an 'open edge', just drop excess grains from sandpile for too large s[x]
        if (s[tuple(x)] >= crit_slope) and at_open_edge(s, x, open_bounds):
            sPrime[tuple(x)] = 0
            relaxEvents = np.append(arr=relaxEvents, values=[it], axis=0)   # Bookkeeping (see below)
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

        # Sort slope values (descending order) and corresponding indices
        sort_idx = np.argsort(crit_slopes)[::-1]
        crit_slopes_sorted = crit_slopes[sort_idx]
        crit_neighbours_sorted = crit_neighbours[sort_idx]

        # Find neighbours with maximum slope
        current_max_slope = crit_slopes_sorted[0]
        max_slopes_idx = np.where(crit_slopes_sorted == current_max_slope)[0]

        # Choose random neighbour out of those with maximum slope
        tRnd = np.random.choice(a=max_slopes_idx, size=1)[0]
        rnd_crit_neighbour = crit_neighbours_sorted[tRnd]

        ##-- Drop grains to the chosen neighbour     --##
        ##-- until slope becomes non-critical again. --##

        toDrop = np.ceil((current_max_slope - crit_slope + 1) / 2.0)

        sPrime[tuple(x)]                  -= toDrop
        sPrime[tuple(rnd_crit_neighbour)] += toDrop

    # If no relaxation actually happened the avalanche stops at this recursion level
    if len(relaxEvents) == 0:
        return s

    # Increase avalanche size about the number of additional relaxation events
    avalanche_stats["size"] += len(relaxEvents)

    # Now after simultaneous relaxations at positions in x_array
    # relax all neighbours of actually relaxed positions in x_array simultaneously
    x_array_neighbours = get_neighbours(x_array[relaxEvents[0]])
    for it in relaxEvents[1:]:  #Skip first event as this is initial content of x_array_neighbours
        x_array_neighbours = np.append(arr=x_array_neighbours, values=get_neighbours(x_array[it]), axis=0)

    # Remove duplicate neighbours from x_array_neighbours
    x_array_neighbours = get_unique_rows(x_array_neighbours)

    # Use sPrime as updated sandbox for next relaxation step
    returnSandbox = do_relaxation(s=sPrime, x_array=x_array_neighbours, crit_slope=crit_slope, open_bounds=open_bounds,
                                  avalanche_stats=avalanche_stats, recLevel=recLevel+1)

    return returnSandbox


#@njit
def add_sand(s, x, crit_slope, open_bounds, avalanche_stats):
    """
    Adds grain of sand at x and initiates relaxation mechanism.

    :param s: np.array sandbox
    :param x: int position of grain drop-off
    :param crit_slope: int critical height of pile of sand grains
    :param open_bounds: boolean array of boundary conditions (open/closed)
    :param avalanche_stats: dict used for gathering avalanche statistics
    :return: np.array changed sandbox after adding sand and performing relaxation
    """

    # If off-boundary, 'drop' the grain from the sandbox and do nothing
    if off_boundary(s, x):
        return s

    # Add grain to sandbox
    s[tuple(x)] += 1

    # Initiate relaxation of the sandpile
    s = do_relaxation(s, x, crit_slope, open_bounds, avalanche_stats)

    # Return fully relaxed sandpile
    return s


#@njit
def add_sand_random(s, crit_slope, open_bounds, avalanche_stats = {"time" : 0, "size" : 0}):
    """
    Adds one grain of sand at a random place in the sandbox.

    :param s: np.array sandbox
    :param crit_slope: int critical height of pile of sand grains
    :param open_bounds: boolean array of boundary conditions (open/closed)
    :param avalanche_stats: dict used for gathering avalanche statistics
    :return: np.array changed sandbox after adding sand and performing relaxation
    """

    # Generate random position
    x_rand = np.zeros(shape=s.ndim, dtype=np.uint32)
    for i in range(s.ndim):
        x_rand[i] = np.random.randint(low=0, high=s.shape[i])

    # Add grain of sand at this position
    s = add_sand(s, x_rand, crit_slope, open_bounds, avalanche_stats)

    # Return relaxed sandpile with one more grain of sand on it
    return s


def get_2d_sandboxSlice(sandbox):
    """
    Returns 2-dim sub-array of sandbox for plotting purposes if dimension is larger than 2.

    :param sandbox: np.array sandbox
    :return: 2-dim sub-array of sandbox
    """

    if sandbox.ndim <= 2:
        return sandbox

    tdSlice = np.copy(sandbox)
    tDim = sandbox.ndim

    # Select sub-array in the middle until slice has 2 dimensions
    while tDim > 2:
        tdSlice = tdSlice[int(np.ceil((sandbox.shape[tDim-1]-1) / 2.0))]
        tDim -= 1

    return tdSlice


def plot3d(sandbox, iterations, crit_slope, open_bounds, pause):
    """
    Plots evolution over time of sandpiles in 3D bar plot. Very slow, only suitable for N <= 20.

    :param sandbox: np.array sandbox to start from
    :param iterations: number of grain drops
    :param crit_slope: critical slope
    :param open_bounds: boolean array of boundary conditions (open/closed)
    :param pause: time delay between grain drops
    """

    # If sandbox dimension is larger 2, choose 2-dim slice in the middle for plotting
    sbSlice = get_2d_sandboxSlice(sandbox)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xedges = np.arange(sbSlice.shape[0])
    yedges = np.arange(sbSlice.shape[1])
    xpos, ypos = np.meshgrid(xedges + 0.25, yedges + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    dx = 0.5*np.ones_like(zpos)
    dy = dx.copy()

    for i in range(iterations):
        sandbox = add_sand_random(s=sandbox, crit_slope=crit_slope, open_bounds=open_bounds)

        # If sandbox dimension is larger 2, choose 2-dim slice in the middle for plotting
        sbSlice = get_2d_sandboxSlice(sandbox)

        dz = sbSlice.flatten()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

        # Pause to allow interactive plotting
        plt.pause(pause)

        if i != iterations-1:
            ax.cla()

    plt.ioff()
    plt.show()

    return sandbox


def plot2d(sandbox, iterations, crit_slope, open_bounds, pause):
    """
    Plots evolution over time of sandpiles in 2D heat map plot.

    :param sandbox: np.array sandbox to start from
    :param iterations: number of grain drops
    :param crit_slope: critical slope
    :param open_bounds: boolean array of boundary conditions (open/closed)
    :param pause: time delay between grain drops
    :return: np.array changed sandbox after iterations
    """

    # Interactive plotting
    plt.ion()

    # If sandbox dimension is larger 2, choose 2-dim slice in the middle for plotting
    sbSlice = get_2d_sandboxSlice(sandbox)

    # Make image with colormap BlueGreenRed
    img = plt.imshow(X=sbSlice, cmap='jet')

    # Add colorbar
    plt.colorbar(img)

    for _ in range(iterations):
        sandbox = add_sand_random(s=sandbox, crit_slope=crit_slope, open_bounds=open_bounds)

        # If sandbox dimension is larger 2, choose 2-dim slice in the middle for plotting
        sbSlice = get_2d_sandboxSlice(sandbox)

        # Update image
        img.set_data(sbSlice)

        # Pause to allow interactive plotting
        plt.pause(pause)

    plt.ioff()
    plt.show()

    return sandbox



def main():

    # Init variables and sandbox
    iterations = 100000
    crit_slope = 5
    dimension = 2
    length = 30
    sandbox = init_sandbox(dim=dimension, length=length, state='empty')

    # Define boundary conditions
    open_boundaries=(True,)*2*dimension         # Open boundary conditions at all lower/upper edges
    #open_boundaries=(False,)*2*dimension        # Closed boundary conditions at all lower/upper edges
    #open_boundaries=(False,False,False,True)    # 2-dim model, one open boundary
    #open_boundaries=(True,False,True,True)      # 2-dim model, one closed boundary
    #open_boundaries=(True,False,False,True)     # 2-dim model, two closed boundaries


    # Create random critical sandpile
    for i in range(iterations):
        sandbox = add_sand_random(s=sandbox, crit_slope=crit_slope, open_bounds=open_boundaries)


    # Study avalanche statistics
    for i in range(1000):
        avalanche_statistics = {"time" : 0, "size" : 0}
        sandbox = add_sand_random(s=sandbox, crit_slope=crit_slope, open_bounds=open_boundaries,
                                  avalanche_stats=avalanche_statistics)
        if avalanche_statistics["time"] > 1:
            print(avalanche_statistics)

    # Plot evolution of critical sandpile
    plot2d(sandbox=sandbox, iterations=50, crit_slope=crit_slope, open_bounds=open_boundaries, pause=0.25)



if __name__ == '__main__':
    main()
