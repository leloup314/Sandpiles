"""Implementation of slightly modified, iterative Bak-Tang-Wiesenfeld approach for cellular automation of sanpile dynamics"""

import time
import numpy as np  # Vectorized arrays
import matplotlib.pyplot as plt  # Plotting

from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # Is used in background for 3D plotting
from scipy.spatial.distance import cdist  # Calculate distances
from scipy.optimize import curve_fit
from numba import njit, jit


def init_sandbox(n, dim, state=None, crit_pile=4):
    """
    Initialises a square NxN sandbox lattice on which the simulation is done.

    :param n: int of lattice sites of sandbox
    :param dim: int of dimension like n^dim
    :param state: str some state of the initial sandbox
    :param crit_pile: int critical height of pile of sand grains

    """

    if state == 'one':
        res = np.ones(shape=(n, ) * dim, dtype=np.int16)
    elif state == 'crit':
        res = np.random.choice([crit_pile-1, crit_pile], shape=(n, ) * dim, dtype=np.int16)
    elif state == 'over_crit':
        res = np.random.choice([crit_pile, crit_pile + 1], shape=(n, ) * dim, dtype=np.int16)
    else:  # None, ground state
        res = np.zeros(shape=(n, ) * dim, dtype=np.int16)

    return res
    
'''
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

    return nn
'''

def neighbours(s, point, distance=1, metric='euclidean'):
    """
    Returns list of tuples of nearest neighbours of point in sandbox s with distance.
    Returns only neighbours who are actually on the lattice of s.
    
    :param s: np.array of sandbox
    :param point: tuple of point
    :param distance: scalar of distance the neighbours must have to point
    :param metric: str of any accepted metric by cdist
    """
    
    # Make array of coordinates of all lattice points
    coordinates = np.column_stack(np.indices(s.shape).reshape(len(s.shape), np.product(s.shape)))
    
    # Return list of coordinate tuples with distance to point
    return map(tuple, coordinates[(cdist(np.array([point]), coordinates, metric=metric) == distance)[0]])
            

@njit
def relax_pile(s, critical_point, neighbours, amount):
    """
    Relaxes a critical point of the sandbox s by substracting the amount of sand
    an redistributing it to the neighbours. If there are less neighbouring points
    than the geometry of the sandbox implies, the amount is reduced by
    the amount of grains falling off the lattice.
    
    :param s: np.array of sandbox
    :param critical_point: tuple of critical point
    :param neighbours: array of tuples of neighbouring points of critical_point
    :param amount: int amount of sand critical_point is reduced by
    """
    
    # Relax critical point
    s[critical_point] -= amount
    
    # Neighbours and index
    n_neighbours = len(neighbours)
    i = 0
    
    # Throw as many grains away as neighbours are missing; Equivalent to letting sand fall off lattice
    amount = int(amount * n_neighbours / (2.0*len(s.shape)))
    
    # Redistribute among neighbours
    while amount:
        n = neighbours[i % n_neighbours]
        s[n] += 1
        amount -= 1
        i += 1
        

def relax_sandbox(s, critical_points, amount, neighbour_LUD):
    """
    Relaxes s by looping over all critical_points and calling relax_pile on them.
    Using a look-up dictionary with points as keys and their neighbours as values in
    order to avoid calculating the neighbours of a given point twice.
    
    :param s: np.array of sandbox
    :param critical_points: array of tuples of critical points
    :param amount: int amount of sand each critical_point is reduced by
    :param neighbour_LUD: dict that is/will be filled with point, neighbours as key,value
    """
    for p in critical_points:
        
        if p in neighbour_LUD:
            n = neighbour_LUD[p]
        else:
            n = neighbours(s, p)
            neighbour_LUD[p] = n

        relax_pile(s, p, n, amount)


def add_sand(s, point=None, amount=1):
    """
    Add a single grain of sand at point. If point is None, add random.
    
    :param s: np.array of sandbox
    :param point: tuple of point or None
    :param amount: int of sand grains which are added to point
    """
    if point is None:
        # Make random point
        point = np.random.randint(low=0, high=np.amin(s.shape), size=s.ndim)
    
    point = tuple(point)
    
    s[point] += amount


def fill_sandbox(s, crit_pile, level=0.8):
    """
    Fills s with sand until level per cent of the sandbox are critical.
    
    :param s: np.array of sandbox
    :param crit_pile: int critical height of sand piles
    :param level: float percentage of lattice sites which must be critical
    """

    while (np.sum(s, dtype=np.float) / np.sum(np.full(shape=s.shape, fill_value=crit_pile), dtype=np.float)) < level:
        m = s < crit_pile
        s[m] = np.random.choice(np.arange(crit_pile+1), size=s[m].shape)


def do_simulation(s, crit_pile, total_drops, results, plot):
    
    # Capture start time of main loop
    start = time.time()
    
    # Make look-up dict of points as keys and their neighbours as values
    # Immense speed-up: 7.61 seconds (w/ LUD) vs. 125.97 seconds (w/o LUD) for a 50x50 sandbox with _SAND_DROPS=10000 
    neighbour_LUD = {}
    
    
    if plot:
        plt.ion()
        img = plt.imshow(s, cmap='jet', vmin=0, vmax=crit_pile)  # make image with colormap BlueGreenRed
        plt.colorbar(img)  # add colorbar
    
    # Capture start time of main loop
    start = time.time()
    
    for drop in xrange(total_drops):
        
        # Add sand at random position
        add_sand(s)
    
        i = 0
        while True in (s >= crit_pile):
            
            # Get critical points
            critical_points = map(tuple, np.column_stack(np.where(s >= crit_pile)))
            
            # Relax the sandbox
            relax_sandbox(s, critical_points, crit_pile, neighbour_LUD)
            
            # Increment counter
            i+=1
            
            if i % 1 == 0 and plot:
                img.set_data(s)
                plt.pause(.0002)
                
        results['iterations'][drop] = i
        
    # Capture time of main loop
    sim_time = time.time()-start
    
    results['time'] = sim_time
    
    if plot:
        plt.ioff()
        plt.show()
        
    return results
    

def plot_sandbox(s, total_drops, point=None, output_pdf=None):
    
    title = '%s sandbox with %i ' % (str(s.shape), total_drops)
    title += 'randomly dropped grains' if point is None else 'grains dropped at %s' % str(point)
    plt.title(title)
    img = plt.imshow(s, cmap='jet', vmin=0, vmax=np.amax(s)+1)  # make image with colormap BlueGreenRed
    plt.colorbar(img)  # add colorbar
    plt.show()
    
    if output_pdf is not None:
        with PdfPages(output_pdf, keep_empty=False) as out:
            out.savefig(plt.figure())
            
        
def plot_hist(data):
    
    data_unique, data_count = np.unique(data, return_counts=True)
    counts, bin_edges, _ = plt.hist(data, bins=len(data_unique))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, counts, 'rx')
    plt.loglog()
    plt.grid()
    plt.show()
    
    

    
def main():
    
    ### Initialization of several static variables 
    
    # Number of dimensions
    _DIM = 2
    
    # Length of sandbox
    _L = 100
    
    # Set critical sand pile height
    _CRIT_H = 4
    
    # Number of total sand drops
    _SAND_DROPS = 100000
    
    # Set plotting
    _PLOT = False
    
    ### Init sandbox and several bookkeeping/speed-up dicts
    
    # Init sandbox
    s = init_sandbox(_L, _DIM)
    
    # Fill sandbox until critical
    #fill_sandbox(s, _CRIT_H)
    
    # Init result dict
    results = {'iterations': np.empty(shape=_SAND_DROPS), 'time': None}
    
    # Do actual simulation
    results = do_simulation(s, _CRIT_H, _SAND_DROPS, results, _PLOT)
    
    print 'Needed %f for %i iterations' % (results['time'], _SAND_DROPS)
    
    plot_sandbox(s, _SAND_DROPS, point=None)
    
    plot_hist(results['iterations'])
    
    with open('timing.log', 'a') as f:
        t = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
        msg = '%s, %i drops, %f seconds\n' % (str(s.shape), _SAND_DROPS, results['time'])
        t += ':\t'
        f.write(t)
        f.write(msg)
    
    # TODO: actual book-keeping to file
    
if __name__ == "__main__":
    main()
    
    
    
