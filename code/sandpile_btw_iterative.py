"""Implementation of slightly modified, iterative Bak-Tang-Wiesenfeld approach of cellular automata for sandpile dynamics"""

import os  # File saving etc.
import time  # Timing
import logging  # User feedback
import numpy as np  # Arrays with fast, vectorized operations and flexible structure
import matplotlib.pyplot as plt  # Plotting
import pyqtgraph as pg  # Plotting

from PyQt5 import QtWidgets, QtCore  # Plotting
from matplotlib.backends.backend_pdf import PdfPages  # Plotting
from scipy.spatial.distance import cdist, pdist  # Calculate distances
from numba import njit  # Speed-up
from collections import Iterable  # Type checking

logging.basicConfig(level=logging.INFO)


### Simulation functions ###


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
    coordinates = np.column_stack(np.indices(s.shape).reshape(s.ndim, np.product(s.shape)))
    
    # Return list of coordinate tuples with distance to point
    return map(tuple, coordinates[(cdist(np.array([point]), coordinates, metric=metric) == distance)[0]])
            

@njit
def relax_pile(s, critical_point, neighbours, amount, result):
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
    
    # Get number of neighbours
    n_neighbours = len(neighbours)
    
    # Throw as many grains away as neighbours are missing; Equivalent to letting sand fall off lattice
    amount = int(amount * n_neighbours / (2.0 * s.ndim))
    
    # Redistribute amount among neighbours
    i = 0
    while amount:
        n = neighbours[i % n_neighbours]
        s[n] += 1
        amount -= 1
        i += 1
        
    # Increase total amount of dropped grains by i
    result['total_drops'] += i
        

def relax_sandbox(s, critical_points, amount, neighbour_LUD, result, avalanche):
    """
    Relaxes s by looping over all critical_points and calling relax_pile on them.
    Using a look-up dictionary with points as keys and their neighbours as values in
    order to avoid calculating the neighbours of a given point twice.
    
    :param s: np.array of sandbox
    :param critical_points: list of tuples of critical points
    :param amount: int amount of sand each critical_point is reduced by
    :param neighbour_LUD: dict that is/will be filled with point, neighbours as key,values
    :param result: np.array in which the results are stored
    :param avalanche: np.array of bools like s; stores which sites took part in avalanche
    """
    
    # Relax all critical sites
    for p in critical_points:
        
        # Get neighbours of current critical point; either from look-up or function call
        if p in neighbour_LUD:
            n = neighbour_LUD[p]
        else:
            n = neighbours(s, p)
            neighbour_LUD[p] = n  # Write to look-up dict

        # Relax current critical point
        relax_pile(s, p, n, amount, result)
                
        # Critical point and neighbours took part in the avalanche; set to 1
        avalanche[p] = 1
        for nn in n:
            avalanche[nn] = 1        


def add_sand(s, point=None, amount=1):
    """
    Add a single grain of sand at point. If point is None, add randomly.
    
    :param s: np.array of sandbox
    :param point: tuple of point or None
    :param amount: int of sand grains which are added to point
    """
    
    # Make random point
    if point is None:
        point = np.random.randint(low=0, high=np.amin(s.shape), size=s.ndim)
    
    # Add to point
    s[tuple(point)] += amount
        

def do_simulation(s, crit_pile, total_drops, point, result_array, plot_simulation=False, avalanche=None):
    """
    Does entire simulation of sandbox evolution by dropping total_drops of grains on point in s.
    Drops one grain of sand at a time on point. Checks after each drop whether some sites are at
    or above crit_pile. As long as that's the case, relaxes these sites simultaniously. Then moves
    on to next drop.
    
    :param s: np.array of sandbox
    :param crit_pile: int of critical pile height
    :param total_drops: int of total amount of grains that are dropped (iterations)
    :param point: tuple of point on which the sand is dropped or None; if None, add randomly
    :param result_array: np.array in which the results of all iterations are stored
    :param plot_simulation: False or pg.ImageItem; If ImageItem, the entire evolution of the sandbox will be plot
    :param avalanche: np.array of np.arrays of bools like s or None; if None, dont store avalanche configurations, else store which sites took part in avalanche for each iteration
    """
    
    # Make look-up dict of points as keys and their neighbours as values; speed-up by a factor of 10
    neighbour_LUD = {}
    
    # Make temporary array to store avalanche configuration in order to calculate linear size
    tmp_avalanche = np.zeros_like(s, dtype=np.bool)
    
    # Timing estimate
    estimate_time = time.time()

    # Drop sand iteratively
    for drop in xrange(total_drops):
        
        # Feedback
        if drop % (5e-3 * total_drops) == 0 and drop > 0:
            # Timing estimate
            avg_time = (time.time() - estimate_time) / drop  # Average time taken for the last 0.5% of total_drops
            est_hours = avg_time * (total_drops-drop)/60**2
            est_mins = (est_hours % 1) * 60
            est_secs = (est_mins % 1) * 60
            msg = 'At drop %i of %i total drops (%.1f %s). Estimated time left: %i h %i m %i s' % (drop, total_drops, 100 * float(drop)/total_drops, "%", int(est_hours), int(est_mins), int(est_secs))
            logging.info(msg)
        
        # Extract result array for current iteration
        current_result = result_array[drop]
        
        # Extract array for recording avalanches if avalanche array is given; else only store current configuration and reset after
        current_avalanche = tmp_avalanche if avalanche is None else avalanche[drop]
        
        # Add one grain of sand
        add_sand(s, point)
        
        # Initialise counter and start relaxations
        relaxations = 0
        while True in (s >= crit_pile):
            
            # Get critical points
            critical_points = map(tuple, np.column_stack(np.where(s >= crit_pile)))
            
            # Relax the sandbox simultaneously
            relax_sandbox(s, critical_points, crit_pile, neighbour_LUD, current_result, current_avalanche)
            
            # Increment relaxation counter
            relaxations += 1
            
            # Fast plotting
            if plot_simulation:
                plot_simulation.setImage(s, levels=(0, crit_pile), autoHistogramRange=False)
                pg.QtGui.QApplication.processEvents()
        
        ### RESULTS ###        
                
        # Add initially dropped grain to total drops
        current_result['total_drops'] += 1
        
        # Store amount of relaxations within this iteration; equivalent to avalanche duration
        current_result['relaxations'] = relaxations
        
        # Get number of sites participating in current avalanche
        current_result['area'] = np.count_nonzero(current_avalanche)
        
        # Get the linear size of the current avalanche if there were relaxations
        if relaxations != 0:
            
            # Get coordinates of avalanche sites
            coords = np.column_stack(np.where(current_avalanche))
            
            # Get maximum distance between them
            try:
                current_result['lin_size'] = np.amax(pdist(coords))  # This gets slow (> 10 ms) for large (> 50 x 50) sandboxes but is still fastest choice
            
            # Memory error for large amount of avalanche sites
            except MemoryError:
                logging.warning('Memory error due to large avalanche for drop %i. No "lin_size" calculated.' % drop)
                pass  # Do nothing
                
        # Reset avalanche configuration for use in next iteration
        tmp_avalanche[:] = 0
        
        # Fast plotting
        if plot_simulation:
            plot_simulation.setImage(s, levels=(0, crit_pile), autoHistogramRange=False)
            pg.QtGui.QApplication.processEvents()
            
            
def fill_sandbox(s, crit_pile, level=0.8):
    """
    Fills s with sand until a 'level' fraction of sites of the sandbox are critical.
    
    :param s: np.array of sandbox
    :param crit_pile: int critical height of sand piles
    :param level: float percentage of lattice sites which must be critical
    """

    while (np.sum(s, dtype=np.float) / np.sum(np.full(shape=s.shape, fill_value=crit_pile), dtype=np.float)) < level:
        m = s < crit_pile
        s[m] = np.random.choice(np.arange(crit_pile+1), size=s[m].shape)


### Plotting functions and classes ###

    
def plot_sandbox(s, total_drops, point=None, output_pdf=None):
    """
    Plots the configuration of s in a 2D heatmap
    
    :param s: np.array of sandbox
    :param total_drops: int of total amount of grains that were dropped
    :param point: tuple of point on which the sand was dropped or None; if None, drops were random
    :param output_pdf: str of output_pdf file or None
    """
    
    title = '%s sandbox with %i ' % (str(s.shape), total_drops)
    title += 'randomly dropped grains' if point is None else 'grains dropped at %s' % str(point)
    plt.title(title)
    img = plt.imshow(s, cmap='jet', vmin=0, vmax=np.amax(s)+1)  # make image with colormap BlueGreenRed
    plt.colorbar(img)  # add colorbar
    plt.show()
    
    if output_pdf is not None:
        with PdfPages(output_pdf, keep_empty=False) as out:
            out.savefig(plt.figure())
            

def plot_hist(data, title=None):
    """
    Histogramms data and bin centers
    
    :param data: np.array of data to histogram
    :param title: str of title
    """
    
    if title is not None:
        plt.title(title)
    data_unique, data_count = np.unique(data, return_counts=True)
    counts, bin_edges, _ = plt.hist(data)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, counts,'ro', label='Bin centers')
    plt.loglog()
    plt.show()
    
    
class SimulationPlotter(pg.ImageView):
    """
    Subclass of pyqtgraph.ImageView to plot evolution of sandpiles in sandbox
    """
    
    def __init__(self, title=None, parent=None, **kwargs):
        super(SimulationPlotter, self).__init__(parent=parent, **kwargs)
        
        # Set window title
        self.setWindowTitle(title)
        
        # Make color map for image
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color = np.array([[0, 0, 128, 255], [0, 128, 0, 255], [255, 255, 0, 255], [255, 140, 0, 255], [255, 0, 0, 255]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.setColorMap(cmap)
        
        # Show window
        self.show()
    
    
### Saving results functions ###
    
            
def save_array(array, out_file):
    """
    Function to save a numpy array. Avoids overwriting existing arrays.
    
    :param array: np.array to save
    :param out_file: str of path to output file
    """
    
    # Check whether out_file is already in location
    if os.path.isfile(out_file):
        i = 0
        a, b = out_file.split('.')
        while os.path.isfile(a+str(i)+b):
            i += 1
        
        # Set new path    
        out_file = a + '_%i.' % i + b
    
    # Save array to out_file
    np.save(out_file, array)
    
    
def save_simulation(s, sim, total_drops=None, point=None, out_file=None):
    """
    Function to save result array of simulation.
    
    :param s: np.array of sandbox of simulation
    :param sim: np.array of results of simulation
    :param total_drops: int of total dropped grains of sand
    :param point: tuple of point on which was dropped; if None, random
    :param out_file: str of path to output file
    """
    
    # Find output path if not given
    if out_file is None:
        
        # Path where simulations are stored
        sim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../simulations/statistics')
        
        # Total drops
        td = str(total_drops) if total_drops is not None else '?'
        
        # Check whether random drops or on point
        if point is None:
            sm = '%s_simulation_%s_drops_random.npy' % ('x'.join([str(dim) for dim in s.shape]), td)
        else:
            sm = '%s_simulation_%s_drops_at_%s.npy' % ('x'.join([str(dim) for dim in s.shape]), td, '_'.join(str(c) for c in point))
        
        # Set new path
        out_file = os.path.join(sim_path, sm)
    
    # Save array to out_file
    save_array(sim, out_file)


def save_sandbox(s, total_drops=None, point=None, out_file=None):
    """
    Function to save critical sandbox after simulation.
    
    :param s: np.array of sandbox of simulation
    :param total_drops: int of total dropped grains of sand
    :param point: tuple of point on which was dropped; if None, random
    :param out_file: str of path to output file
    """
    
    # Find output path if not given
    if out_file is None:
        
        # Path where sandboxes are stored
        sandbox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../simulations/sandboxes')
        
        # Total drops
        td = str(total_drops) if total_drops is not None else '?'
        
        # Check whether random drops or on point
        if point is None:
            sb = '%s_sandbox_%s_drops_random.npy' % ('x'.join([str(dim) for dim in s.shape]), td)
        else:
            sb = '%s_sandbox_%s_drops_at_%s.npy' % ('x'.join([str(dim) for dim in s.shape]), td, '_'.join(str(c) for c in point))
        
        # Set new path
        out_file = os.path.join(sandbox_path, sb)
    
    # Save array to out_file
    save_array(s, out_file)


### Main ###


def main():
    
    ### Initialization simulation variables ###
    
    # Length of sandbox; can be iterable for several simulations
    _LEN = (20, 50, 70, 100)
    
    # Dimensions; can be iterable for several simulations
    _DIM = (1, 2, 3)
    
    # Set critical sand pile height; usually equal to number of neighbours
    _CRIT_H = tuple(2 * _D for _D in _DIM) if isinstance(_DIM, Iterable) else 2 * _DIM
    
    # Number of total sand drops
    _SAND_DROPS = 1000000
    
    # Point to drop to in sandbox;if None, drop randomly
    _POINT = None
    
    # Whether to plot results
    _PLOT_RES = True
    
    # Whether to plot the evolution of the sandbox
    _PLOT_SIM = False
    
    # Save results of simulation after simulation is done
    _SAVE_SIMULATION = False
    
    # Save sandbox after simulation is done
    _SAVE_SANDBOX = False
    
    # Check for multiple lengths and dimensions
    _LEN = _LEN if isinstance(_LEN, Iterable) else [_LEN]
    _DIM = _DIM if isinstance(_DIM, Iterable) else [_DIM]
    _CRIT_H = _CRIT_H if isinstance(_CRIT_H, Iterable) else [_CRIT_H]
    
    # Do simulation for multiple sandbox lengths and dimensions in loops
    for i, _D in enumerate(_DIM):    
        for _L in _LEN:
            
            # Init sandbox
            s = init_sandbox(_L, _D)
            
            # Fill sandbox until critical
            #fill_sandbox(s, _CRIT_H[i], level=0.75)
        
            # Make structured np.array to store results in
            result_array = np.array(np.zeros(shape=_SAND_DROPS),
                                    dtype=[('relaxations', 'i4'), ('area', 'i4'),
                                           ('total_drops', 'i4'), ('lin_size', 'f4')])
                                           
            # Array to record all avalanches; this array gets really large: 10 GB for 1e6 drops on a 100 x 100 sandbox; use only if enough RAM available
            # avalanche = np.zeros(shape=(_SAND_DROPS,) + s.shape, dtype=np.bool)
            
            # Capture start time of main loop
            start = time.time()
            
            ### Do actual simulation ###
            
            # Show simulation for 2 dims via pyqtgraph
            if _PLOT_SIM and s.ndim == 2:
                app = QtWidgets.QApplication([])
                pg.setConfigOptions(antialias=True)
                title = '%i Drops On %s Sandbox' % (_SAND_DROPS, ' x '.join([str(dim) for dim in s.shape]))
                sim_plotter = SimulationPlotter(title=title, view=pg.PlotItem())
                do_simulation(s, _CRIT_H[i], _SAND_DROPS, _POINT, result_array, plot_simulation=sim_plotter, avalanche=None)
            # Just do simulation
            else:
                do_simulation(s, _CRIT_H[i], _SAND_DROPS, _POINT, result_array, avalanche=None)
            
            # Capture time of simulation
            _RUNTIME = time.time() - start
            
            # Remove events without avalanches
            # avalanche = avalanche[~(avalanche == 0).all(axis=tuple(range(1, avalanche.ndim)))]
            
            logging.info('Needed %.2f seconds for %i dropped grains in %s sandbox' % (_RUNTIME, _SAND_DROPS, str(s.shape)))
            
            # Plot all results
            if _PLOT_RES:
            
                plot_sandbox(s, _SAND_DROPS, point=_POINT)
                
                # Plot all histograms
                for field in result_array.dtype.names:
                    plot_hist(result_array[field], title=field)
            
            # Save the simulation results
            if _SAVE_SIMULATION:
                save_simulation(s, result_array, total_drops=_SAND_DROPS, point=_POINT)
                
            # Save the resulting sandbox
            if _SAVE_SANDBOX:
                save_sandbox(s, total_drops=_SAND_DROPS, point=_POINT)
            
            # Write timing with info to log
            with open('timing.log', 'a') as f:
                t = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
                msg = '%s, %i drops, %f seconds\n' % (str(s.shape), _SAND_DROPS, _RUNTIME)
                t += ':\t'
                f.write(t)
                f.write(msg)


if __name__ == "__main__":
    main()
    
    
    
