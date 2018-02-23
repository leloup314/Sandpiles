"""Implementation of slightly modified, iterative Bak-Tang-Wiesenfeld approach of cellular automata for sandpile dynamics"""

import os  # File saving etc.
import time  # Timing
import numpy as np  # Arrays with fast, vectorized operations and flexible structure
import matplotlib.pyplot as plt  # Plotting
import pyqtgraph as pg  # Plotting
import powerlaw as pl  # Fitting power laws

from PyQt5 import QtWidgets, QtCore  # Plotting
from matplotlib.backends.backend_pdf import PdfPages  # Plotting
from scipy.spatial.distance import cdist  # Calculate distances
from scipy.optimize import curve_fit  # Fitting
from numba import njit  # Speed-up


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
        
    # Write the statistics to results
    result['total_drops'] += i
    result['area'] += n_neighbours + 1
        

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
    
    # Drop sand iteratively
    for drop in xrange(total_drops):
        
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
            
            # Real-time plotting
            if plot_simulation:
                plot_simulation.setImage(s, levels=(0, crit_pile), autoDownsample=True)
                pg.QtGui.QApplication.processEvents()
        ### RESULTS ###        
                
        # Add initially dropped grain to total drops
        current_result['total_drops'] += 1
        
        # Store amount of relaxations within this iteration
        current_result['relaxations'] = relaxations
        
        # Get the linear size of the current avalanche if there were relaxations
        if relaxations != 0:
            coords = np.column_stack(np.where(current_avalanche))
            current_result['lin_size'] = np.amax(cdist(coords, coords))  # This gets slow (> 10 ms) for large (> 50 x 50) sandboxes but is still fastest choice
        
        # Reset avalanche configuration for use in next iteration
        tmp_avalanche[:] = 0
        
        # Real-time plotting
        if plot_simulation:
            plot_simulation.setImage(s, levels=(0, crit_pile), autoDownsample=True)
            pg.QtGui.QApplication.processEvents()
    
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
            

def plot_hist(data):
    """
    Histogramms data
    
    :param data: np.array of data to histogram
    """
    
    data_unique, data_count = np.unique(data, return_counts=True)
    counts, bin_edges, _ = plt.hist(data, bins=len(data_unique), alpha=0.7)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.errorbar(bin_centers, counts, yerr=np.sqrt(data_count), fmt='o', label='Bin centers')
    
    #fit = pl.Fit(counts)
    #cc=fit.power_law.alpha
    #print cc
    #def power_law(x, a, b):
    #    return a * np.power(x, -b)
    #p0 = (fit.power_law.sigma, cc)
    #popt, pcov = curve_fit(power_law, bin_centers, counts, p0=p0,sigma=np.sqrt(counts), absolute_sigma=True, maxfev=5000)
    #perr = np.sqrt(np.diag(pcov))
    #plt.plot(bin_centers, power_law(bin_centers, *popt), ls='--', label=r'$f(x)=a\cdot x^{-b}$'+'\n\t'+r'a=$%f\pm %f$; b$=%f\pm %f$' % (popt[0], perr[0], popt[1], perr[1]))
    plt.legend()
    plt.loglog()
    plt.grid()
    plt.show()
    
            
def save_array(array, out_file):
    if os.path.isfile(out_file):
        i = 0
        a, b = out_file.split('.')
        while os.path.isfile(a+str(i)+b):
            i += 1
        out_file = a + '_%i.' % i + b
        
    np.save(out_file, array)
    
    
def save_simulation(s, sim, total_drops=None, point=None, out_file=None):
    if out_file is None:

        sim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulations')
        
        td = str(total_drops) if total_drops is not None else '?'
        
        if point is None:
            sm = '%s_simulation_%s_drops_random.npy' % ('x'.join([str(dim) for dim in s.shape]), td)
        else:
            sm = '%s_simulation_%s_drops_at_%s.npy' % ('x'.join([str(dim) for dim in s.shape]), td, '_'.join(str(c) for c in point))
        
        out_file = os.path.join(sim_path, sm)
        
    save_array(sim, out_file)


def save_sandbox(s, total_drops=None, point=None, out_file=None):
    if out_file is None:

        sandbox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sandboxes')
        
        td = str(total_drops) if total_drops is not None else '?'
        
        if point is None:
            sb = '%s_sandbox_%s_drops_random.npy' % ('x'.join([str(dim) for dim in s.shape]), td)
        else:
            sb = '%s_sandbox_%s_drops_at_%s.npy' % ('x'.join([str(dim) for dim in s.shape]), td, '_'.join(str(c) for c in point))
        
        out_file = os.path.join(sandbox_path, sb)
        
    save_array(s, out_file)
    
    
class SimulationPlotter(pg.GraphicsWindow):
    
    def __init__(self, parent=None):
        super(SimulationPlotter, self).__init__(parent=parent)
        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.img = pg.ImageItem(border='w')
        
        #pos = np.array([0.0, 0.6, 1.0])
        #color = np.array([[25, 25, 112, 255], [173, 255, 47, 255], [255, 0, 0, 255]], dtype=np.ubyte)
        #cmap = pg.ColorMap(pos, color)
        #lut = cmap.getLookupTable(0.0, 1.0, 100)
        #self.img.setLookupTable(lut)

        self.hist = pg.HistogramLUTItem()#rgbHistogram=True)
        self.hist.setImageItem(self.img)
        self.plot = self.addPlot()
        
        self.plot.addItem(self.img)
        self.addItem(self.hist)
        
    def plotSimulation(self, s, crit_pile, total_drops, point, result_array, avalanche):
        self.hist.setHistogramRange(0, crit_pile)
        do_simulation(s, crit_pile, total_drops, point, result_array, self.img, avalanche)
        self.close()


def main():
    
    ### Initialization simulation variables ### 
    
    # Number of dimensions
    _DIM = 2
    
    # Length of sandbox
    _L = 100
    
    # Set critical sand pile height; usually equal to number of neighbours
    _CRIT_H = 2 * _DIM
    
    # Number of total sand drops
    _SAND_DROPS = 100000
    
    # Point to drop to in sandbox;if None, drop randomly
    _POINT = None
    
    # Whether to plot results
    _PLOT_RES = False
    
    # Whether to plot the evolution of the sandbox
    _PLOT_SIM = False
    
    # Save results of simulation after simulation is done
    _SAVE_SIMULATION = True
    
    # Save sandbox after simulation is done
    _SAVE_SANDBOX = False
    
    # Save runtime of simulation
    _RUNTIME = 0
    
    # Make structured np.array to store results in
    result_array = np.array(np.zeros(shape=_SAND_DROPS),
                            dtype=[('relaxations', 'i4'), ('area', 'i4'),
                                   ('total_drops', 'i4'), ('lin_size', 'f4')])
                                   
    # Array to record all avalanches; this array gets really large: 10 GB for 1e6 drops on a 100 x 100 sandbox; use only if enough RAM available
    # avalanche = np.zeros(shape=(_SAND_DROPS,) + s.shape, dtype=np.bool)
    
    # Init sandbox
    s = init_sandbox(_L, _DIM)
    
    # Fill sandbox until critical
    fill_sandbox(s, _CRIT_H, level=0.75)
    
    # Capture start time of main loop
    start = time.time()
    
    ### Do actual simulation ###
    
    # Show simulation for 2 dims via real-time plotting
    if _PLOT_SIM and s.ndim == 2:
        app = QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        sim_plotter = SimulationPlotter()
        sim_plotter.plotSimulation(s, _CRIT_H, _SAND_DROPS, _POINT, result_array, avalanche=None)
        sim_plotter.show()
        app.exec_()
    
    # Just do simulation
    else:
        do_simulation(s, _CRIT_H, _SAND_DROPS, _POINT, result_array, avalanche=None)
    
    # Capture time of simulation
    _RUNTIME = time.time() - start
    
    # Remove events without avalanches
    # avalanche = avalanche[~(avalanche == 0).all(axis=tuple(range(1, avalanche.ndim)))]
    
    print 'Needed %f for %i dropped grains' % (_RUNTIME, _SAND_DROPS)
    
    # Plot all results
    if _PLOT_RES:
    
        plot_sandbox(s, _SAND_DROPS, point=_POINT)
        
        # Plot all histograms
        for field in result_array.dtype.names:
            plot_hist(result_array[field])
    
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
    
    
    
