#!/usr/bin/env python

"""Implementation of slightly modified, iterative Bak-Tang-Wiesenfeld approach of cellular automata for sandpile dynamics"""

import os  # File saving etc.
import time  # Timing
import logging  # User feedback
import numpy as np  # Arrays with fast, vectorized operations and flexible structure
import matplotlib.pyplot as plt  # Plotting
import argparse  # simulation setup file

from matplotlib.backends.backend_pdf import PdfPages  # Plotting
from scipy.spatial.distance import cdist, pdist  # Calculate distances
from numba import njit  # Speed-up
from collections import Iterable  # Type checking

logging.basicConfig(level=logging.INFO)

try:
    yaml_flag = False
    import yaml  # simulation setup file; don't require yaml if setup is selected in script
except ImportError:
    yaml_flag = True
    logging.info('Starting simulation from setup.yaml disabled. Could not import yaml.')
    
try:
    pg_flag = False
    import pyqtgraph as pg  # Plotting; don't require pg if no live plotting
except ImportError:
    pg_flag = True
    logging.info('Plotting live evolution of sandpiles disabled. Could not import pyqtgraph.')


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
    
    # Get amount of grains which will be redistributed and don't fall of the lattice by comparing number of actual neighbours to 2*dimension (number of default neighbours)
    to_drop = int(amount * n_neighbours / (2.0 * s.ndim))
    
    # Redistribute to_drop among neighbours
    i = 0
    while to_drop:
        n = neighbours[i % n_neighbours]
        s[n] += 1
        to_drop -= 1
        i += 1
        
    # Increase size of avalanche by i dropped grains
    result['size'] += i
        

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
    
    # Make temporary array to store avalanche configuration in order to calculate linear size and area
    tmp_avalanche = np.zeros_like(s, dtype=np.bool)
    
    # Flag indicating whether or not to calculate the lin_size; large avalanches (in sandboxes > 2 dimensions, > 50 length) cause several 10 GB RAM consumption when calculating lin_size
    lin_size_flag = True if np.power(s.shape[0], s.ndim) > 50**3 else False
    
    #Feedback
    if lin_size_flag:
        logging.info('No calculation of "lin_size" for %s sandbox. Too large.' % str(s.shape))
    
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
                plot_simulation.setData(s)
                pg.QtGui.QApplication.processEvents()
        
        ### RESULTS ###        
                
        # Add initially dropped grain to avalanche size
        current_result['size'] += 1
        
        # Store amount of relaxations within this iteration; equivalent to avalanche duration
        current_result['duration'] = relaxations
        
        # Get number of sites participating in current avalanche
        current_result['area'] = np.count_nonzero(current_avalanche)
        
        # Get the linear size of the current avalanche if there were relaxations
        if relaxations != 0 and not lin_size_flag:
            
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
            plot_simulation.setData(s)
            pg.QtGui.QApplication.processEvents()
            
            
def fill_sandbox(s, crit_pile, level=0.5):
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
    cmap = plt.get_cmap('jet', np.max(s)-np.min(s)+2)  # discrete colormap
    img = plt.imshow(s, cmap=cmap, vmin=np.min(s) - 0.5, vmax=np.max(s) + 1 + 0.5)  # make image with colormap BlueGreenRed
    plt.colorbar(img, ticks=np.arange(np.min(s), np.max(s)+2))  # add colorbar
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
    
    
class ColorBar(pg.GraphicsWidget):
    """
    Adds colorscale for ImageItem
    
    Modified from https://gist.github.com/maedoc/b61090021d2a5161c5b9
    """
    def __init__(self, cmap, width, height, ticks=None, tick_labels=None, label=None):
        pg.GraphicsWidget.__init__(self)

        # handle args
        label = label or ''
        w, h = width, height
        stops, colors = cmap.getStops('float')
        smn, spp = stops.min(), stops.ptp()
        stops = (stops - stops.min())/stops.ptp()
        if ticks is None:
            ticks = np.r_[0.0:1.0:5j, 1.0] * spp + smn
        tick_labels = tick_labels or ["%0.2g" % (t,) for t in ticks]

        # setup picture
        self.pic = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.pic)

        # draw bar with gradient following colormap
        p.setPen(pg.mkPen('k'))
        grad = pg.QtGui.QLinearGradient(w/2.0, 0.0, w/2.0, h*1.0)
        for stop, color in zip(stops, colors):
            grad.setColorAt(1.0 - stop, pg.QtGui.QColor(*[255*c for c in color]))
        p.setBrush(pg.QtGui.QBrush(grad))
        p.drawRect(pg.QtCore.QRectF(0, 0, w, h))

        # draw ticks & tick labels
        mintx = 0.0
        for tick, tick_label in zip(ticks, tick_labels):
            y_ = (1.0 - (tick - smn)/spp) * h
            p.drawLine(0.0, y_, -5.0, y_)
            br = p.boundingRect(0, 0, 0, 0, pg.QtCore.Qt.AlignRight, tick_label)
            if br.x() < mintx:
                mintx = br.x()
            p.drawText(br.x() - 10.0, y_ + br.height() / 4.0, tick_label)

        # draw label
        br = p.boundingRect(0, 0, 0, 0, pg.QtCore.Qt.AlignRight, label)
        p.drawText(-br.width() / 2.0, h + br.height() + 5.0, label)
        
        # done
        p.end()

        # compute rect bounds for underlying mask
        self.zone = mintx - 12.0, -15.0, br.width() - mintx, h + br.height() + 30.0
        
    def paint(self, p, *args):
        # paint underlying mask
        p.setPen(pg.QtGui.QColor(255, 255, 255, 0))
        p.setBrush(pg.QtGui.QColor(255, 255, 255, 200))
        p.drawRoundedRect(*(self.zone + (9.0, 9.0)))
        
        # paint colorbar
        p.drawPicture(0, 0, self.pic)
        
    def boundingRect(self):
        return pg.QtCore.QRectF(self.pic.boundingRect())
    
    
class SimulationPlotter(pg.GraphicsWindow):
    """
    Subclass of pyqtgraph.GraphicsWindow to plot evolution of sandpiles in sandbox.
    Simulation continues even after closing the plotting window.
    
    :param s: np.array of sandbox
    :param crit_pile: int of critical pile height
    """
    
    def __init__(self, s, crit_pile, title=None, parent=None, **kwargs):
        super(SimulationPlotter, self).__init__(parent=parent, **kwargs)
        
        # Set window title
        self.setWindowTitle(title)
        
        # Store critical pile height
        self.crit_pile = crit_pile
        
        # Creat plot item
        self.plot = self.addPlot()

        # Make bar plot for 1 dim
        if s.ndim == 1:
            self.img = pg.BarGraphItem(x=np.arange(s.shape[0]), height=self.crit_pile, width=0.5)
        # Make image for 2 dim
        elif s.ndim == 2:
            self.img = pg.ImageItem()
            # make colormap
            stops = np.linspace(0, 1, self.crit_pile + 1)
            colors = np.array([[0.0, 0.0, 0.5, 1.0], [0.0, 0.5, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.55, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
            cm = pg.ColorMap(stops, colors)
            self.img.setLookupTable(cm.getLookupTable())
            cb = ColorBar(cm, self.width()*0.05, self.height()*0.9, tick_labels=[str(i) for i in range(self.crit_pile + 1)])
            self.addItem(cb)
        # Add image to plot
        self.plot.addItem(self.img)
        
    def setData(self, data):
        if data.ndim == 1:
            self.img.setOpts(height=data)
        elif data.ndim == 2:
            self.img.setImage(data, levels=(0, self.crit_pile), autoDownsample=True)        
    
    
### Saving/loading results functions ###
    
            
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
        while os.path.isfile(a + '_%i.' % i + b):
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
        
        # If path does not exist, make it
        if not os.path.exists(sim_path):
            os.mkdir(sim_path)
        
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
        
        # If path does not exist, make it
        if not os.path.exists(sandbox_path):
            os.mkdir(sandbox_path)
        
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
    
    
def get_critical_sandbox(length, dimension, path=None):
    """
    Method to load a previously saved, critical sandbox and return it.
    If none is found, init a sandbox and fill it.
    
    :param length: int sandbox length
    :param dimension: int sandbox dimension
    :param path: str of custom location where sandbox files are or None; if None, look in default path
    """
    
    # Path where sandboxes are stored or given path were sandboxes are
    sandbox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../simulations/sandboxes') if path is None else path
    
    # Default saving pattern which file needs to have
    required_pattern = 'x'.join([str(dim) for dim in [length] * dimension])
    
    # If path does exist, loop through .npy files, if patterns match, load and return
    if os.path.exists(sandbox_path):
        for sandbox in os.listdir(sandbox_path):
            if sandbox.endswith(".npy"):
                pattern = sandbox.split('_')[0]
                if pattern == required_pattern:
                    logging.info('Loading critical %s sandbox from %s in %s' % (required_pattern, sandbox, sandbox_path))
                    return np.load(os.path.join(sandbox_path, sandbox))
    
    # No sandboxes were found or path does not exist; init sandbox and return
    logging.info('Initializing %s sandbox and filling up.' % required_pattern)
    s = init_sandbox(length, dimension)
    fill_sandbox(s, 2 * s.ndim)
    return s


### Main ###


def main(length=None, dimension=None, crit_pile=None, total_drops=None, point=None, save_sim=False, save_sbox=False, plot_sim=False, plot_res=False):
    
    ### Initialization simulation variables ###
    
    # Length of sandbox; can be iterable for several simulations
    _LEN = 100 if length is None else length
    
    # Dimensions; can be iterable for several simulations
    _DIM = 2 if dimension is None else dimension
    
    # Set critical sand pile height; usually equal to number of neighbours
    _CRIT_H = (tuple(2 * _D for _D in _DIM) if isinstance(_DIM, Iterable) else 2 * _DIM) if crit_pile is None else crit_pile
    
    # Number of total sand drops
    _SAND_DROPS = 10000 if total_drops is None else total_drops
    
    # Point to drop to in sandbox;if None, drop randomly
    _POINT = point
    
    # Whether to plot results
    _PLOT_RES = plot_res
    
    # Whether to plot the evolution of the sandbox
    _PLOT_SIM = plot_sim
    
    # Save results of simulation after simulation is done
    _SAVE_SIMULATION = save_sim
    
    # Save sandbox after simulation is done
    _SAVE_SANDBOX = save_sbox
    
    # Check for multiple lengths and dimensions
    _LEN = _LEN if isinstance(_LEN, Iterable) else [_LEN]
    _DIM = _DIM if isinstance(_DIM, Iterable) else [_DIM]
    _CRIT_H = _CRIT_H if isinstance(_CRIT_H, Iterable) else [_CRIT_H]
    
    # Do simulation for multiple sandbox lengths and dimensions in loops
    for i, _D in enumerate(_DIM):    
        for _L in _LEN:
            
            # Get critical sandbox
            s = get_critical_sandbox(_L, _D)
        
            # Make structured np.array to store results in
            result_array = np.array(np.zeros(shape=_SAND_DROPS),
                                    dtype=[('duration', 'i4'), ('area', 'i4'),
                                           ('size', 'i4'), ('lin_size', 'f4')])
                                           
            # Array to record all avalanches; this array gets really large: 10 GB for 1e6 drops on a 100 x 100 sandbox; use only if enough RAM available
            # avalanche = np.zeros(shape=(_SAND_DROPS,) + s.shape, dtype=np.bool)
            
            # Capture start time of main loop
            start = time.time()
            
            ### Do actual simulation ###
            
            # Show simulation for 1 or 2 dims via pyqtgraph if pg_flag is False
            if _PLOT_SIM and s.ndim in (1, 2) and not pg_flag:
                app = pg.QtGui.QApplication([])
                pg.setConfigOptions(antialias=True)
                pg.setConfigOption('background', 'w')
                pg.setConfigOption('foreground', 'k')
                title = '%i Drops On %s Sandbox' % (_SAND_DROPS, ' x '.join([str(dim) for dim in s.shape]))
                sim_plotter = SimulationPlotter(s, _CRIT_H[i], title=title)
                do_simulation(s, _CRIT_H[i], _SAND_DROPS, _POINT, result_array, plot_simulation=sim_plotter, avalanche=None)
                app.deleteLater()  # Important for several simulations
            # Just do simulation
            else:
                do_simulation(s, _CRIT_H[i], _SAND_DROPS, _POINT, result_array, avalanche=None)
            
            # Capture time of simulation
            _RUNTIME = time.time() - start
            
            # Remove events without avalanches
            # avalanche = avalanche[~(avalanche == 0).all(axis=tuple(range(1, avalanche.ndim)))]
            
            logging.info('Needed %.2f seconds for %i dropped grains in %s sandbox' % (_RUNTIME, _SAND_DROPS, str(s.shape)))
            
            # Save the simulation results
            if _SAVE_SIMULATION:
                
                out_file = _SAVE_SIMULATION if isinstance(_SAVE_SIMULATION, str) else None
                
                save_simulation(s, result_array, total_drops=_SAND_DROPS, point=_POINT, out_file=out_file)
                
            # Save the resulting sandbox
            if _SAVE_SANDBOX:
                
                out_file = _SAVE_SANDBOX if isinstance(_SAVE_SANDBOX, str) else None
                
                save_sandbox(s, total_drops=_SAND_DROPS, point=_POINT, out_file=out_file)
                
            # Plot all results
            if _PLOT_RES:
                
                if s.ndim == 2:
                    plot_sandbox(s, _SAND_DROPS, point=_POINT)
                
                # Plot all histograms
                for field in result_array.dtype.names:
                    plot_hist(result_array[field], title=field)
                    
            # Write timing with info to log
            with open('timing.log', 'a') as f:
                t = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
                msg = '%s, %i drops, %f seconds\n' % (str(s.shape), _SAND_DROPS, _RUNTIME)
                t += ':\t'
                f.write(t)
                f.write(msg)


if __name__ == "__main__":
    
    # Possibility to get simulation setup from yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setup', help='Yaml-file with simulation setup', required=False)
    args = vars(parser.parse_args())
    
    # Load simulation setup from yaml file
    if args['setup'] and not yaml_flag:
        with open(args['setup'], 'r') as setup:
            simulation_setup = yaml.safe_load(setup)
        logging.info('Starting simulation from setup file %s:%s' % (str(args['setup']), '\n\n\t' + '\n\t'.join(str(key) + ': ' + str(simulation_setup[key]) for key in simulation_setup.keys()) + '\n'))
    # Use default values
    else:
        simulation_setup = {}
    
    # Start simulation
    main(**simulation_setup)
    
    
    
