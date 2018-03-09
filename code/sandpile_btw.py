#!/usr/bin/env python

"""
Implementation of slightly modified, iterative Bak-Tang-Wiesenfeld approach of cellular automata for sandpile dynamics.
The lattice sites in the sandbox in the context of this model must be regarded as slopes, not actual heights of sand piles. 
"""

import os  # File saving etc.
import time  # Timing
import logging  # User feedback
import numpy as np  # Arrays with fast, vectorized operations and flexible structure
import argparse  # simulation setup file

from scipy.spatial.distance import cdist, pdist  # Calculate distances
from numba import njit  # Speed-up
from collections import Iterable  # Type checking
from plot_utils import plot_hist, plot_sandbox  # plotting
from tools import save_simulation, save_sandbox

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
    from plot_utils import SimulationPlotter  # Live plotting
except ImportError:
    pg_flag = True
    logging.info('Plotting live evolution of sandbox disabled. Could not import pyqtgraph.')


### Simulation functions ###


def init_sandbox(n, dim, state=None):
    """
    Initialises a square NxN sandbox lattice on which the simulation is done.

    :param n: int of lattice sites of sandbox
    :param dim: int of dimension like n^dim
    :param state: str some state of the initial sandbox

    """
    critical_slope = 2 * dim 

    if state == 'one':
        res = np.ones(shape=(n, ) * dim, dtype=np.int16)
    elif state == 'crit':
        res = np.random.choice([critical_slope-1, critical_slope], shape=(n, ) * dim, dtype=np.int16)
    elif state == 'over_crit':
        res = np.random.choice([critical_slope, critical_slope + 1], shape=(n, ) * dim, dtype=np.int16)
    else:  # None, ground state
        res = np.zeros(shape=(n, ) * dim, dtype=np.int16)

    return res


def neighbours(s, site, distance=1, metric='euclidean'):
    """
    Returns list of tuples of nearest neighbours of site in sandbox s with distance.
    Returns only neighbours who are actually on the lattice of s.
    
    :param s: np.array of sandbox
    :param site: tuple of coordinates of site
    :param distance: scalar of distance the neighbours must have to site
    :param metric: str of any accepted metric by cdist
    """
    
    # Make array of coordinates of all lattice sites
    coordinates = np.column_stack(np.indices(s.shape).reshape(s.ndim, np.product(s.shape)))
    
    # Return list of coordinate tuples with distance to site
    return map(tuple, coordinates[(cdist(np.array([site]), coordinates, metric=metric) == distance)[0]])
            

@njit
def relax_site(s, critical_site, neighbours, critical_slope):
    """
    Relaxes a critical site of the sandbox s by substracting critical_slope of critical_site
    an redistributing it to the neighbours. If there are less neighbouring sites
    than the geometry of the sandbox implies, avoid redistribution of reduced slope
    which is same as letting slope units fall off lattice.
    
    :param s: np.array of sandbox
    :param critical_site: tuple of coordinates of critical site
    :param neighbours: array of coordinates of tuples of neighbouring sites of critical_site
    :param critical_slope: int of critical_slope that critical_site is reduced by
    """
    
    # Relax critical site
    s[critical_site] -= critical_slope
    
    # Get number of neighbours
    n_neighbours = len(neighbours)
    
    # Get amount of slope which will be redistributed (doesn't fall of the lattice) by comparing
    # number of actual neighbours to 2*dimension (number of default neighbours)
    to_drop = int(critical_slope * n_neighbours / (2.0 * s.ndim))

    # Redistribute to_drop among neighbours
    i = 0
    while to_drop:
        n = neighbours[i % n_neighbours]
        s[n] += 1
        to_drop -= 1
        i += 1
        

def relax_sandbox(s, critical_sites, critical_slope, neighbour_LUD, avalanche):
    """
    Relaxes s by looping over all critical_sites and calling relax_site on them.
    Using a look-up dictionary with sites as keys and their neighbours as values in
    order to avoid calculating the neighbours of a given site twice.
    
    :param s: np.array of sandbox
    :param critical_sites: list of tuples of coordinates of critical sites
    :param critical_slope: int of critical_slope each critical_site is reduced by
    :param neighbour_LUD: dict that is/will be filled with site, neighbours as key,values
    :param avalanche: np.array of bools like s; stores which sites took part in avalanche
    """
    
    # Relax all critical sites
    for site in critical_sites:
        
        # Get neighbours of current critical site; either from look-up or function call
        if site in neighbour_LUD:
            n = neighbour_LUD[site]
        else:
            n = neighbours(s, site)
            neighbour_LUD[site] = n  # Write to look-up dict

        # Relax current critical site
        relax_site(s, site, n, critical_slope)
                
        # Critical site and neighbours took part in the avalanche; set to 1
        avalanche[site] = 1
        for nn in n:
            avalanche[nn] = 1        


def drive_simulation(s, site=None, amount=1, mode='nc'):
    """
    Drives the simulation by adding slope units to the sandbox.
    If mode is non-conservative ('nc'), add amount of slope to site.
    If mode is conservative ('c'), add to slope at site and reduce uphill neighbours.
    If site is None (default), add to random site.
    
    :param s: np.array of sandbox
    :param site: tuple of coordinates of site or None
    :param amount: int amount of slope which is added to site in 'nc' mode
    :param mode: str either 'nc' for non-conservative or 'c' for conservative adding
    """
    
    # Make coordinates of random site within lattice
    if site is None:
        site = np.random.randint(low=0, high=np.amin(s.shape), size=s.ndim)
    
    # Add non-conservative
    if mode == 'nc':
        # Add amount of slope to site
        s[tuple(site)] += amount
    
    # Add conservative; adding to slope also affects conservative neighbour slopes
    elif mode == 'c':
        
        # Add s.ndim slope to site
        s[tuple(site)] += s.ndim
        
        # Get uphill neighbours
        n = np.array(neighbours(s, tuple(site)))
        uphill_n = tuple(n[np.where(np.sum(n - np.array(site), axis=1) == -1)])
        
        # Reduce uphill slopes if there are conservative neighbours
        if uphill_n:
            s[uphill_n] -= 1
        

def do_simulation(s, critical_slope, total_drives, site, result_array, plot_simulation=False, avalanche=None):
    """
    Does entire simulation of sandbox evolution by driving the simulation total_drives times by adding slope to (random) site.
    Checks after each drive whether some sites are at or above critical_slope. As long as that's the case, relaxes these sites
    simultaniously. Then moves on to next drive.
    
    :param s: np.array of sandbox
    :param critical_slope: int of critical slope
    :param total_drives: int of total simulation drives (iterations)
    :param site: tuple of coordinates of site on which the sand is dropped or None; if None (default), add randomly
    :param result_array: np.array in which the results of all drives are stored
    :param plot_simulation: False or SimulationPlotter; If SimulationPlotter, the entire evolution of the sandbox will be plot
    :param avalanche: np.array of np.arrays of bools like s or None; if None, dont store avalanche configurations,
                      else store which sites took part in avalanche for each iteration
    """
    
    # Make look-up dict of site coordinates as keys and their neighbours as values; speed-up by a factor of 10
    neighbour_LUD = {}
    
    # Make temporary array to store avalanche configuration in order to calculate linear size and area
    tmp_avalanche = np.zeros_like(s, dtype=np.bool)
    
    # Flag indicating whether or not to calculate the lin_size; large avalanches (in sandboxes >= 2 dimensions, > 100 length)
    # cause several 10 GB RAM consumption when calculating lin_size
    lin_size_flag = True if np.power(s.shape[0], s.ndim) > 100**2 else False
    
    #Feedback
    if lin_size_flag:
        logging.info('No calculation of "lin_size" for %s sandbox. Too large.' % str(s.shape))
    
    # Timing estimate
    estimate_time = time.time()

    # Drive simulation iteratively
    for drive in xrange(total_drives):
        
        # Feedback
        if drive % (5e-3 * total_drives) == 0 and drive > 0:
            # Timing estimate
            avg_time = (time.time() - estimate_time) / drive  # Average time taken for the last 0.5% of total_drives
            est_hours = avg_time * (total_drives-drive)/60**2
            est_mins = (est_hours % 1) * 60
            est_secs = (est_mins % 1) * 60
            msg = 'At drive %i of %i total drives (%.1f %s). Estimated time left: %i h %i m %i s' % (drive, total_drives, 100 * float(drive)/total_drives, "%",
                  int(est_hours), int(est_mins), int(est_secs))
            logging.info(msg)
        
        # Extract result array for current iteration
        current_result = result_array[drive]
        
        # Extract array for recording avalanches if avalanche array is given; else only store current configuration and reset after
        current_avalanche = tmp_avalanche if avalanche is None else avalanche[drive]
        
        # Drive simulation by adding slope units
        drive_simulation(s, site)
        
        # Initialise counter and start relaxations if there are critical sites
        relaxations = 0
        while True in (s >= critical_slope):
            
            # Get critical sites
            critical_sites = map(tuple, np.column_stack(np.where(s >= critical_slope)))
            
            # Relax the sandbox simultaneously
            relax_sandbox(s, critical_sites, critical_slope, neighbour_LUD, current_avalanche)
            
            # Increment relaxation counter
            relaxations += 1
            
            ### RESULTS ###
            
            # Add to avalanche size
            current_result['size'] += len(critical_sites)
            
            # Fast plotting
            if plot_simulation:
                plot_simulation.setData(s)
                pg.QtGui.QApplication.processEvents()
        
        ### RESULTS ###
        
        # Store amount of relaxations within this iteration; equivalent to avalanche duration
        current_result['duration'] = relaxations
        
        # Get number of sites participating in current avalanche
        current_result['area'] = np.count_nonzero(current_avalanche)
                
        # Get the linear size of the current avalanche if there were relaxations and flag is not set
        if relaxations != 0 and not lin_size_flag:
            
            # Get coordinates of avalanche sites
            coords = np.column_stack(np.where(current_avalanche))
            
            # Get maximum distance between them
            try:
                current_result['lin_size'] = np.amax(pdist(coords))  # This gets slow (> 10 ms) for large (> 50 x 50) sandboxes but is still fastest choice
            
            # Memory error for large amount of avalanche sites
            except MemoryError:
                logging.warning('Memory error due to large avalanche for drive %i. No "lin_size" calculated.' % drive)
                pass  # Do nothing
                
        # Reset avalanche configuration for use in next iteration
        tmp_avalanche.fill(0)
        
        # Fast plotting
        if plot_simulation:
            plot_simulation.setData(s)
            pg.QtGui.QApplication.processEvents()
            

def make_critical(s, critical_slope):
    """
    Drives sandbox s until it reaches state of SOC. Therefore, drives in 10**(dim + 1) intervals
    and checks mean slope after each interval. If mean slope does not change anymore, breaks. 
    
    :param s: np.array of sandbox
    :param critical_slope: int critical slope of sandbox
    """
    
    # Mean variables to compare mean slopes between intervals
    mean_previous = 0
    mean_current = 0
    
    # Look-up dict for neighbours and avalanche array
    avalanche = np.zeros_like(s)
    neighbour_LUD = {}
    
    # Number of how many times the difference of mean_current and mean_previous must be within specified difference; empirical
    n_conditions = 0 if s.ndim > 3 else int(1000/s.shape[0] * 1.0/s.ndim - s.ndim**2) if int(1000/s.shape[0] * 1.0/s.ndim - s.ndim**2) >= 0 else 0 
    difference = 10**(-s.ndim + 1) if 10**(-s.ndim + 1) >= 1e-2 else 1e-2
    
    # Add counter for how often condition was met
    counter = 0
    
    # Feedback
    logging.info('Driving %s sandbox to state of SOC...' % str(s.shape))
    
    # Loop counter
    i = 0
    while True:
        
        # Drive simulation by adding slope units randomly
        drive_simulation(s)
        
        # Relax sandbox
        while True in (s >= critical_slope):
            
            # Get critical sites
            critical_sites = map(tuple, np.column_stack(np.where(s >= critical_slope)))
            
            # Relax the sandbox simultaneously
            relax_sandbox(s, critical_sites, critical_slope, neighbour_LUD, avalanche)
        
        # Compare mean slope of current and previous interval
        if i % 10**(s.ndim + 1) == 0 and i != 0:
            mean_previous = np.mean([mean_previous, mean_current])
            mean_current = np.mean(s)
            
            # Increase condition counter if clause is met
            if np.absolute(mean_previous-mean_current) <= difference and mean_current >= 0.9 * s.ndim:
                counter += 1
            # If condition is met sufficient times, break
            if counter > n_conditions:
                break
                
            # Feedback
            logging.info('Iteration %i; mean slope at %.3f' % (i, mean_current))
        
        i += 1
    
    # Feedback
    logging.info('Mean slope %.3f after %i iterations.' % (mean_current, i))
    
    
def get_critical_sandbox(length, dimension, model, force_new=False, path=None):
    """
    Method to load a previously saved sandbox in state of SOC and return it.
    If none is found, init a sandbox and drive it to state of SOC.
    
    :param length: int sandbox length
    :param dimension: int sandbox dimension
    :param path: str of custom location where sandbox files are or None; if None, look in default path
    :param model: str either 'btw' or 'custom'
    """

    # If we do not want to force a new critical sandbox
    if not force_new:
        
        # Prefix of default saving pattern which file needs to have
        required_prefix = 'x'.join([str(dim) for dim in [length] * dimension])
        
        # Suffix of default saving pattern which file needs to have
        required_suffix = model
        
        # Path where sandboxes are stored or given path were sandboxes are
        sandbox_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../simulations/sandboxes')) if path is None else path

        # If path does exist, loop through .npy files, if patterns match, load and return
        if os.path.exists(sandbox_path):
            for sandbox in os.listdir(sandbox_path):
               
                if sandbox.endswith(".npy"):
                    
                    # Get rid of extension
                    tmp = sandbox.split('.')[0]
                    
                    # Get prefix
                    prefix = tmp.split('_')[0]
                    
                    # For several saved simulations of the same sandbox dimension, suffix can be either integer or model
                    try:
                        _ = int(tmp.split('_')[-1])
                        suffix = tmp.split('_')[-2]
                    except ValueError:
                        suffix =  tmp.split('_')[-1]
                    
                    # Take first sandbox that matches and return
                    if prefix == required_prefix and suffix == required_suffix :
                        logging.info('Loading critical %s %s model sandbox from %s in %s' % (required_prefix, required_suffix.upper(), sandbox, sandbox_path))
                        return np.load(os.path.join(sandbox_path, sandbox))
    
    # No sandboxes were found or path does not exist; init sandbox and return
    logging.info('Initializing %s sandbox.' % str((length, ) * dimension))
    s = init_sandbox(length, dimension)
    make_critical(s, 2 * s.ndim)
    return s


### Main ###


def main(length=None, dimension=None, critical_slope=None, total_drives=None, site=None, save_sim=False, save_sbox=False, plot_sim=False, plot_res=False):
    
    ### Initialization simulation variables ###
    
    # Variable describing this simulations model
    _MODEL = 'btw'
    
    # Length of sandbox; can be iterable for several simulations
    _LEN = 50 if length is None else length
    
    # Dimensions; can be iterable for several simulations
    _DIM = 2 if dimension is None else dimension
    
    # Set critical slope; usually equal to number of neighbouring sites
    _CRIT_S = (tuple(2 * _D for _D in _DIM) if isinstance(_DIM, Iterable) else 2 * _DIM) if critical_slope is None else critical_slope
    
    # Number of total drives of simulation
    _TOTAL_DRIVES = 10000 if total_drives is None else total_drives
    
    # Site to drop to in sandbox; if None, drop randomly
    _SITE = site
    
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
    _CRIT_S = _CRIT_S if isinstance(_CRIT_S, Iterable) else [_CRIT_S]
    
    # Do simulation for multiple sandbox lengths and dimensions in loops
    for i, _D in enumerate(_DIM):    
        for _L in _LEN:
            
            # Get sandbox in state of SOC
            s = get_critical_sandbox(_L, _D, model=_MODEL)
            
            # Make structured np.array to store results in
            result_array = np.array(np.zeros(shape=_TOTAL_DRIVES), dtype=[('duration', 'i4'), ('area', 'i4'), ('size', 'i4'), ('lin_size', 'f4')])
                                           
            # Array to record all avalanches; this array gets really large: 10 GB for 1e6 drives on a 100 x 100 sandbox; use only if enough RAM available
            # avalanche = np.zeros(shape=(_TOTAL_DRIVES,) + s.shape, dtype=np.bool)
            
            # Capture start time of main loop
            start = time.time()
            
            ### Do actual simulation ###
            
            # Show simulation for 1 or 2 dims via pyqtgraph if pg_flag is False
            if _PLOT_SIM and s.ndim in (1, 2) and not pg_flag:
                app = pg.QtGui.QApplication([])
                pg.setConfigOptions(antialias=True)
                pg.setConfigOption('background', 'w')
                pg.setConfigOption('foreground', 'k')
                title = '%i Drives In %s Sandbox' % (_TOTAL_DRIVES, ' x '.join([str(dim) for dim in s.shape]))
                sim_plotter = SimulationPlotter(s, _MODEL, _CRIT_S[i], title=title)
                do_simulation(s, _CRIT_S[i], _TOTAL_DRIVES, _SITE, result_array, plot_simulation=sim_plotter, avalanche=None)
                app.deleteLater()  # Important for several simulations
            # Just do simulation
            else:
                do_simulation(s, _CRIT_S[i], _TOTAL_DRIVES, _SITE, result_array, avalanche=None)
            
            # Capture time of simulation
            _RUNTIME = time.time() - start
            
            # Remove events without avalanches
            # avalanche = avalanche[~(avalanche == 0).all(axis=tuple(range(1, avalanche.ndim)))]
            
            logging.info('Needed %.2f seconds for %i drives in %s sandbox' % (_RUNTIME, _TOTAL_DRIVES, str(s.shape)))
            
            # Save the simulation results
            if _SAVE_SIMULATION:
                
                out_file = _SAVE_SIMULATION if isinstance(_SAVE_SIMULATION, str) else None
                
                save_simulation(s, result_array, model=_MODEL, total_drives=_TOTAL_DRIVES, site=_SITE, out_file=out_file)
                
            # Save the resulting sandbox
            if _SAVE_SANDBOX:
                
                out_file = _SAVE_SANDBOX if isinstance(_SAVE_SANDBOX, str) else None
                
                save_sandbox(s, model=_MODEL, total_drives=_TOTAL_DRIVES, site=_SITE, out_file=out_file)
                
            # Plot all results if no live plotting; pyqt and matplotlib don't work together
            if _PLOT_RES and (not _PLOT_SIM or s.ndim > 2 or pg_flag):
                
                if s.ndim == 2:
                    plot_sandbox(s, _TOTAL_DRIVES, site=_SITE)
                
                # Plot all histograms
                for field in result_array.dtype.names:
                    plot_hist(result_array[field], field, binning=False, title=None)
                    
            # Write timing with info to log
            with open('timing.log', 'a') as f:
                t = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
                msg = '%s, %i drives, %f seconds\n' % (str(s.shape), _TOTAL_DRIVES, _RUNTIME)
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
        logging.info('Starting simulation from setup file %s:%s' % (str(args['setup']), '\n\n\t' + '\n\t'.join(str(key) + ': ' +
                     str(simulation_setup[key]) for key in simulation_setup.keys()) + '\n'))
    # Use default values
    else:
        simulation_setup = {}
    
    # Start simulation
    main(**simulation_setup)
    
    
    
