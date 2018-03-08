""" Saving/loading results functions """

import os
import numpy as np
            
def save_array(array, out_file):
    """
    Function to save a numpy array. Avoids overwriting existing arrays.
    
    :param array: np.array to save
    :param out_file: str of absolute path to output file
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
    
    
def save_simulation(s, sim, model, total_drives=None, site=None, critical_slope=None, out_file=None):
    """
    Function to save result array of simulation.
    
    :param s: np.array of sandbox of simulation
    :param sim: np.array of results of simulation
    :param model: str either 'btw' or 'custom'
    :param total_drives: int of total drives
    :param site: tuple of coordinates of site to which slope was added; if None, random
    :param out_file: str of absolute path to output file
    """
    
    # Find output path if not given
    if out_file is None:
        
        # Path where simulations are stored
        sim_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../simulations/statistics'))
        
        # If path does not exist, make it
        if not os.path.exists(sim_path):
            os.mkdir(sim_path)
        
        # Total drives
        td = str(total_drives) if total_drives is not None else '?'
        
        cs = '_crit_slope_%s' % str(critical_slope) if critical_slope is not None else ''
        
        # Check whether random drops or on site
        if site is None:
            sm = '%s_simulation_%s_drives_random%s_%s.npy' % ('x'.join([str(dim) for dim in s.shape]), td, cs, model)
        else:
            sm = '%s_simulation_%s_drives_at_%s%s_%s.npy' % ('x'.join([str(dim) for dim in s.shape]), td, '_'.join(str(c) for c in site), cs, model)
        
        # Set new path
        out_file = os.path.join(sim_path, sm)
    
    # Save array to out_file
    save_array(sim, out_file)


def save_sandbox(s, model, total_drives=None, site=None, critical_slope=None, out_file=None):
    """
    Function to save critical sandbox after simulation.
    
    :param s: np.array of sandbox of simulation
    :param model: str either 'btw' or 'custom'
    :param total_drives: int of total dropped grains of sand
    :param site: tuple of coordinates of site to which slope was added; if None, random
    :param out_file: str of absolute path to output file
    """
    
    # Find output path if not given
    if out_file is None:
        
        # Path where sandboxes are stored
        sandbox_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../simulations/sandboxes'))
        
        # If path does not exist, make it
        if not os.path.exists(sandbox_path):
            os.mkdir(sandbox_path)
        
        # Total drives
        td = str(total_drives) if total_drives is not None else '?'
        
        cs = '_crit_slope_%s' % str(critical_slope) if critical_slope is not None else ''
        
        # Check whether random drops or on site
        if site is None:
            sb = '%s_sandbox_%s_drives_random%s_%s.npy' % ('x'.join([str(dim) for dim in s.shape]), td, cs, model)
        else:
            sb = '%s_sandbox_%s_drives_at_%s%s_%s.npy' % ('x'.join([str(dim) for dim in s.shape]), td, '_'.join(str(c) for c in site), cs, model)
        
        # Set new path
        out_file = os.path.join(sandbox_path, sb)
    
    # Save array to out_file
    save_array(s, out_file)
