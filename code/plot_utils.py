""" Classes and functions related to plotting the sandpile simulation"""

import numpy as np
import matplotlib.pyplot as plt  # Plotting
import matplotlib.patches as mp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages  # Plotting

try:
    import pyqtgraph as pg  # Live plotting
    pg_flag = False
except ImportError:
    pg_flag = True
    pass


def get_2d_sandboxSlice(sandbox):
    """
    Returns 2-dim sub-array of sandbox for plotting purposes if dimension is larger than 2.

    :param sandbox: np.array of sandbox
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

def get_slopeBox(sandbox):
    """
    Returns np.array with BTW compatible slopes
    instead of number of grains at each position.

    :param sandbox: np.array of custom model sandbox
    :return: n.array of BTW slopes
    """

    slopebox = np.zeros_like(sandbox);

    # Loop through all axes to sum up slopes in all directions
    for i in xrange(sandbox.ndim):

        # Shift i-th axis about 1
        sShift = np.roll(sandbox, 1, axis=i)

        slopebox += (sShift - sandbox)

    return slopebox


def plot_sandbox(s, total_drops, site=None, discrete=True, output_pdf=None):
    """
    Plots the configuration of s in a 2D heatmap
    
    :param s: np.array of sandbox
    :param total_drops: int of total amount of grains that were dropped
    :param site: tuple of coordinates of site on which the sand was dropped or None; if None, drops were random
    :param output_pdf: str of output_pdf file or None
    """
    
    title = '%s sandbox with %i ' % (str(s.shape), total_drops)
    title += 'randomly dropped grains' if site is None else 'grains dropped at %s' % str(site)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

    if s.ndim > 2:
        s = get_2d_sandboxSlice(s)

    if discrete:
        cmap = plt.get_cmap('jet', np.max(s)-np.min(s)+2)  # discrete colormap
    else:
        cmap = plt.get_cmap('jet')
    img = plt.imshow(s, cmap=cmap, vmin=np.min(s) - 0.5, vmax=np.max(s) + 1 + 0.5)  # make image with colormap BlueGreenRed
    plt.colorbar(img, ticks=np.arange(np.min(s), np.max(s)+2))  # add colorbar
    plt.show()
    
    if output_pdf is not None:
        with PdfPages(output_pdf, keep_empty=False) as out:
            out.savefig(plt.figure())
            

def plot_hist(data, name, binning=True, title=None):
    """
    Histogramms data and bin centers or all data. Just for quick check.
    
    :param data: np.array of data to histogram
    :param binning: bool whether to make histogram or plot every data point without binning
    :param title: str of title
    """
    
    title = name if title is None else title
    plt.title('Avalanche %s' % title)
    plt.xlabel('%s' % name)
    plt.ylabel('%s frequency' % name)
    data_unique, data_count = np.unique(data, return_counts=True)
    if binning:
        counts, bin_edges, _ = plt.hist(data, label='%s histogram' % name)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(bin_centers, counts,'ro', label='Bin centers')
    else:
        data_unique, data_count = np.unique(data, return_counts=True)
        plt.plot(data_unique, data_count, label='%s data' % name)
        plt.grid()
    plt.loglog()
    plt.legend()
    plt.show()

    
def plot_avalanches2d(avalanches):
    """
    Plots all 2d avalanches in avalanches iteratively. Each plot needs to be closed to show the next one.
    
    :param avalanches: np.array of avalanches
    """
    for i in range(avalanches.shape[0]):
        plt.imshow(avalanches[i], cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.title('Avalanche formation on %i x %i sandbox' % (avalanches.shape[1], avalanches.shape[2]))
        plt.xlabel('x')
        plt.ylabel('y')
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, avalanches.shape[1], 2), minor=True)
        ax.set_yticks(np.arange(-.5, avalanches.shape[2], 2), minor=True)
        ax.grid(which='minor', color='k', ls='--', lw=0.25)
        avalanche_patch = mp.Patch(color='k', label='Avalanche sites')
        plt.legend(handles=[avalanche_patch])
        plt.show()
        
        
def plot_avalanches3d(avalanches):
    """
    Plots all 3d avalanches in avalanches iteratively. Each plot needs to be closed to show the next one.
    
    :param avalanches: np.array of avalanches
    """
    for i in range(avalanches.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        c_a = avalanches[i]
        p = np.column_stack(np.where(c_a))
        x, y, z = p[:, 0], p[:, 1], p[:,2]
        ax.set_xlim(0, c_a.shape[0])
        ax.set_ylim(0, c_a.shape[1])
        ax.set_zlim(0, c_a.shape[2])
        ax.scatter(x,y,z, label='Avalanche sites', marker='s', c='k', edgecolors='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Avalanche formation on %i x %i x %i sandbox' % c_a.shape)
        ax.legend(loc='upper right', bbox_to_anchor=(1,1.1))
        plt.show()
    

# Classes depend on pyqtgraph
if not pg_flag:
    
    
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
        Subclass of pyqtgraph.GraphicsWindow to plot evolution of slopes in sandbox.
        Simulation continues even after closing the plotting window.
        
        :param s: np.array of sandbox
        :param model: str of model, either 'btw' or 'custom'
        """
        
        def __init__(self, s, model, critical_slope=None, title=None, parent=None, **kwargs):
            super(SimulationPlotter, self).__init__(parent=parent, **kwargs)
            
            # Set window title
            self.setWindowTitle(title)
            
            # Store string of model
            self.model = model
            
            # Strings of available models
            self.btw = 'btw'
            self.custom = 'custom'
            
            # Store critical slope if BTW model, else None 
            self.critical_slope = critical_slope
            
            # Creat plot item
            self.plot = self.addPlot()

            # Make bar plot for 1 dim
            if s.ndim == 1:
                # BTW model
                if self.model == self.btw:
                    self.img = pg.BarGraphItem(x=np.arange(s.shape[0]), height=self.critical_slope, width=0.5)
                # Custom model
                else:
                    self.img = pg.BarGraphItem(x=np.arange(s.shape[0]), width=0.5)
                
            # Make image for 2 dim
            elif s.ndim == 2:
                self.img = pg.ImageItem()
                # make colormap
                stops = np.linspace(0, 1, 5)
                colors = np.array([[0.0, 0.0, 0.5, 1.0], [0.0, 0.5, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.55, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
                cm = pg.ColorMap(stops, colors)
                self.img.setLookupTable(cm.getLookupTable())
                # BTW model
                if self.model == self.btw:
                    tick_labels = [str(i) for i in range(self.critical_slope + 1)]
                # Custom model
                else:
                    tick_labels = ['low'] + [' ' for _ in range(len(stops) - 2)] + ['high']               
                # Add color bar
                cb = ColorBar(cm, self.width()*0.025, self.height()*0.9, tick_labels=tick_labels)
                self.addItem(cb)
            # Add image to plot
            self.plot.addItem(self.img)
            
        def setData(self, data):
            if data.ndim == 1:
                self.img.setOpts(height=data)
            elif data.ndim == 2:
                # BTW model
                if self.model == self.btw:
                    self.img.setImage(data, levels=(0, self.critical_slope), autoDownsample=True)
                # Custom model
                else:
                    self.img.setImage(data, autoDownsample=True)
