Simulation of the dynamics of sandpiles via cellular automata
=============================================================

Implementation of the Bak-Tank-Wiesenfeld approach of cellular automata for sandpile dynamics.
Additional implementation of a customized model. The simulations store the characteristic statistics of the sandpile dynamics such as

- Avalanche time (equivalent to number of iterations/recursions)
- Avalanche size (equivalent to area)
- Total amount of dropped sand (including redistributions) during avalanches
- Linear size e.g. maximum distance between two sites of an avalanche

Required packages
*****************

The following packages are required for the simulation:

.. code-block:: bash

   numpy numba scipy matplotlib

For plotting the (live) evolution of the sandpiles, these additional packages are required:

.. code-block:: bash

   pyqtgraph

Simulation of 100000 dropped grains of sand into the center of a 200 x 200 sandbox
**********************************************************************************

.. image:: _static/f1.png

