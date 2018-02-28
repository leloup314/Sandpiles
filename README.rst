Simulation of the dynamics of sandpiles via cellular automata
=============================================================

Implementation of the Bak-Tank-Wiesenfeld approach of cellular automata for sandpile dynamics.
Additional implementation of a customized model. The simulations store the characteristic statistics of the sandpile dynamics such as

- Avalanche duration (equivalent to number of iterations/recursions)
- Avalanche size (equivalent to amount of drops and redistributions during avalanche)
- Avalanche area (equivalent to number of lattice sites participating in avalanche)
- Avalanche's linear size (equivalent to maximum distance in between two sites in an avalanche)

In general (depending on memory recources etc.), the implemetations work for arbitrary dimensions. 

Required packages
*****************

The following packages are required for the simulation:

.. code-block:: bash

   numpy numba scipy matplotlib

For plotting the (live) evolution of the sandpiles, these additional packages are required:

.. code-block:: bash

   pyqtgraph

Examples
********

.. image:: _static/f1.png
   :width: 75 %
   :align: center

.. image:: _static/f2.png
   :width: 75 %
   :align: center
