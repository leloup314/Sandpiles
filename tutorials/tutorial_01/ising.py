"""Tutorial 1; Ising model"""

import numpy as np
import matplotlib.pyplot as plt


class IsingLattice:

    def __init__(self, n, j=None, lattice=None, spins=None):

        self.n = n  # store dimension of n x n lattice
        self.lattice = lattice # store lattice if given, else None
        self.spins = spins if spins is not None else [1, -1]  # default spins are +-1
        self.j = j if j is not None else 1  # default j is 1
        self.t = None  # Don't include temperature yet
        self.energy = 0

        self._init_lattice()
        self.energy = self.hamiltonian(self.lattice, self.j, self.t)

    def _init_lattice(self):

        # lattice was provided
        if isinstance(self.lattice, np.ndarray):
            pass
        # lattice was provided but is not np.array
        elif not isinstance(self.lattice, np.ndarray) and self.lattice:
            self.lattice = np.array(self.lattice)
        # no lattice provided
        else:
            self.lattice = np.random.choice(self.spins, size=(self.n, self.n))

    def hamiltonian(self, lattice, j, temp):
        """Calculate lattice hamiltonian"""
        energy = 0
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                # Walk through lattice; avoid double counting by only going right and down.
                # Nearest neighbour of border (i = self.n) is i=0.
                energy += lattice[i][j]*(lattice[(i+1) % lattice.shape[0]][j] + lattice[i][(j+1) % lattice.shape[1]])
        
        # Multiply by j
        return j*energy
        
        #TODO: Use np.roll for hamiltonian
        
    def rand_lattice(self):
        """Creates new random lattice"""
        self.lattice = np.random.choice(self.spins, size=(self.n, self.n))

    def show_lattice(self):
        """Show lattice config using matplotlib"""
        plt.imshow(self.lattice, interpolation='none', cmap='gray')
        plt.colorbar()
        plt.title('Configuration of %i x %i Ising lattice' % (self.n, self.n))
        plt.show()

ising = IsingLattice(6)
ising.show_lattice()
