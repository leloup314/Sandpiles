"""Tutorial 1; Ising model"""

import numpy as np
import matplotlib.pyplot as plt


class IsingLattice:

    def __init__(self, n, j=None, lattice=None, spins=None):

        self.n = n  # store dimension of n x n lattice
        self.lattice = lattice # store lattice if given, else None
        self.spins = spins if spins is not None else [1, -1]  # default spins are +-1
        self.j = j if j is not None else 1  # default j is 1
        self.energy = 0

        self._init_lattice()
        self._hamiltonian()

    def _init_lattice(self):

        # lattice was provided
        if self.lattice:
            return
        else:
            self.lattice = np.random.choice(self.spins, size=(self.n, self.n))

    def _hamiltonian(self):
        """Calculate lattice hamiltonian"""
        i = 0
        j = 0
        for i in range(self.n):
            for j in range(self.n):
                # Walk through lattice; avoid double counting by only going right and down.
                # Nearest neighbour of border (i = self.n) is i=0.
                # FIXME: This might still be incorrect
                self.energy += self.lattice[i][j]*(self.lattice[(i+1) % self.n][j] + self.lattice[(i-1) % self.n][j])

        # Multiply by self.j
        self.energy *= -self.j

    def rand_lattice(self):
        """Creates new random lattice"""
        self.lattice = np.random.choice(self.spins, size=(self.n, self.n))

    def get_lattice(self):
        """Return lattice"""
        return self.lattice

    def show_lattice(self):
        """Show lattice config using matplotlib"""
        plt.imshow(self.lattice, interpolation='none', cmap='gray')
        plt.colorbar()
        plt.title('Configuration of %i x %i Ising lattice' % (self.n, self.n))
        plt.show()

ising = IsingLattice(10)
print ising.energy
ising.show_lattice()