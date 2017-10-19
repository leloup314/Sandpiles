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

    def _init_lattice(self):

        # lattice was provided
        if self.lattice:
            return
        else:
            self.lattice = np.random.choice(self.spins, size=(self.n, self.n))

    def hamiltonian(self):

        for i in range(self.n):
            for j in range(self.n):

                self.energy += self.lattice[i][j]*(self.lattice[i % self.n - 1][j] + self.lattice[i][j % self.n - 1])

    def get_lattice(self):
        return self.lattice

    def show_lattice(self):
        print self.lattice
        #plt.imshow(self.lattice)


i_0 = IsingLattice(3)

i_0.show_lattice()
i_0.hamiltonian()