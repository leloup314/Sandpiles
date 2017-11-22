"""
Simulation of the Ising model
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
import itertools


def rand_lattice(lx, ly, sp):
    """
    Return a lattice of dimensions lx * ly with random spin entries from sp
    :param lx: int lattice dimension in x
    :param ly: int lattice dimension in y
    :param sp: iterable with elements from which lattice spins are drawn
    :return: np.array with random spins
    """
    return np.random.choice(sp, size=(lx, ly))


def plot_lattice(l):
    """
    Show lattice using matplotlib
    :param l: np.array or array
    """
    plt.imshow(l, interpolation='none', cmap='gray')
    plt.colorbar()
    plt.title('Configuration of %i x %i Ising lattice' % (l.shape[0], l.shape[1]))
    plt.show()


@njit  # Speed-up looping through lattice
def l_hamiltonian(l, j):
    """
    Calculates the Hamiltonian of the Ising lattice
    :param l: np.array of lattice
    :param j: float prefactor of energy in Hamiltonian
    :return: energy of lattice spin config
    """

    e = 0  # Initialize for njit
    for i in xrange(l.shape[0]):
        for k in xrange(l.shape[1]):
            e += l[i][k] * (l[(i + 1) % l.shape[0]][k] + l[i][(k + 1) % l.shape[1]])
    return j * e


#@jit
def possible_spin_configs(lx, ly, sp):
    """

    :param lx: int lattice dimension in x
    :param ly: int lattice dimension in y
    :param sp: iterable with elements from which lattice spins are drawn
    :return: np.array of all possible spin configurations of a lattice of lx * ly with spins sp
    """

    # Other attempts to get all possible combos:
    #
    # 10 loops, best of 3: 153 ms per loop for lx=ly=4
    # return [np.reshape(np.array(i), (lx, ly)) for i in itertools.product(spins, repeat=lx * ly)]
    # 100 loops, best of 3: 2.86 ms per loop for lx=ly=4
    # return np.vstack([y.flat for y in np.meshgrid(*([spins] * (lx * ly)))]).T.reshape(-1, lx, ly)

    # 100 loops, best of 3: 3.08 ms per loop for lx=ly=4
    return np.array(sp)[np.rollaxis(np.indices((len(sp),) * lx * ly), 0, lx * ly + 1).reshape(-1, lx, ly)]


#@jit
def normalization(sp_configs, j, kb, t):
    """

    :param sp_configs: np.array with all possible spin configurations
    :param j: float prefactor of energy in Hamiltonian
    :param kb: float Boltzman constant
    :param t: float temperature
    :return: float normalization factor (partition function Z) for calculation of probability
    """

    return np.sum([np.exp(-1.0 * l_hamiltonian(s_prime, j)/(kb * t)) for s_prime in sp_configs])


def probability(n, l, j, kb, t):
    """
    :param n: float normalization factor (partition function Z)
    :param l: np.array of lattice
    :param kb: float Boltzman constant
    :param t: float temperature
    :return: float probability for lattice configuration s
    """

    return 1.0/n * np.exp(-l_hamiltonian(l, j)/(kb * t))


def abs_magnetisation_sp(l):
    """
    :param l: np.array of lattice
    :return: float absolute magnetisation per spin of the given lattice
    """

    return 1.0/(l.shape[0] * l.shape[1]) * np.absolute(np.sum(l.ravel()))


def ising_test():
    """
    Test brute force ising calc for lattice with lx=ly=2,3,4,5
    :return:
    """

    spins = [-1, 1]
    NN_lattice = [2, 3, 4, 5]
    kb = 1
    j = 1
    t = 2
    res = []

    for dim in NN_lattice:

        # Init random lattice
        l = rand_lattice(dim, dim, spins)

        # All spin configs
        all_sp = possible_spin_configs(dim, dim, spins)

        # Normalization
        norm = normalization(all_sp, j, kb, t)

        # Probability
        p = probability(norm, l, j, kb, t)

        # Expectation val of abs magnetisation
        exp_m = p*abs_magnetisation_sp(l)

        res.append(exp_m)

    return res

print ising_test()