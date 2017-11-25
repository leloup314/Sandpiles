"""
Simulation of the Ising model
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
import itertools


########################################################################################################################
#
# Ising model with brute force approach
#
########################################################################################################################



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


def energy_roll(l, j):
    """
    Calculates the energy on a lattice
    :param np.array lattice: numpy array with spins
    :returns np.array: energies on the lattice points
    """

    d = len(l.shape)

    return np.sum(-j * l / 2 * np.sum(np.roll(l, shift, axis) for shift in [-1, 1] for axis in range(d)))


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
def exponent(sp_config, j, kb, t):
    """

    :param sp_config: lattice with possible spin config
    :param j: float prefactor of energy in Hamiltonian
    :param kb: float Boltzman constant
    :param t: float temperature
    :return: float nth partial sum of normalization factor (partition function Z) for calculation of probability
    """

    return np.exp(-1.0 * l_hamiltonian(sp_config, j)/(kb * t))


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

        # n_th normalization factor
        n_norm = 0

        # n_th probability
        n_prob = 0

        # n_th expectation value
        n_a = 0

        # Init random lattice
        l = rand_lattice(dim, dim, spins)

        # All spin configs
        all_sp = possible_spin_configs(dim, dim, spins)

        print all_sp.shape

        for s in all_sp:

            tmp_exp = exponent(s, j, kb, t)

            n_norm += tmp_exp
            n_prob += tmp_exp * abs_magnetisation_sp(s)

            n_a = n_prob/n_norm

        res.append(n_a)

    return res


def ising_test_N(N):

    spins = (-1, 1)
    dim = N
    kb = 1
    j = 1
    t = 2

    n_norm = 0
    n_prob = 0

    res = [[], []]

    # lattice
    l = rand_lattice(dim, dim, spins)

    # All spin configs
    all_sp = possible_spin_configs(dim, dim, spins)

    for i, s in enumerate(all_sp):
        tmp_exp = exponent(s, j, kb, t)

        n_norm += tmp_exp
        n_prob += tmp_exp * abs_magnetisation_sp(s)

        if i % 200 == 0:
            res[0].append(i)
            res[1].append(n_norm)

    return res
