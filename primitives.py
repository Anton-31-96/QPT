import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as la
import sys


def hs(A, B):
    """ Hilbert-Schmidt distance between two matrices """
    dist = np.sqrt(abs(np.trace((A - B) * (A - B)))) / np.sqrt(2)
    if dist < 1e-15:
        return 0
    else:
        return dist


def trace(A, B):
    """ Trace distance between two matrices """
    dist = abs(np.trace(la.sqrtm((A - B) @ (A - B)))) / 2
    if dist < 1e-15:
        return 0
    else:
        return dist


def infidelity(A, B):
    """ Infidelity between two matrices """
    dist = 1 - np.abs(np.trace(la.sqrtm(la.sqrtm(A) @ B @ la.sqrtm(A))) ** 2)
    if dist < 1e-15:
        return 0
    else:
        return dist


def product(A, B):
    """ Hermitian inner product in matrix space """
    return np.trace(A @ np.conj(B.T), dtype=np.complex128)


def density(psi):
    """
    Construct a density matrix of a pure state
    Input:
        psi = [x1, x2]
    """
    return np.matrix(psi, dtype=np.complex128).T @ np.conj(np.matrix(psi, dtype=np.complex128))


def left_inv(A):
    return la.inv(A.T @ A) @ A.T


def is_physical(rho, tol=1e-12):
    is_physical_flag = True
    if np.abs(np.trace(rho) - 1) > tol:
        print('Trace = ', np.trace(rho), file=sys.stderr)
        is_physical_flag = False
    if np.min(np.real(la.eig(rho)[0])) < -tol:
        print('Eigenvalues: ', la.eig(rho)[0], file=sys.stderr)
        is_physical_flag = False
    if hs(rho, np.conj(rho.T)) > tol:
        print('Non-hermitian', file=sys.stderr)
        is_physical_flag = False
    return is_physical_flag


# Pauli basis

SIGMA_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# POVMs

MUB_POVM = [[density([1, 0])],
            [density([1/np.sqrt(2), 1/np.sqrt(2)])],
            [density([1/np.sqrt(2), 1j/np.sqrt(2)])]]


class Basis:
    def __init__(self, dim=1):
        self.basis = self.generate_pauli(dim=dim)
        self.dim = dim

    def matrix_to_vector(self, matrix, bloch=True):
        """ Decomposition of a matrix in the preferred basis """
        begin_idx = 1 if bloch else 0
        denominator = 1 if bloch else 2 ** self.dim
        vector = np.array([np.real(product(basis_element, matrix)) for basis_element in self.basis[begin_idx:]])
        return vector / denominator

    def vector_to_matrix(self, vector, bloch=True):
        """ Restore a matrix from its decomposition """
        if bloch:
            matrix = np.matrix(np.eye(2 ** self.dim), dtype=np.complex128)
            for i in range(4 ** self.dim - 1):
                matrix += self.basis[i + 1] * vector[i]
            matrix /= 2 ** self.dim
        else:
            matrix = np.matrix(np.zeros((2 ** self.dim, 2 ** self.dim)), dtype=np.complex128)
            for i in range(4 ** self.dim):
                matrix += self.basis[i] * vector[i]
        return matrix

    @staticmethod
    def generate_pauli(dim):
        basis = pauli_0 = [SIGMA_I, SIGMA_X, SIGMA_Y, SIGMA_Z]
        for _ in range(dim - 1):
            basis = np.kron(basis, pauli_0)
        return basis


def state_to_proba(set_of_POVMs):
    """
    Input:
        list of POVMs
    Example:
        [[density([1,0])],
        [density([1/np.sqrt(2),1/np.sqrt(2)])],
        [density([1/np.sqrt(2),-1j/np.sqrt(2)])]]
    Output:
        Operator B in p=Bs
    """
    POVMs = []
    for subset in set_of_POVMs:
        vectorized_subset = np.zeros((len(subset), 4))
        for i, POVM in enumerate(subset):
            vectorized_subset[i] = Basis().matrix_to_vector(POVM, bloch=False)
        POVMs.append(vectorized_subset)
    POVMs.append(Basis().matrix_to_vector(np.eye(2), bloch=False)) # to make set of POVMs IC
    return 2 * np.vstack(POVMs)


def proba_to_state(set_of_POVMs):
    """ Obtain an operator A in s=Ap """
    A = left_inv(state_to_proba(set_of_POVMs))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i, j]) < 1e-15:
                A[i, j] = 0
    return A


def get_delta(dist_list, conf_level):
    """
    Compute delta for the preferred confidence level
    Input:
        Sorted list of distances
    """
    n = len(dist_list) - 1
    return dist_list[int(n * conf_level)]
