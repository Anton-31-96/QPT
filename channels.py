# -*- coding: utf-8 -*-

import numpy as np
import itertools as it

# Initialize Pauli matrices
X_PAULI = np.array([[0, 1], [1, 0]], dtype = complex)
Y_PAULI = np.array([[0, -1j], [1j, 0]], dtype = complex)
Z_PAULI = np.array([[1, 0], [0, -1]], dtype = complex)
I_PAULI = np.eye(2, dtype = complex) 
PAULI_SET = [I_PAULI, X_PAULI, Y_PAULI, Z_PAULI]


"""
Fucntions on channels
"""

def choi_to_mat(channel):
    """
    Represent channel in Pauli matrices basis 
    """
    
    # Create list of basis elements
    grid = [np.kron(x,y) for x,y in list(it.product(PAULI_SET, PAULI_SET))]
    
    coefficients = []
    for element in grid:
        coefficients.append(np.trace( element.T @ channel ))
    
    coefficients = np.array(coefficients)    

    return coefficients / 4



def channel_to_choi(channel):
    """
    Returns choi matrix of channel
    - only for qubit
    """
    c00 = np.array([[1,0],[0,0]])
    c01 = np.array([[0,1],[0,0]])
    c10 = np.array([[0,0],[1,0]])
    c11 = np.array([[0,0],[0,1]])
    
    c00 = channel(c00)#, p)
    c01 = channel(c01)#, p)
    c10 = channel(c10)#, p)
    c11 = channel(c11)#, p)
    
    c0 = np.hstack((c00,c01))
    c1 = np.hstack((c10,c11))
    
    return np.vstack((c0,c1))



def pauli_dot(vector):
    """
    Scalar product of Pauli matrices and vector 
    """ 
    
    return X_PAULI * vector[0] + Y_PAULI * vector[1] + Z_PAULI * vector[2]



def bloch_to_dmat(vector):
    """
    returns density matrix corresponding to vector on Bloch sphere (2x2 case)
    """
    return np.array(0.5*(np.eye(2) + pauli_dot(vector)))



"""
Different channels block
"""

def depolarize_channel(dmatrix, p):
    """
    Depolarizing channel action
    """
    return (1-p)*dmatrix + p/3.*(X_PAULI@dmatrix@X_PAULI \
           + Y_PAULI@dmatrix@Y_PAULI + Z_PAULI@dmatrix@Z_PAULI)


def identical_channel(dmatrix):
    """
    Identical channel action
    """
    return dmatrix


def squeeze_channel(dmatrix, p):
    """
    Squeezing channel action
    """
    return p * dmatrix



def dephase_channel(dmatrix, p):
    """
    Dephasing channel action
    """
    return (1 - 0.5 * p) * dmatrix + 0.5 * p * Z_PAULI @ dmatrix @ Z_PAULI 



def gate_action(dmatrix, gate):
    """
    Action of gate from PAULI set on a density matrix dmatrix

    gate: 'X', 'Y', 'Z' 
    """
    assert(gate == 'X' or gate == 'Y' or gate == 'Z'), \
        "Wrong gate value. Use gate from PUAILY set (X, Y, Z) instead"

    if gate == 'X':
        return X_PAULI @ dmatrix @ X_PAULI

    elif gate == 'Y':
        return Y_PAULI @ dmatrix @ Y_PAULI
    
    elif gate == 'Z':
        return Z_PAULI @ dmatrix @ Z_PAULI
        