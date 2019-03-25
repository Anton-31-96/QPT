# -*- coding: utf-8 -*-

"""
Quatum State and Process Tomography routines 
"""

import numpy as np
import itertools as it
import numpy.linalg as LA
from scipy.linalg import sqrtm
from numpy.linalg import matrix_power

import channels as ch


"""
QIT block (create different file later)
"""

# Initialize Pauli matrices
X_PAULI = np.array([[0, 1], [1, 0]], dtype = complex)
Y_PAULI = np.array([[0, -1j], [1j, 0]], dtype = complex)
Z_PAULI = np.array([[1, 0], [0, -1]], dtype = complex)
I_PAULI = np.eye(2, dtype = complex) 
PAULI_SET = np.array([I_PAULI, X_PAULI, Y_PAULI, Z_PAULI])


def pauli_dot(vector):
    """
    Scalar product of Pauli matrices and vector
    """ 

    return X_PAULI * vector[0] + Y_PAULI * vector[1] + Z_PAULI * vector[2]



def prepare_states(case = "tetr"):
    """
    Returns input states for two following cases: 
    1) "tetr" - states located in vertices of tetrahedron
    2) "axis" - states are Z, -Z, X, -X, Y, -Y 
    """
    
    assert(case == 'tetr' or case == 'axis'), "Wrong value of case. Use \
        'tetr' or 'axis' instead."

    if case == "tetr":
        a0 = np.array([1,1,1])/np.sqrt(3)
        a1 = np.array([1,-1,-1])/np.sqrt(3)
        a2 = np.array([-1,1,-1])/np.sqrt(3)
        a3 = np.array([-1,-1,1])/np.sqrt(3)
        a = np.array([a0,a1,a2,a3])
        
    elif case == "axis":
        a0 = np.array([1,0,0])   # X
        a1 = np.array([-1,0,0])  # -X
        a2 = np.array([0,1,0])   # Y
        a3 = np.array([0,-1,0])  # Y
        a4 = np.array([0,0,1])   # Z
        a5 = np.array([0,0,-1])  # -Z
        a = np.array([a0,a1,a2,a3,a4,a5])

    return a



def bloch_to_dmat(vector):
    """
    returns density matrix corresponding to vector on Bloch sphere (2x2 case)
    """
    return np.array(0.5*(np.eye(2) + pauli_dot(vector)))



def oper_vec_to_mat(vec):
    """
    takes measurement operator parameterized as expanded Bloch vector (R4)
    and returns its matrix form 
    """

    oper = np.zeros((2,2), dtype = complex)
    for i in range(4):
        oper += PAULI_SET[i] * vec[i]
    
    return oper



def dmat_to_bloch(dmat, case = 'R3'):
    """
    represents density matrix as a Bloch's vector
    
    case: 'R3' - usual Bloch sphere
          'R4' - Bloch vector with additional coefficient a0 = 1 
    """

    assert( case == 'R3' or case == 'R4'), \
        'Wrong case value. Use "R3" or "R4" instead'
        
    a1 = np.trace( X_PAULI @ dmat )
    a2 = np.trace( Y_PAULI @ dmat )
    a3 = np.trace( Z_PAULI @ dmat )
    
    if case == 'R3':
        dmat_vec = np.array([a1,a2,a3])
    elif case == 'R4':
        dmat_vec = np.array([1,a1,a2,a3])
    
    return np.real(dmat_vec)



# Metrics functions

def hs_dist(rho_A, rho_B):
    """
    Hilbert-Schmidt distance between two density matrices
    """
    return np.real(np.trace(matrix_power(rho_A - rho_B, 2)))

def tr_dist(rho_A, rho_B):
    """
    Trace distance between two density matrices
    """
    M = np.matrix(rho_A - rho_B)
    return np.real(0.5 * np.trace(sqrtm(M.H@M)))

def if_dist(rho_A, rho_B):
    """
    Infidelity distance between two density matrices
    """
    sq = sqrtm(rho_A)
    mult = sqrtm(sq @ rho_B @ sq)
    return np.real((1-np.trace(matrix_power(mult, 2))))



"""
Tomography block
"""


def protocol_QST(case = 'tetr', n = 1):
    """
    returns matrix of state measurement protocol, where
       row - measurement operator
       column - operator's coefficients

    case: 'tetr', 'axis'
    n: number of qubits (int)
    """

    assert(case == 'tetr' or case == 'axis'), \
        "Wrong value of case. Use 'tetr' or 'axis' instead."
    
    if case == 'tetr':
        sq3 = 1/np.sqrt(3)
        a0 = np.array([1,sq3,sq3,sq3])
        a1 = np.array([1,sq3,-sq3,-sq3])
        a2 = np.array([1,-sq3,sq3,-sq3])
        a3 = np.array([1,-sq3,-sq3,sq3])
        A = np.array([a0,a1,a2,a3])
        S = A
        for i in range(n-1):
            S = np.kron(S,A)
        S = S/S.shape[0]
    
    elif case == 'axis':
        x = np.array([1,1,0,0])
        y = np.array([1,0,1,0])
        z = np.array([1,0,0,1])
        x_ = np.array([1,-1,0,0])
        y_ = np.array([1,0,-1,0])
        z_ = np.array([1,0,0,-1])
        A = np.array([x,x_,y,y_,z,z_])
        S = A
        for i in range(n-1):
            S = np.kron(S, A)
        S = S/S.shape[0]
             
    return np.array(np.matrix(S))



def protocol_QPT(prepare_case = 'tetr', measure_case = 'tetr'):
    """
    returns matrix corresponding to the protocol of quantum process tomography
    
    prepare_case: 'tetr', 'axis' - preparation basis
    measure_case: 'tetr', 'axis' - measurement basis
    """
    
    # Check input 
    assert(prepare_case == 'tetr' or prepare_case == 'axis'), \
        "Wrong value of prepare_case. Use 'tetr' or 'axis' instead."
    
    assert(measure_case == 'tetr' or measure_case == 'axis'), \
        "Wrong value of measure_case. Use 'tetr' or 'axis' instead."
    
    input_states_vec = prepare_states(prepare_case)
    
    input_states = []
    for bloch_vec in input_states_vec:
        input_states.append(  bloch_to_dmat(bloch_vec)  )
        
    E_vec = protocol_QST(measure_case)
    E = []
    for e in E_vec:
        E.append(oper_vec_to_mat(e))
        
    A = []
    for r,e in list(it.product(input_states, E)):
        a = np.kron(r, e.T).flatten()
        A.append(a)
            
    return np.array(A)



def CP_proj(C):
    """
    returns projection of map C on the set of completely positive maps 
    """
    C = C.reshape(4,4)
    w, v = LA.eig(C)
    v = np.matrix(v)
    for i in range(w.size):
        if w[i] < 0:
            w[i] = 0
    return (v * np.diag(w) * v.H).reshape(-1,1)



def TP_proj(C):
    """
    returns projection of map C on the set of trace preserving maps
    """
    k = 2 # dimension of system
    basis = np.eye(k)
    b = np.eye(k).reshape(k**2, 1)
    
    # create partial trace operator 
    M = np.zeros((4,16))
    for i in range(k):
        j = np.kron(np.eye(k), basis[i])
        M += np.kron(j,j)
    M = np.matrix(M)
    C = C.reshape(-1, 1)
    
    C_tp = C - 1./k*M.H*M*C + 1./k*M.H*b
    
    return C_tp #.reshape(k**2, k**2)



def CPTP_proj(C):
    """
    projection of map C on the set of Completely Positive and Trace Preserving maps
    """
    shape = C.reshape(-1,1).shape
    x = [np.matrix(C.reshape(-1,1))]
    p = [np.matrix(np.zeros(shape))]
    q = [np.matrix(np.zeros(shape))]
    y = [np.matrix(np.zeros(shape))]
    for i in range(10000):
        y.append( np.matrix(TP_proj(x[i]+p[i])) )
        p.append( x[i]+p[i]-y[i+1] )
        x.append( np.matrix(CP_proj(y[i+1] + q[i])) )
        q.append( y[i+1]+q[i]-x[i+1] )
        
        VOB = LA.norm(p[i] - p[i+1])**2 + LA.norm(q[i] - q[i+1])**2 \
        + np.abs(2*p[i].H @ (x[i+1] - x[i])) + np.abs(2*p[i].H @ (y[i+1]-y[i]))
        VOB = np.real(VOB)
        if VOB < 1e-4:
#             print(VOB)
            break

    return CP_proj(x[-1]).reshape(4,4)



def measurements(channel, prepare_case = 'tetr', measure_case = 'tetr', N = 1000):
    """
    returns frequency probabilies for measurements of state rho 
    according to protocol defined by case parameter
    
    channel - function of channel
    prepare_case: 'tetr', 'axis'
    measure_case: 'tetr', 'axis' 
    N - number of measurements (int)
    """
#     if (noise > 1) or (noise < 0):
#         print("incorrect level of noise")
#         return False

    
    # """ Вот сейчас начинается дикий костыль. 
    # Требуется заимплементить параметризованное действие канала"""
    
    input_states_vec = prepare_states(prepare_case)
    
    input_states = []
    for bloch_vec in input_states_vec:
        input_states.append(  ch.bloch_to_dmat(bloch_vec)  )
    
    output_states_vec = []
    for rho in input_states:
        rho_out = channel(rho)
        output_states_vec.append(  dmat_to_bloch(rho_out, case = 'R4')  )
#     print(output_states_vec)
    
    E = protocol_QST(measure_case)
    
    p = np.zeros((E.shape[0], len(input_states_vec)))
    n = np.zeros((E.shape[0], len(input_states_vec)))
    beta = 0.5
    
    for j in range(len(input_states_vec)):
        p[:,j] = E @ output_states_vec[j]
    
        pos_outcomes = np.arange(E.shape[0])
        outcomes = np.random.choice(pos_outcomes, size = N, p = p[:,j])
        
        
        for i in range(E.shape[0]):
            ind = np.where(outcomes == pos_outcomes[i])[0]
            n[i,j] = (len(ind) + beta) / (N + 2 * beta)
    
#     if noise > 0:
#         # level of noise
#         noise_pos = np.random.choice([0,1], p = [1-noise, noise], size=N)
# #         replace = np.random.choice(range(N), replace=False, size= int(N*noise) )
#         outcomes[np.where(noise_pos == 1)] = np.random.choice(pos_outcomes, size = np.sum(noise_pos))

        
    return n



def current_probability(C, protocol, prepare_case = 'tetr', measure_case = 'tetr'):
    """
    Compute probability of experimental outcome for a given channel C

    C: map in Choi matrix form
    protocol: matrix of protocol QPT
    prepare_case: 'tetr', 'axis'
    measure_case: 'tetr', 'axis'
    """

    # Check conditions
    assert(prepare_case == 'tetr' or prepare_case == 'axis'),\
        "Wrong value of prepare_case. Use 'tetr' or 'axis' instead"
    assert(measure_case == 'tetr' or measure_case == 'axis'),\
        "Wrong value of measure_case. Use 'tetr' or 'axis' instead"

    # A = protocol_QPT(prepare_case=prepare_case, measure_case=measure_case)

    prob = np.real(protocol @ C.reshape(-1,1))

    return prob


def cost(C, frequencies, current_prob, prepare_case = 'tetr', measure_case = 'tetr'):
    """
    compute cost function (corresponds to loglikelyhood)
    
    С - map in Choi matrix form
    frequencies - frequency probabilities obtained from experiment
    prepare_case: 'tetr', 'axis'
    measure_case: 'tetr', 'axis'
    """
    
    # A = protocol_QPT(prepare_case=prepare_case, measure_case=measure_case)

    # prob = np.real(A @ C.reshape(-1,1))
    
    cost_val = - frequencies.T @ np.log(current_prob)
    
    return np.real(cost_val)



def gradient(C, protocol, frequencies, current_prob, prepare_case = 'tetr', measure_case = 'tetr'):
    """
    computes gradient of the LLH function
    
    C - channel as Choi matrix
    protocol: matrix of protocol QPT
    frequencies - frequency probabilities obtained from experiment
    prepare_case: 'tetr', 'axis'
    measure_case: 'tetr', 'axis'
    """
    
    # A = protocol_QPT(prepare_case=prepare_case, measure_case=measure_case)
    # A = np.matrix(A)

    # prob = np.real(A @ C.reshape(-1,1))

    # cost_val = - frequencies.T @ np.log(prob)
    
    grad = - protocol.H @ ( frequencies / current_prob)
            
    return grad.reshape(4,4)


def grad_descent(frequencies, protocol, C_0, prepare_case = 'tetr', measure_case = 'tetr'):
    """
    performs gradient descent optimization
    
    method returns history of algorithm's steps
    """

#     C_0 = np.array([[1,0,0,1],
#                     [0,0,0,0],
#                     [0,0,0,0],
#                     [1,0,0,1]]) # identity channel
    C = [np.matrix(C_0)]
    
    mu = 3./(2*4)
    gamma = 0.3

    current_prob = current_probability( C_0,
                                            protocol = protocol,
                                            prepare_case = prepare_case,
                                            measure_case = measure_case)

    grad_C_0 = grad_C = gradient( C_0,
                           protocol = protocol,
                           frequencies = frequencies,
                           current_prob = current_prob,
                           prepare_case = prepare_case, 
                           measure_case = measure_case)
    
    error = 1e-10 # eror of algorythm
    for i in range(5000):
        alpha = 2.
        current_prob = current_probability( C[i],
                                            protocol = protocol,
                                            prepare_case = prepare_case,
                                            measure_case = measure_case)
        grad_C = gradient( C[i],
                           protocol = protocol,
                           frequencies = frequencies,
                           current_prob = current_prob,
                           prepare_case = prepare_case, 
                           measure_case = measure_case)

        D = CPTP_proj(  C[i] - grad_C / mu ) - C[i]
        
        C_current_cost = cost(C=C[i], 
                              frequencies = frequencies,
                              current_prob = current_prob,
                              prepare_case = prepare_case, 
                              measure_case = measure_case)
#         print('current_cost = ', C_current_cost)
        
        B = np.real(C_current_cost + gamma * alpha * np.dot( D.reshape(-1,1).T, grad_C.reshape(-1,1) ))
        
        next_prob = current_probability(C[i]+alpha*D,
                                        protocol = protocol,
                                        prepare_case = prepare_case,
                                        measure_case = measure_case)

        C_next_cost = cost( C=(C[i] + alpha * D), 
                            frequencies=frequencies,
                            current_prob = next_prob,
                            prepare_case=prepare_case, 
                            measure_case=measure_case)
#         print('next cost = ', C_next_cost)
        
        while C_next_cost > B:
            alpha = 0.5*alpha
            B = np.real( C_current_cost + gamma * alpha * np.dot( D.reshape(-1,1).T, C[i].reshape(-1,1) ) )
            
            next_prob = current_probability(C[i] + alpha * D,
                                            protocol = protocol,
                                            prepare_case = prepare_case,
                                            measure_case = measure_case)



            C_next_cost = cost( C=(C[i]+alpha*D), 
                                frequencies=frequencies,
                                current_prob = next_prob,
                                prepare_case=prepare_case, 
                                measure_case=measure_case)
            
#         print('alpha = ', alpha)
        C_new = CPTP_proj(C[i] + alpha * D) 
        C.append(C_new)
#         print(i)
#         print(C_current_cost)
#         print 'alpha:', alpha
#         print 'C_current_cost:', C_current_cost
#         print 'C_next_cost:', C_next_cost
#         print (np.abs(C_current_cost - C_next_cost))

        # Stopping criteria
        # crit = LA.norm(grad_C - grad_C_0)/LA.norm(grad_C_0)
        # print(crit)
        # if (crit < error) & (i > 5):
        # print('i = ',i)
        if np.abs(C_current_cost - C_next_cost) < error:
            print('End of algorithm. next_cost = ', C_next_cost)
            return C
    return C
