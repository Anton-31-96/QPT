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
    if len(vector) == 4:
        # In case of extended vector
        product = np.tensordot(PAULI_SET, vector, axes=(0,0))

    elif len(vector) == 3:
        # in case of usual Bloch vector
        product = np.tensordot(PAULI_SET[1:], vector, axes=(0,0))
    else:
        print("the wrong dimensionality of vector")
        return False
    return product


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
    if len(vector) == 4:
        # in case of extended Bloch vector
        dmat = 0.5 * np.array(pauli_dot(vector))

    elif len(vector) == 3:
        # in case of usual Bloch vector
        dmat = np.array(0.5*(np.eye(2) + pauli_dot(vector)))

    else:
        print('The wrong dimensionality of the vector')
        return False

    return dmat 



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


""" 
Quantum State Tomography (SQS)
"""


def protocol_QST(case = 'tetr', n = 1):
    """
    returns matrix of state measurement protocol, where
       row - measurement operator
       column - operator's coefficients

    Args:
        case: str
            'tetr', 'axis'
        n: number of qubits (int)
    Returns:
        protocol: np.matrix
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

    protocol = np.array(np.matrix(S))
             
    return protocol



def measurements_QST(protocol, rho, N = 1000, noise = 0):
    """
    returns frequency probabilies for measurements of state rho according to protocol
    E - (nd.array)
    rho - state as expanded vector on Bloch sphere
    N - number of measurements (int)
    noise - level of noise while measurements (from 0 to 1)
    """
    if (noise > 1) or (noise < 0):
        print("incorrect level of noise")
        return False
        
    p = protocol @ rho
    n = np.zeros(protocol.shape[0])
    pos_outcomes = np.arange(protocol.shape[0])
    
    outcomes = np.random.choice(pos_outcomes, size = N, p = p)
    
    if noise > 0:
        # level of noise
        noise_pos = np.random.choice([0,1], p = [1-noise, noise], size=N)
#         replace = np.random.choice(range(N), replace=False, size= int(N*noise) )
        outcomes[np.where(noise_pos == 1)] = np.random.choice(pos_outcomes, size = np.sum(noise_pos))
        
    beta = 0.5
    
    for i in range(protocol.shape[0]):
        ind = np.where(outcomes == pos_outcomes[i])
        n[i] = (len(ind[0])+beta)/(N+2*beta)
        
    return n



def gen_state(n = 1):
    """
    returns random state of n qubits
    """
    
    def rand_qubit():
        """
        returns random state of single qubit
        """
        fi = np.random.uniform(0, 2*np.pi)
        theta = np.arccos( np.random.uniform(-1., 1.) )
        x = np.cos(fi) * np.sin(theta)
        y = np.sin(fi) * np.sin(theta)
        z = np.cos(theta)
#         return np.array([1,0,0,1])
        return np.array([1,x,y,z])
        
    state = rand_qubit()   
    for i in range(n-1):
        state = np.kron(state, rand_qubit())
    
    return state



# def pauli_dot(vector):
#     """
#     Scalar product of matrices Pauli and vector (to return ro matrix representation)
#     """ 
#     # Pauli matrixes
#     X = np.matrix([[0, 1], [1, 0]], dtype = complex)
#     Y = np.matrix([[0, -1j], [1j, 0]], dtype = complex)
#     Z = np.matrix([[1, 0], [0, -1]], dtype = complex)
#     I2 = np.eye(2)
#     s = np.array([I2,X,Y,Z])
#     S = s
    
#     # define number of qubits:
#     k = int(np.log2(np.sqrt(len(vector))))
    
#     for i in range(n-1):
#         S =  np.kron(S,s)
        
#     rho = S[0]*vector[0]/(2**k)
#     for i in range(1,len(S)):
#         rho = rho + 0.5 * S[i] * vector[i]
    
#     return rho



def check_physical(rho):
    '''
    Input: matrix rho
    Output: matrix rho, modified to be physical
    '''            
    # Positive semi-definiteness
    eigenvalues, s = LA.eigh(rho)

    for i in range(eigenvalues.shape[0]):
        value = eigenvalues[i]
#         print(value)
        if eigenvalues[i] < 0:
            eigenvalues[i] = 0
    rho_modified = np.matmul(np.matmul(s, np.diag(eigenvalues)), LA.inv(s))

    
    # Trace adjustment
    tr = np.trace(rho_modified @ rho_modified)
    if tr > 1:
        rho_modified = rho_modified/tr
    return rho_modified



def find_rho(E, b):
    """
    return rho_est via pseudo inverse matrix
    """
    S = np.matrix(E)
    rho_vec_est = np.array((((S.H @ S).I) @ S.H) @ b).reshape(-1)    
    return rho_vec_est



def operator_from_vec(vec):
    vec = np.array(vec).reshape(-1)
    X = np.matrix([[0, 1], [1, 0]], dtype = complex)
    Y = np.matrix([[0, -1j], [1j, 0]], dtype = complex)
    Z = np.matrix([[1, 0], [0, -1]], dtype = complex)
    I2 = np.eye(2)
    s = np.array([I2,X,Y,Z])
    S = s

    # define number of qubits:
    k = int(np.log2(np.sqrt(len(vec))))
    
    for i in range(k-1):
        S =  np.kron(S,s)
        
#     rho = S[0]*vector[0]/(2**k)
    opert = np.zeros(S[0].shape)
    for i in range(S.shape[0]):
        opert = opert + S[i] * vec[i]
    
    return opert


def gradient_QST(S, f, rho_vec, eps):
    """
    optimization of ML function via direct gradient descent method
    """
    rho = bloch_to_dmat(rho_vec)
    
    def R(rho):
        S0 = operator_from_vec(S[0])
        p0 = np.trace(S0 @ rho)
        r = S0 * f[0]/p0
        for j in range(1,E.shape[0]):
            Sj = operator_from_vec(S[j])
            pj =  np.trace(Sj@rho)
            r = r + (Sj * f[j]/pj)
        return r
    
    def LLH(rho):
        return np.trace(R(rho) @ rho)
    
    def rho_next(eps, Rk):
        I = np.eye(rho.shape[0])
        numer = (I+eps/2*(Rk - I)) @ rho @ (I+eps/2*(Rk+I))
        denom = np.trace(numer)
        return numer/denom

    b = 1 # for setup of eps1, eps2
    LLH_c = LLH(rho)
    
    conv_t = []
    conv_hs = []
    conv_if = []
    zero_el = np.zeros((2,2))
    
    for k in range(1000):
        Rk = R(rho)
        eps_ar = np.array([0, np.random.rand(), np.random.rand()])*b
        
        LLH_1 = LLH(rho_next(eps_ar[1], Rk))
        LLH_2 = LLH(rho_next(eps_ar[2], Rk))        
        
        eps_c = np.argmin([LLH_c, LLH_1, LLH_2])
        
        rho_n = rho_next(eps_c, Rk)
        LLH_c = LLH(rho)
        
        # check if algorithm still are improoving current guess
        if (tr_dist(Rk@rho,rho) + hs_dist(Rk@rho,rho) + if_dist(Rk@rho, rho)) < eps:
            break
        
        if k>1:
            conv_t.append(tr_dist(rho, rho_n))#/t_norm(rho, zero_el))
            conv_hs.append(hs_dist(rho, rho_n))#/hs_norm(rho, zero_el))
            conv_if.append(if_dist(rho, rho_n))#/if_norm(rho, zero_el))
        
        LLH_c = LLH(rho_n)
        rho = rho_n
        
    
    return rho, conv_t, conv_hs, conv_if




"""
Quantum Process Tomography (QPT)
"""


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
            print('End of algorithm. cost = ', C_next_cost)
            return C
    print('End of algorithm. Iteration number exceeded')
    return C

