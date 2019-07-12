import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

import primitives as pm

def is_feasible(bloch_vector):
    return 1 - la.norm(bloch_vector, ord=2)

class Tomograph:
    ''' Basic class for QST '''
    def __init__(self, set_of_POVMs=pm.MUB_POVM, dst='hs'):
        '''
        Initialize quantum state tomography procedure with the preferred set_of_POVMs
        Input:
            set_of_POVMs -- like in state_to_proba function (list of POVMs)
        '''
        self.set_of_POVMs = set_of_POVMs
        self.dst = dst
    
    def experiment(self, state, n_measurements):
        """
        Perform quantum tomography on the state with set_of_POVMs
        Input:
            State -- numpy matrix
            set_of_POVMs -- like in state_to_proba function (list of POVMs)
            n_measurements -- list of ints, each number corresponds to each POVM in set_of_POVMs
        Example:
            state = density([1, 0])
            povm = [[density([1, 0])],
                    [density([1/np.sqrt(2), 1/np.sqrt(2)])],
                    [density([1/np.sqrt(2), -1j/np.sqrt(2)])]]
            n_measurements = [1000] * 3
        """
        self.state = state
        self.results = []
        for subset, num in zip(self.set_of_POVMs, n_measurements):
            proba_subset = []
            p = 0
            for POVM in subset:
                proba_subset.append(abs(pm.product(state, POVM)))
                p += proba_subset[-1]
            proba_subset.append(1-p)
            results_subset = np.random.multinomial(num, proba_subset)
            self.results.append(results_subset)
        self._compute_freq()
        
    def point_estimate(self, method='lin', physical=True):
        if method == 'lin':
            return self._point_estimate_lin(physical=physical)
        else:
            return self._point_estimate_mle(physical=physical)
    
    def bootstrap_state(self, state, n_measurements, n_repeats, method='lin', dst='hs'):
        """
        Perform quantum tomography on the point estimate *n_repeats* times
        Output:
            Sorted list of distances between point estimate and corresponding estimated matrices
        """
        dist = [0]
        for i in range(n_repeats):
            try:
                results = self.experiment(state, n_measurements)
                rho = self.point_estimate(method=method)
            except ValueError:
                i -= 1
                continue
            if dst == 'hs':
                dist.append(pm.hs(rho, state))
            elif dst == 'trace':
                dist.append(pm.trace(rho, state))
            else:
                dist.append(pm.infidelity(rho, state))
        dist.sort()
        proba = np.linspace(0, 1, n_repeats+1).tolist()
        return dist, proba
    
    def _compute_freq(self):
        """ 
        Input:
            List of numpy arrays with number of measurement outcomes corresponding to each POVM
        Example:
            [np.array([30, 20]), np.array([10, 90]), np.array([50, 50])]
        Output:
            Vector of frequencies and variance of 1-norm of this vector in self.freq and self.var respectively
        """
        freq = []
        variance = []
        for subset in self.results:
            N = subset.sum()
            freq.append(subset[:-1] / N)
            variance.append(freq[-1] * (1 - freq[-1]) / N)
        freq.append(np.array(1))
        self.freq = np.array(np.hstack(freq)).T
        self.var = max(variance)
    
    def _likelihood(self, bloch_vector, eps=1e-8):
        rho = pm.Basis(dim=2).vector_to_matrix(bloch_vector)
        likelihood = 0
        for i, subset in enumerate(self.set_of_POVMs):
            last_proba = 1
            for j, POVM in enumerate(subset):
                proba = np.real(pm.product(POVM, rho))
                last_proba -= proba
                likelihood += self.results[i][j] * np.log(proba + eps)
            likelihood += self.results[i][-1] * np.log(last_proba + eps)
        return -likelihood
    
    def _point_estimate_lin(self, physical):
        s = pm.proba_to_state(self.set_of_POVMs) @ self.freq
        if physical and la.norm(s[1:], ord=2) > 0.5:
            s[1:] /= la.norm(s[1:], ord=2) * 2
        rho = pm.Basis(dim=2).vector_to_matrix(s, bloch=False)
        return rho
    
    def _point_estimate_mle(self, physical=True):
        constraints = [
            {'type': 'ineq', 'fun': is_feasible},
        ]
#         x0 = pm.Basis().matrix_to_vector(pm.SIGMA_I / 2)
        x0 = pm.Basis(dim=2).matrix_to_vector(self.point_estimate('lin'))
        opt_res = minimize(self._likelihood, x0, constraints=constraints, method='SLSQP')
        if physical and la.norm(opt_res.x, ord=2) > 1:
            s = opt_res.x / la.norm(opt_res.x, ord=2)
            rho = pm.Basis(dim=2).vector_to_matrix(s)
        else:
            rho = pm.Basis(dim=2).vector_to_matrix(opt_res.x)
        return rho
