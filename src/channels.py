import numpy as np
import scipy.linalg

class QuantumChannelSemigroup:
    def __init__(self, L0, construction='C', gamma=1.0):
        self.L0 = L0
        self.n = L0.shape[0]
        self.gamma = gamma
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(L0)
        
        self.G = self._build_generator(construction)
        self.Pi_inf = self._build_steady_state_projection(construction)

    def _build_generator(self, construction):
        D = np.zeros(self.n)
        if construction == 'C':
            for i in range(self.n):
                if self.eigenvalues[i] > 1e-5:
                    D[i] = -self.gamma
        elif construction == 'C_prime':
            for i in range(1, self.n): 
                D[i] = -self.gamma
        return self.eigenvectors @ np.diag(D) @ self.eigenvectors.T

    def _build_steady_state_projection(self, construction):
        Pi = np.zeros((self.n, self.n))
        if construction == 'C':
            for i in range(self.n):
                if abs(self.eigenvalues[i]) < 1e-5:
                    Pi += np.outer(self.eigenvectors[:, i], self.eigenvectors[:, i])
        elif construction == 'C_prime':
            Pi += np.outer(self.eigenvectors[:, 0], self.eigenvectors[:, 0])
        return Pi

    def get_L_t(self, t):
        eGt = scipy.linalg.expm(t * self.G)
        return eGt.T @ self.L0 @ eGt