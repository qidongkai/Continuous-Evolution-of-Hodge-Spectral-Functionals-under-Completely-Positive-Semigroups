import numpy as np

def beta_epsilon(L_t, epsilon):
    """定义 3.1: 正则化上同调维数"""
    evals = np.linalg.eigvalsh(L_t)
    return np.sum(epsilon**2 / (evals + epsilon**2))

def evolution_rhs(G, L_t, epsilon):
    """定理 3.9 右侧的配对"""
    n = L_t.shape[0]
    F = epsilon**2 * np.linalg.inv(L_t + epsilon**2 * np.eye(n))
    return np.trace(G.T @ F) + np.trace(F @ G)