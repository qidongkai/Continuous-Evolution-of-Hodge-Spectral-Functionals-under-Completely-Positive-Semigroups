import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.topology import build_simplicial_complex, get_hodge_laplacian
from src.channels import QuantumChannelSemigroup
from src.spectral_functional import beta_epsilon, evolution_rhs
from src.utils import plot_evolution

def main():
    np.random.seed(42)
    points = np.random.rand(30, 3)
    st = build_simplicial_complex(points, max_edge_length=0.4, max_dim=2)
    L1 = get_hodge_laplacian(st, k=1)
    
    if L1 is None:
        print("Failed to build complex.")
        return

    channel = QuantumChannelSemigroup(L1, construction='C')
    epsilon = 0.1
    t_vals = np.linspace(0, 5, 50)
    
    beta_vals = [beta_epsilon(channel.get_L_t(t), epsilon) for t in t_vals]
    rhs_vals = [evolution_rhs(channel.G, channel.get_L_t(t), epsilon) for t in t_vals]
    
    dt = t_vals[1] - t_vals[0]
    dbeta_dt = np.gradient(beta_vals, dt)
    
    error = np.mean(np.abs(dbeta_dt[1:-1] - np.array(rhs_vals)[1:-1]))
    print(f"Mean Absolute Error between numerical dβ/dt and RHS: {error:.6e}")
    
    plot_evolution(t_vals, beta_vals, rhs_vals, 'theorem_3_9_verification.png')
    print("Saved plot to theorem_3_9_verification.png")

if __name__ == "__main__":
    main()