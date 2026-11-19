import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.topology import build_simplicial_complex, get_hodge_laplacian
from src.channels import QuantumChannelSemigroup
from src.spectral_functional import beta_epsilon, evolution_rhs
from src.utils import plot_evolution

def load_bitcoin_proxy_data():
    """模拟比特币订单流网络(在没有真实数据时)"""
    print("Generating proxy Bitcoin network data...")
    np.random.seed(101)
    # 模拟 100 个节点的特征点云
    return np.random.rand(100, 3)

def main():
    points = load_bitcoin_proxy_data()
    st = build_simplicial_complex(points, max_edge_length=0.15, max_dim=1)
    L1 = get_hodge_laplacian(st, k=1)
    
    if L1 is None:
        print("Failed to build complex.")
        return

    channel = QuantumChannelSemigroup(L1, construction='C')
    epsilon = 0.05
    t_vals = np.linspace(0, 2, 40)
    
    beta_vals = [beta_epsilon(channel.get_L_t(t), epsilon) for t in t_vals]
    rhs_vals = [evolution_rhs(channel.G, channel.get_L_t(t), epsilon) for t in t_vals]
    
    print("Bitcoin Consistency Check (Section 5) completed.")
    plot_evolution(t_vals, beta_vals, rhs_vals, 'bitcoin_consistency_check.png')

if __name__ == "__main__":
    main()