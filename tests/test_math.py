import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.channels import QuantumChannelSemigroup
from src.spectral_functional import beta_epsilon, evolution_rhs

def test_beta_epsilon():
    L = np.diag([0.0, 1.0, 2.0])
    eps = 1.0
    # \sum eps^2/(lambda+eps^2) = 1/1 + 1/2 + 1/3 = 1.8333
    expected = 1.0 + 0.5 + 1.0/3.0
    assert np.isclose(beta_epsilon(L, eps), expected), "beta_epsilon calculation failed"

def test_generator_dissipative():
    L = np.diag([0.0, 1.0])
    channel = QuantumChannelSemigroup(L, 'C')
    evals = np.linalg.eigvals(channel.G)
    # 检查 G 的非零特征值是否具有负实部 (A6a)
    assert np.all(np.real(evals) <= 1e-10), "Generator is not dissipative"

if __name__ == "__main__":
    test_beta_epsilon()
    test_generator_dissipative()
    print("All tests passed!")