import matplotlib.pyplot as plt

def plot_evolution(t_vals, beta_vals, rhs_vals, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(t_vals, beta_vals, 'b-')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Regularized Dimension')
    ax1.set_title('Evolution of β(t)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_vals[1:-1], rhs_vals[1:-1], 'r--', label='RHS (Theorem 3.9)')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Rate of Change')
    ax2.set_title('Verification of Theorem 3.9')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()