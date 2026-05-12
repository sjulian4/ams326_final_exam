import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# internal slope resolution
def compute_implicit_slope(t, x):
    """Calculates internal root derivative states explicitly using standard native loops."""
    C = x**3 - 3.0 * np.exp(-t**3)
    u = 0.0  # base starting search estimate
    
    for _ in range(50): # inner fixed bounds search array
        g_val = np.exp(-u) - u - C
        g_prime = -np.exp(-u) - 1.0
        step = g_val / g_prime
        u -= step
        if abs(step) < 1e-11:
            break
    return u

# numerical solution --> RK4
N_rk = 1000
t_arr = np.linspace(0.0, 5.0, N_rk + 1)
dt = 5.0 / N_rk
x_arr = np.zeros(N_rk + 1)
x_arr[0] = 1.0  # initial condition x(0) = 1

for n in range(N_rk):
    t_n = t_arr[n]
    x_n = x_arr[n]
    
    k1 = compute_implicit_slope(t_n, x_n)
    k2 = compute_implicit_slope(t_n + 0.5 * dt, x_n + 0.5 * dt * k1)
    k3 = compute_implicit_slope(t_n + 0.5 * dt, x_n + 0.5 * dt * k2)
    k4 = compute_implicit_slope(t_n + dt, x_n + dt * k3)
    
    x_arr[n+1] = x_n + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

# polynomial interpolation
target_indices = [int(node / dt) for node in range(6)]
t_nodes = t_arr[target_indices]
x_nodes = x_arr[target_indices]

# explicit resolution of exact mapping polynomial coefficient structures
p5_coefficients = np.polyfit(t_nodes, x_nodes, 5)
p5_evaluator = np.poly1d(p5_coefficients)
x_poly_dense = p5_evaluator(t_arr)

# function fitting
def analytical_model(t, alpha, beta):
    return 1.0 + alpha * t * np.exp(-beta * t)

# perform nonlinear regression optimization over exact matching array blocks
opt_params, _ = curve_fit(analytical_model, t_nodes, x_nodes, p0=[1.0, 0.5])
alpha_opt, beta_opt = opt_params
x_model_dense = analytical_model(t_arr, alpha_opt, beta_opt)

print(f"Optimal Structural Fit Parameter Alpha : {alpha_opt:.5f}")
print(f"Optimal Structural Fit Parameter Beta  : {beta_opt:.5f}")

# visualization
plt.figure(figsize=(10, 6))
plt.plot(t_arr, x_arr, 'b-', linewidth=2.5, label='Numerical Trajectory (RK4)')
plt.plot(t_arr, x_poly_dense, 'g--', linewidth=1.5, label='P_5(t) Polynomial Approximator')
plt.plot(t_arr, x_model_dense, 'r-.', linewidth=1.5, label=f'Model: 1 + {alpha_opt:.2f}t e^{{-{beta_opt:.2f}t}}')
plt.scatter(t_nodes, x_nodes, color='black', zorder=5, label='Sample Extraction Nodes')
plt.title("Implicit Non-Autonomous IVP Optimization Comparison")
plt.xlabel("Time t")
plt.ylabel("State Variable x(t)")
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()