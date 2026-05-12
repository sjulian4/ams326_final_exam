import numpy as np 
import matplotlib.pyplot as plt

# shooting method --> explicit RK4 engine
def rk4_system(s, N_steps=300):
    """Integrates the 2D state trajectory natively using manual RK4 steps."""
    t_arr = np.linspace(0.0, 3.0, N_steps + 1)
    dt = 3.0 / N_steps
    u1, u2 = np.zeros(N_steps + 1), np.zeros(N_steps + 1)
    
    u1[0] = 0.0  # y(0) = 0
    u2[0] = s    # y'(0) = s (shooting parameter)
    
    for i in range(N_steps):
        t = t_arr[i]
        y, v = u1[i], u2[i]
        
        # explicit evaluation of internal slope vectors
        k1_y = v
        k1_v = -4.0 * y + (t**3) * np.exp(-t**2)
        
        k2_y = v + 0.5 * dt * k1_v
        k2_v = -4.0 * (y + 0.5 * dt * k1_y) + ((t + 0.5*dt)**3) * np.exp(-(t + 0.5*dt)**2)
        
        k3_y = v + 0.5 * dt * k2_v
        k3_v = -4.0 * (y + 0.5 * dt * k2_y) + ((t + 0.5*dt)**3) * np.exp(-(t + 0.5*dt)**2)
        
        k4_y = v + dt * k3_v
        k4_v = -4.0 * (y + dt * k3_y) + ((t + dt)**3) * np.exp(-(t + dt)**2)
        
        u1[i+1] = y + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        u2[i+1] = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
    return t_arr, u1

# native secant root finding to determine optimal initial slope s*
s_0 = 0.0
_, u1_s0 = rk4_system(s_0)
phi_0 = u1_s0[-1]

s_1 = 1.0
_, u1_s1 = rk4_system(s_1)
phi_1 = u1_s1[-1]

# linear interpolation step yields exact initial trajectory slope
s_opt = s_1 - phi_1 * (s_1 - s_0) / (phi_1 - phi_0)
t_shoot, y_shoot = rk4_system(s_opt, N_steps=100) # output requested 100 interval grid

# finite difference method
dt = 0.1
N_fd = int(3.0 / dt)
t_fd = np.linspace(0.0, 3.0, N_fd + 1)

# explicit manual matrix assembly for internal nodes (indices 1 to N_fd-1)
M = N_fd - 1
A = np.zeros((M, M))
b = np.zeros(M)

main_diag = 4.0 * (dt**2) - 2.0
for i in range(M):
    t_val = t_fd[i + 1]
    A[i, i] = main_diag
    if i > 0:
        A[i, i - 1] = 1.0
    if i < M - 1:
        A[i, i + 1] = 1.0
    b[i] = (dt**2) * (t_val**3) * np.exp(-t_val**2)

# solve system natively
y_inner = np.linalg.solve(A, b)
y_fd = np.concatenate(([0.0], y_inner, [0.0]))

print(f"Optimal Shooting Slope s* : {s_opt:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(t_shoot, y_shoot, 'b-', linewidth=2, label=f'Shooting Method ($s^*={s_opt:.4f}$)')
plt.plot(t_fd, y_fd, 'ro', markersize=5, label='Finite Difference ($\Delta t=0.1$)')
plt.title("BVP Numerical Trajectory Comparisons")
plt.xlabel("Time t")
plt.ylabel("State y(t)")
plt.grid(True, linestyle=':')
plt.legend()
plt.show()