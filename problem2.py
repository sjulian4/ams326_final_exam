import numpy as np 
import matplotlib.pyplot as plt

# shooting method --> explicit RK4 engine
def rk4_system(s, N_steps=300):
    t_arr = np.linspace(0.0, 3.0, N_steps + 1)
    dt = 3.0 / N_steps
    u1, u2 = np.zeros(N_steps + 1), np.zeros(N_steps + 1)
    
    u1[0] = 0.0  # y(0) = 0
    u2[0] = s    # y'(0) = s 
    
    for i in range(N_steps):
        t = t_arr[i]
        y, v = u1[i], u2[i]
        
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

# Determine optimal initial slope s*
s_0, s_1 = 0.0, 1.0
_, u1_s0 = rk4_system(s_0)
_, u1_s1 = rk4_system(s_1)
phi_0, phi_1 = u1_s0[-1], u1_s1[-1]

s_opt = s_1 - phi_1 * (s_1 - s_0) / (phi_1 - phi_0)

# Set grid for 100 intervals (N=100)
N = 100
t_shoot, y_shoot = rk4_system(s_opt, N_steps=N)

# Update Finite Difference Method to use the same N=100 grid
dt_fd = 3.0 / N
t_fd = np.linspace(0.0, 3.0, N + 1)
M = N - 1
A = np.zeros((M, M))
b = np.zeros(M)
main_diag = 4.0 * (dt_fd**2) - 2.0

for i in range(M):
    t_val = t_fd[i + 1]
    A[i, i] = main_diag
    if i > 0: A[i, i - 1] = 1.0
    if i < M - 1: A[i, i + 1] = 1.0
    b[i] = (dt_fd**2) * (t_val**3) * np.exp(-t_val**2)

y_inner = np.linalg.solve(A, b)
y_fd = np.concatenate(([0.0], y_inner, [0.0]))

# --- ADDED: TABLE GENERATION ---
print(f"{'Index (n)':<10} | {'t_n':<10} | {'y_shoot(t_n)':<15} | {'y_fd(t_n)':<15}")
print("-" * 55)
for n in range(len(t_shoot)):
    # Display every point or a specific subset
    print(f"{n:<10} | {t_shoot[n]:<10.3f} | {y_shoot[n]:<15.6f} | {y_fd[n]:<15.6f}")

print(f"\nOptimal Shooting Slope s* : {s_opt:.6f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(t_shoot, y_shoot, 'b-', label='Shooting Method')
plt.plot(t_fd, y_fd, 'r--', label='Finite Difference Method')
plt.title("Numerical Solution Comparison (N=100)")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.show()