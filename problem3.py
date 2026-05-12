import numpy as np 
import matplotlib.pyplot as plt

# stock simulation
V_0 = 250.0
mu = 0.0019
sigma = 0.0084
N_days = 100

# explicit standard normal scale mapping
Z = np.random.normal(0.0, 1.0, N_days)
x_rates = mu + sigma * Z

V = np.zeros(N_days)
V[0] = V_0
for n in range(N_days - 1):
    V[n+1] = V[n] * (1.0 + x_rates[n])

# sample validation table
print(f"{'n':<5} | {'x_n':<12} | {'V_n':<10}")
print("-" * 32)
for n in range(N_days):
    print(f"{n:<5} | {x_rates[n]:<12.6f} | {V[n]:<10.2f}")

# frequency analysis DFT
def native_dft(vector):
    """Executes the standard discrete mathematical Fourier mapping explicitly."""
    N = len(vector)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        # native integration summation loop
        harmonic_sum = 0.0 + 0.0j
        for n in range(N):
            angle = -2.0j * np.pi * k * n / N
            harmonic_sum += vector[n] * np.exp(angle)
        X[k] = harmonic_sum
    return X

dft_vals = native_dft(V)
dft_magnitude = np.abs(dft_vals)

# visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(range(N_days), V, 'g-o', markersize=4)
ax1.set_title("Simulated Stock Asset Price Trajectory ($V_n$)")
ax1.set_ylabel("Price")
ax1.grid(True, linestyle=':')

ax2.stem(range(N_days // 2), dft_magnitude[:N_days // 2], basefmt="b-")
ax2.set_title("Explicit Native DFT Harmonic Spectral Amplitude")
ax2.set_xlabel("Frequency Index k")
ax2.set_ylabel("|X_k|")
ax2.grid(True, linestyle=':')
plt.tight_layout()
plt.show()