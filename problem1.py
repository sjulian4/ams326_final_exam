import numpy as np
import matplotlib.pyplot as plt

np.random.seed(326) 

# coordinate generation
N_cities = 20
coords = np.random.uniform(0, 100, size=(N_cities, 3))

# distance matrix
D = np.zeros((N_cities, N_cities))
for i in range(N_cities):
    for j in range(N_cities):
        # native component-wise distance calculation
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dz = coords[i, 2] - coords[j, 2]
        D[i, j] = np.sqrt(dx**2 + dy**2 + dz**2)

def compute_tour_length(tour, dist_matrix):
    """Calculates total path distance explicitly by summing sequential edge weights."""
    length = 0.0
    for i in range(len(tour) - 1):
        length += dist_matrix[tour[i], tour[i+1]]
    return length

# greedy heuristic 
unvisited = list(range(1, N_cities))
greedy_tour = [0]  # start at city 1 (index 0)

current_city = 0
while unvisited:
    # explicit loop to find nearest neighbor
    min_dist = float('inf')
    nearest_city = -1
    for candidate in unvisited:
        if D[current_city, candidate] < min_dist:
            min_dist = D[current_city, candidate]
            nearest_city = candidate
    
    greedy_tour.append(nearest_city)
    unvisited.remove(nearest_city)
    current_city = nearest_city

greedy_tour.append(0)  # close tour back to city 1
greedy_length = compute_tour_length(greedy_tour, D)

# stochastic refinement (simulated annealing)
n_iterations = 100000
T = 100.0       # initial temperature
T_min = 0.001   # final temperature
cooling_rate = (T_min / T) ** (1.0 / n_iterations)

current_tour = greedy_tour[:-1]  # exclude closing node during interior optimization
current_length = compute_tour_length(current_tour + [current_tour[0]], D)

best_tour = list(current_tour)
best_length = current_length

for step in range(n_iterations):
    # propose a 2-opt segment reversal
    i, j = sorted(np.random.choice(range(1, N_cities), size=2, replace=False))
    new_tour = current_tour[:i] + current_tour[i:j+1][::-1] + current_tour[j+1:]
    new_length = compute_tour_length(new_tour + [new_tour[0]], D)
    
    # explicit acceptance evaluation
    dE = new_length - current_length
    if dE < 0 or np.random.rand() < np.exp(-dE / T):
        current_tour = new_tour
        current_length = new_length
        if current_length < best_length:
            best_tour = list(current_tour)
            best_length = current_length
            
    T *= cooling_rate  # geometric cooling update

final_tour = best_tour + [best_tour[0]]



# final analysis

print(f"Greedy Baseline Path Length : {greedy_length:.4f}")
print(f"Optimized Path Length       : {best_length:.4f}")
print(f"Optimization Improvement    : {greedy_length - best_length:.4f}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
pt = coords[final_tour]
ax.plot(pt[:,0], pt[:,1], pt[:,2], 'b-o', linewidth=2, label=f'Optimized Tour ({best_length:.1f})')
ax.scatter(coords[0,0], coords[0,1], coords[0,2], color='red', s=100, label='Start (City 1)')
ax.set_title("Stochastically Refined 3D Flight Path")
ax.legend()
plt.show()