import numpy as np

def generate_ordered_data(n=20, m=3, b=0.01, U=1e6):
    t0 = round(np.random.uniform(0, 2), 3)
    theta = np.round(np.random.uniform(0.2, 2, n), 3)
    T = np.round(np.random.uniform(1, 5, n), 3)
    Z = np.round(np.random.uniform(0.2, 2, n), 3)
    alpha = (b + 1) ** 2 * theta + (b + 2) * T + (b + 1) * Z
    sorted_indices = np.argsort(alpha)
    theta_sorted = theta[sorted_indices].tolist()
    T_sorted = T[sorted_indices].tolist()
    Z_sorted = Z[sorted_indices].tolist()
    print("# ==== Copy the following content to the main code ====")
    print(f"n = {n}       # The number of tasks")
    print(f"m = {m}       # The number of drivers")
    print(f"b = {b}       # Deterioration effect")
    print(f"t0 = {t0}     # The earliest available time")
    print(f"U = {U}      # A large constant")
    print(f"theta = {theta_sorted}  # The loading times θ_L_i for each task, sorted by task priority α_i")
    print(f"T = {T_sorted}          # The transportation times T_i for each task, sorted by task priority α_i")
    print(f"Z = {Z_sorted}          # The unloading times θ_U_i for each task, sorted by task priority α_i")
    print("# ==============================")

if __name__ == "__main__":
    generate_ordered_data(n=20, m=5, b=0.01)