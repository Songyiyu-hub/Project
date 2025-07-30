import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

n = 100              # The number of tasks
m = 10              # The number of drivers
b = 0.02             # Deterioration effect
t0 = 0.113           # The earliest available time
U = 1000000.0       # A large constant
# The loading times θ_L_i for each task, sorted by task priority α_i
theta = [1.388, 0.772, 1.029, 0.945, 1.402, 0.297, 0.511, 0.68, 1.015, 0.757, 1.06, 1.521, 0.438, 1.902, 0.535, 0.308, 1.484, 1.027, 0.386, 1.889, 0.656, 1.049, 1.925, 0.255, 1.101, 1.989, 1.279, 0.91, 1.86, 1.658, 1.791, 1.246, 1.8, 1.164, 0.36, 1.58, 0.752, 0.917, 1.485, 1.811, 1.431, 0.679, 0.476, 1.184, 1.658, 1.848, 0.567, 1.692, 1.003, 1.743, 0.9, 1.839, 1.248, 1.816, 1.641, 1.561, 0.702, 1.311, 0.38, 1.106, 0.622, 0.763, 0.356, 1.07, 0.978, 0.383, 0.757, 1.879, 0.61, 0.366, 0.837, 0.654, 1.939, 1.953, 1.259, 1.187, 0.846, 1.113, 1.173, 0.904, 1.602, 1.712, 1.458, 1.306, 1.252, 0.282, 1.729, 1.173, 1.678, 1.655, 1.701, 1.595, 1.261, 1.372, 1.87, 1.16, 1.974, 0.883, 1.435, 1.412]
# The transportation times T_i for each task, sorted by task priority α_i
T = [1.052, 1.469, 1.141, 1.39, 1.209, 1.955, 1.948, 1.374, 1.913, 1.751, 1.704, 1.973, 2.181, 1.627, 2.441, 2.586, 1.762, 2.329, 2.78, 1.941, 2.211, 2.663, 2.329, 2.627, 2.491, 1.554, 1.952, 2.314, 2.189, 1.783, 2.007, 2.261, 2.097, 2.526, 2.727, 2.606, 2.762, 2.373, 2.608, 2.21, 2.978, 2.828, 3.34, 2.739, 2.573, 2.904, 3.325, 2.454, 3.179, 2.937, 3.709, 3.082, 2.957, 3.182, 2.974, 3.217, 4.355, 3.953, 4.225, 3.474, 3.926, 4.266, 4.639, 3.959, 4.332, 4.474, 4.502, 3.367, 4.638, 4.414, 4.452, 4.143, 3.376, 3.995, 4.406, 4.497, 3.88, 3.758, 4.094, 4.722, 3.969, 4.204, 4.355, 4.884, 4.915, 4.919, 4.269, 4.3, 4.671, 4.456, 4.718, 4.681, 4.875, 4.544, 4.49, 4.634, 4.24, 4.668, 4.738, 4.767]
# The unloading times θ_U_i for each task, sorted by task priority α_i
Z = [0.761, 1.01, 1.435, 1.122, 1.16, 0.834, 0.77, 1.806, 0.608, 1.328, 1.163, 0.258, 1.091, 0.739, 0.56, 0.529, 0.988, 0.388, 0.251, 0.445, 1.56, 0.269, 0.464, 1.586, 1.013, 1.989, 1.96, 1.626, 0.961, 1.976, 1.487, 1.542, 1.32, 1.146, 1.599, 0.619, 1.168, 1.857, 0.905, 1.37, 0.297, 1.371, 0.685, 1.158, 1.047, 0.289, 0.826, 1.576, 0.918, 0.878, 0.573, 0.982, 1.899, 0.889, 1.661, 1.437, 0.219, 0.402, 0.927, 1.718, 1.356, 0.57, 0.265, 0.982, 0.352, 0.696, 0.313, 1.43, 0.268, 0.97, 0.544, 1.392, 1.67, 0.431, 0.372, 0.296, 1.959, 1.994, 1.388, 0.615, 1.548, 1.121, 1.123, 0.364, 0.501, 1.487, 1.325, 1.932, 0.715, 1.215, 0.658, 0.917, 0.884, 1.486, 1.09, 1.655, 1.646, 1.955, 1.374, 1.604]

df = pd.DataFrame({
    'task_name': ['task_' + str(i + 1) for i in range(n)],
    'theta_i': theta,
    'T_i': T,
    'Z_i': Z
})
tasks_list = [(row['task_name'], row['theta_i'], row['T_i'], row['Z_i']) for _, row in df.iterrows()]

def heuristic_1(df, m, t0, b):
    completion_times = [t0] * m
    drivers_tasks = {k: [] for k in range(m)}
    task_assignments = []
    for i, row in df.iterrows():
        task = (row['task_name'], row['theta_i'], row['T_i'], row['Z_i'])
        if i < m:
            k = i
        else:
            k = completion_times.index(min(completion_times))
        drivers_tasks[k].append(task)
        C_k = completion_times[k]
        C_k = (b + 1) ** 2 * task[1] + (b + 2) * task[2] + (b + 1) * task[3] + C_k * (b + 1) ** 2
        completion_times[k] = round(C_k, 2)
        task_assignments.append(k)
    return task_assignments, max(completion_times)

def calculate_completion_times(decoded_tasks, b, t0):
    m = len(decoded_tasks)
    completion_times = [0] * m
    for k in range(m):
        tasks_for_driver = decoded_tasks[k]
        n_k = len(tasks_for_driver)
        f_values = [(b + 1) ** 2 * task[1] + (b + 2) * task[2] + (b + 1) * task[3] for task in tasks_for_driver]
        sorted_indices = np.argsort(f_values)
        C_k = (b + 1) ** (2 * n_k) * t0
        for j in range(n_k):
            C_k += f_values[sorted_indices[j]] * (b + 1) ** (2 * (n_k - j - 1))
        completion_times[k] = round(C_k, 2)
    return completion_times

def decode_and_calculate_cmax(encoding, tasks_list, b, t0, m):
    decoded_tasks = {k: [] for k in range(m)}
    for task_index, driver_id in enumerate(encoding):
        task = tasks_list[task_index]
        decoded_tasks[driver_id].append(task)
    completion_times = calculate_completion_times(decoded_tasks, b, t0)
    Cmax = round(max(completion_times), 2)
    return Cmax, completion_times

def reverse_mutation(encoding):
    x1, x2 = sorted(random.sample(range(len(encoding)), 2))
    new_encoding = encoding[:x1] + encoding[x1:x2 + 1][::-1] + encoding[x2 + 1:]
    return new_encoding

def roulette_wheel_mutation(encoding, tasks_list, b, t0):
    m = len(set(encoding))
    n = len(encoding)
    decoded_tasks = [[] for _ in range(m)]
    for task_index, driver_id in enumerate(encoding):
        decoded_tasks[driver_id].append(tasks_list[task_index])
    completion_times = calculate_completion_times(decoded_tasks, b, t0)
    if any(c == 0 for c in completion_times):
        print("Warning: Some drivers have zero completion time and will be skipped.")
        completion_times = [max(c, 1) for c in completion_times]
    fitness_values = [ C for C in completion_times]
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    selected_driver = np.random.choice(range(m), p=probabilities)
    driver_tasks_indices = [i for i in range(n) if encoding[i] == selected_driver]
    if driver_tasks_indices:
        task_to_mutate = random.choice(driver_tasks_indices)
        new_driver = random.choice([i for i in range(m) if i != selected_driver])
        new_encoding = encoding[:task_to_mutate] + [new_driver] + encoding[task_to_mutate + 1:]
        return new_encoding
    return encoding

def selective_exchange_operator(encoding, tasks_list, b, t0, m, n):
    decoded_tasks = {k: [] for k in range(m)}
    for task_index, driver_id in enumerate(encoding):
        task = tasks_list[task_index]
        decoded_tasks[driver_id].append(task)
    completion_times = calculate_completion_times(decoded_tasks, b, t0)
    driver_long = completion_times.index(max(completion_times))
    driver_short = completion_times.index(min(completion_times))
    new_encoding = encoding[:]
    tasks_long = [i for i in range(n) if new_encoding[i] == driver_long]
    tasks_short = [i for i in range(n) if new_encoding[i] == driver_short]
    if not tasks_long or not tasks_short:
        return new_encoding
    task_long = random.choice(tasks_long)
    task_short = random.choice(tasks_short)
    new_encoding[task_long], new_encoding[task_short] = new_encoding[task_short], new_encoding[task_long]
    return new_encoding

def shake(encoding,shaking_time=1):
    new_encoding = encoding[:]
    for i in range(shaking_time):
        x1, x2 = random.sample(range(len(encoding)), 2)
        new_encoding[x1], new_encoding[x2] = new_encoding[x2], new_encoding[x1]
    return new_encoding

def random_task_reassignment(encoding, m):
    n = len(encoding)
    segment_size = random.randint(1, n // 2)
    segment_start = random.randint(0, n - segment_size)
    segment_end = segment_start + segment_size
    new_segment = [random.randint(0, m - 1) for _ in range(segment_size)]
    new_encoding = encoding[:segment_start] + new_segment + encoding[segment_end:]
    return new_encoding

def save_cmax_data(time_values, Cmax_values, filename='vns_cmax_data.csv'):
    data = {
        'Time': time_values,
        'Cmax': Cmax_values
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def VNS_algorithm(tasks_list, b, t0, m, n, max_iterations=10000000, time_limit=60, shaking_time=1,save_interval=0.01, filename='vns_cmax_data.csv'):
    best_solution, best_Cmax = heuristic_1(df, m, t0, b)
    iteration = 0
    k = 1
    Cmax_values = [best_Cmax]
    time_values = [0]
    start_time = time.time()
    elapsed_time = 0
    save_cmax_data([elapsed_time], [best_Cmax], filename)
    last_saved_time = 0
    while iteration < max_iterations:
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            print(f"Time limit reached: {elapsed_time:.2f} seconds")
            break
        new_solution = shake(best_solution,shaking_time)
        if k == 1:
            new_solution = random_task_reassignment(new_solution, m)
        elif k == 2:
            new_solution = reverse_mutation(new_solution)
        elif k == 3:
            new_solution = roulette_wheel_mutation(new_solution, tasks_list, b, t0)
        elif k == 4:
            new_solution = selective_exchange_operator(new_solution, tasks_list, b, t0, m, n)
        else:
            break
        new_Cmax, _ = decode_and_calculate_cmax(new_solution, tasks_list, b, t0, m)
        if new_Cmax < best_Cmax:
            best_solution = new_solution
            best_Cmax = new_Cmax
            k = 1
        else:
            k += 1
        if k > 4:
            k = 1
        if elapsed_time - last_saved_time >= save_interval:
            Cmax_values.append(round(best_Cmax, 2))
            time_values.append(round(elapsed_time, 2))
            save_cmax_data(time_values, Cmax_values, filename)
            last_saved_time = elapsed_time
        iteration += 1
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    return best_solution, best_Cmax, Cmax_values, time_values, total_time

tasks_list = [(row['task_name'], row['theta_i'], row['T_i'], row['Z_i']) for index, row in df.iterrows()]
best_solution, best_Cmax, Cmax_values, time_values, total_time = VNS_algorithm(tasks_list, b, t0, m, n, max_iterations=10000000, time_limit=60, shaking_time=1,save_interval=0.01)
print(f"Optimized best task allocation scheme:{best_solution}")
print(f"Corresponding maximum completion time (Cmax):{round(best_Cmax, 2)}")
print(f"Total runtime of the VNS-E algorithm:{round(total_time, 2)} seconds")
plt.figure(figsize=(10, 6))
plt.plot(time_values, Cmax_values, marker='o', color='b')
plt.title('VNS-E Algorithm Convergence')
plt.xlabel('Time (seconds)')
plt.ylabel('Cmax')
plt.grid(True)
plt.show()