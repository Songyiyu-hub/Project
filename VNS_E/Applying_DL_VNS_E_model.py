import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from ast import literal_eval
import joblib
import pickle

loaded_model = load_model('time_limit_predictor.keras')
scaler = joblib.load('scaler.joblib')
with open('time_limit_cmax_dict.pkl', 'rb') as f:
    time_limit_cmax_dict = pickle.load(f)

def process_list_column(data, column_name):
    data[column_name] = data[column_name].apply(literal_eval)
    data[column_name + '_mean'] = data[column_name].apply(lambda x: np.mean(x))
    data[column_name + '_max'] = data[column_name].apply(lambda x: np.max(x))
    data[column_name + '_min'] = data[column_name].apply(lambda x: np.min(x))
    data[column_name + '_std'] = data[column_name].apply(lambda x: np.std(x))
    return data

def preprocess_features(input_data):
    input_df = pd.DataFrame([input_data], columns=['n', 'm', 'b', 't0', 'theta_sorted', 'T_sorted', 'Z_sorted'])
    input_df = process_list_column(input_df, 'theta_sorted')
    input_df = process_list_column(input_df, 'T_sorted')
    input_df = process_list_column(input_df, 'Z_sorted')
    features = input_df[['n', 'm', 'b', 't0',
                         'theta_sorted_mean', 'theta_sorted_max', 'theta_sorted_min', 'theta_sorted_std',
                         'T_sorted_mean', 'T_sorted_max', 'T_sorted_min', 'T_sorted_std',
                         'Z_sorted_mean', 'Z_sorted_max', 'Z_sorted_min', 'Z_sorted_std']].values
    features = scaler.transform(features)
    return features

def predict_optimal_time_limit(input_data):
    processed_features = preprocess_features(input_data)
    min_distance = float('inf')
    closest_key = None
    for key in time_limit_cmax_dict:
        features = time_limit_cmax_dict[key]['Features']
        feature_array = np.array([
            features['n'], features['m'], features['b'], features['t0'],
            features['theta_sorted_mean'], features['theta_sorted_max'], features['theta_sorted_min'],
            features['theta_sorted_std'],
            features['T_sorted_mean'], features['T_sorted_max'], features['T_sorted_min'], features['T_sorted_std'],
            features['Z_sorted_mean'], features['Z_sorted_max'], features['Z_sorted_min'], features['Z_sorted_std']
        ])
        standardized_feature_array = scaler.transform(feature_array.reshape(1, -1))[0]
        distance = np.linalg.norm(processed_features[0] - standardized_feature_array)
        if distance < min_distance:
            min_distance = distance
            closest_key = key
    if closest_key is not None:
        optimal_time_limit = time_limit_cmax_dict[closest_key]['Time_limit']
    else:
        print('What happened?')
    optimal_time_limit = np.clip(optimal_time_limit, 1, 10)
    return optimal_time_limit

if __name__ == "__main__":
    n = 100  # The number of tasks
    m = 10  # The number of drivers
    b = 0.02  # Deterioration effect
    t0 = 0.113  # The earliest available time
    U = 1000000.0  # A large constant
    # The loading times θ_L_i for each task, sorted by task priority α_i
    theta = [1.388, 0.772, 1.029, 0.945, 1.402, 0.297, 0.511, 0.68, 1.015, 0.757, 1.06, 1.521, 0.438, 1.902, 0.535,
             0.308, 1.484, 1.027, 0.386, 1.889, 0.656, 1.049, 1.925, 0.255, 1.101, 1.989, 1.279, 0.91, 1.86, 1.658,
             1.791, 1.246, 1.8, 1.164, 0.36, 1.58, 0.752, 0.917, 1.485, 1.811, 1.431, 0.679, 0.476, 1.184, 1.658, 1.848,
             0.567, 1.692, 1.003, 1.743, 0.9, 1.839, 1.248, 1.816, 1.641, 1.561, 0.702, 1.311, 0.38, 1.106, 0.622,
             0.763, 0.356, 1.07, 0.978, 0.383, 0.757, 1.879, 0.61, 0.366, 0.837, 0.654, 1.939, 1.953, 1.259, 1.187,
             0.846, 1.113, 1.173, 0.904, 1.602, 1.712, 1.458, 1.306, 1.252, 0.282, 1.729, 1.173, 1.678, 1.655, 1.701,
             1.595, 1.261, 1.372, 1.87, 1.16, 1.974, 0.883, 1.435, 1.412]
    # The transportation times T_i for each task, sorted by task priority α_i
    T = [1.052, 1.469, 1.141, 1.39, 1.209, 1.955, 1.948, 1.374, 1.913, 1.751, 1.704, 1.973, 2.181, 1.627, 2.441, 2.586,
         1.762, 2.329, 2.78, 1.941, 2.211, 2.663, 2.329, 2.627, 2.491, 1.554, 1.952, 2.314, 2.189, 1.783, 2.007, 2.261,
         2.097, 2.526, 2.727, 2.606, 2.762, 2.373, 2.608, 2.21, 2.978, 2.828, 3.34, 2.739, 2.573, 2.904, 3.325, 2.454,
         3.179, 2.937, 3.709, 3.082, 2.957, 3.182, 2.974, 3.217, 4.355, 3.953, 4.225, 3.474, 3.926, 4.266, 4.639, 3.959,
         4.332, 4.474, 4.502, 3.367, 4.638, 4.414, 4.452, 4.143, 3.376, 3.995, 4.406, 4.497, 3.88, 3.758, 4.094, 4.722,
         3.969, 4.204, 4.355, 4.884, 4.915, 4.919, 4.269, 4.3, 4.671, 4.456, 4.718, 4.681, 4.875, 4.544, 4.49, 4.634,
         4.24, 4.668, 4.738, 4.767]
    # The unloading times θ_U_i for each task, sorted by task priority α_i
    Z = [0.761, 1.01, 1.435, 1.122, 1.16, 0.834, 0.77, 1.806, 0.608, 1.328, 1.163, 0.258, 1.091, 0.739, 0.56, 0.529,
         0.988, 0.388, 0.251, 0.445, 1.56, 0.269, 0.464, 1.586, 1.013, 1.989, 1.96, 1.626, 0.961, 1.976, 1.487, 1.542,
         1.32, 1.146, 1.599, 0.619, 1.168, 1.857, 0.905, 1.37, 0.297, 1.371, 0.685, 1.158, 1.047, 0.289, 0.826, 1.576,
         0.918, 0.878, 0.573, 0.982, 1.899, 0.889, 1.661, 1.437, 0.219, 0.402, 0.927, 1.718, 1.356, 0.57, 0.265, 0.982,
         0.352, 0.696, 0.313, 1.43, 0.268, 0.97, 0.544, 1.392, 1.67, 0.431, 0.372, 0.296, 1.959, 1.994, 1.388, 0.615,
         1.548, 1.121, 1.123, 0.364, 0.501, 1.487, 1.325, 1.932, 0.715, 1.215, 0.658, 0.917, 0.884, 1.486, 1.09, 1.655,
         1.646, 1.955, 1.374, 1.604]
    theta_str = str(theta)
    T_str = str(T)
    Z_str = str(Z)
    input_data = {
        'n': n,
        'm': m,
        'b': b,
        't0': t0,
        'theta_sorted': theta_str,
        'T_sorted': T_str,
        'Z_sorted': Z_str
    }
    optimal_time_limit = predict_optimal_time_limit(input_data)
    print(f"Predicted optimal Time_limit: {optimal_time_limit}")