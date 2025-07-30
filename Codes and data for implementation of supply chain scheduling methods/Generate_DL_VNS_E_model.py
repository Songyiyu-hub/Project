import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import pickle
from ast import literal_eval

data = pd.read_csv('VNS_results.csv')
X = data[['n', 'm', 'b', 't0', 'theta_sorted', 'T_sorted', 'Z_sorted']].values
y_time_limit = data['Time_limit'].values
y_Cmax = data['Cmax'].values

def process_list_column(data, column_name):
    data[column_name] = data[column_name].apply(literal_eval)
    data[column_name + '_mean'] = data[column_name].apply(lambda x: np.mean(x))
    data[column_name + '_max'] = data[column_name].apply(lambda x: np.max(x))
    data[column_name + '_min'] = data[column_name].apply(lambda x: np.min(x))
    data[column_name + '_std'] = data[column_name].apply(lambda x: np.std(x))
    return data

data = process_list_column(data, 'theta_sorted')
data = process_list_column(data, 'T_sorted')
data = process_list_column(data, 'Z_sorted')

X = data[['n', 'm', 'b', 't0',
          'theta_sorted_mean', 'theta_sorted_max', 'theta_sorted_min', 'theta_sorted_std',
          'T_sorted_mean', 'T_sorted_max', 'T_sorted_min', 'T_sorted_std',
          'Z_sorted_mean', 'Z_sorted_max', 'Z_sorted_min', 'Z_sorted_std']].values

X_train_val, X_test, y_train_val_time_limit, y_test_time_limit, y_train_val_Cmax, y_test_Cmax = train_test_split(
    X, y_time_limit, y_Cmax, test_size=0.2, random_state=42)

X_train, X_val, y_train_time_limit, y_val_time_limit, y_train_Cmax, y_val_Cmax = train_test_split(
    X_train_val, y_train_val_time_limit, y_train_val_Cmax, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.joblib')
time_limit_cmax_dict = {}

grouped = data.groupby(['n', 'm'])
for name, group in grouped:
    n, m = name
    min_cmax = group['Cmax'].min()
    tolerance = min_cmax
    filtered_group = group[group['Cmax'] <= tolerance]
    if not filtered_group.empty:
        min_time_limit = filtered_group['Time_limit'].min()
        row = filtered_group[filtered_group['Time_limit'] == min_time_limit].iloc[0]
        features = {
            'n': n,
            'm': m,
            'b': row['b'],
            't0': row['t0'],
            'theta_sorted_mean': row['theta_sorted_mean'],
            'theta_sorted_max': row['theta_sorted_max'],
            'theta_sorted_min': row['theta_sorted_min'],
            'theta_sorted_std': row['theta_sorted_std'],
            'T_sorted_mean': row['T_sorted_mean'],
            'T_sorted_max': row['T_sorted_max'],
            'T_sorted_min': row['T_sorted_min'],
            'T_sorted_std': row['T_sorted_std'],
            'Z_sorted_mean': row['Z_sorted_mean'],
            'Z_sorted_max': row['Z_sorted_max'],
            'Z_sorted_min': row['Z_sorted_min'],
            'Z_sorted_std': row['Z_sorted_std']
        }
        key = (n, m)
        time_limit_cmax_dict[key] = {
            'Time_limit': min_time_limit,
            'Features': features
        }
with open('time_limit_cmax_dict.pkl', 'wb') as f:
    pickle.dump(time_limit_cmax_dict, f)
print("Dictionary has been updated and saved to time_limit_cmax_dict.pkl")

def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0004), loss='mae')
    return model

pretrained_model = build_model()
pretrained_model.fit(X_train, y_train_Cmax, epochs=30, batch_size=16, verbose=1)

def reptile_meta_train(base_model, X_meta, y_meta_time_limit, y_meta_Cmax, num_tasks=5, meta_lr=0.01, inner_lr=0.001,
                       inner_steps=5, epochs=30):
    meta_model = tf.keras.models.clone_model(base_model)
    meta_model.set_weights(base_model.get_weights())
    for epoch in range(epochs):
        for _ in range(num_tasks):
            task_indices = np.random.choice(len(X_meta), size=32, replace=False)
            X_task, y_task_time_limit, y_task_Cmax = X_meta[task_indices], y_meta_time_limit[task_indices], y_meta_Cmax[
                task_indices]
            model_clone = tf.keras.models.clone_model(meta_model)
            model_clone.set_weights(meta_model.get_weights())
            model_clone.compile(optimizer=Adam(learning_rate=inner_lr), loss='mse')
            model_clone.fit(X_task, y_task_Cmax, epochs=inner_steps, verbose=0)
            meta_weights = meta_model.get_weights()
            new_weights = model_clone.get_weights()
            meta_weights = [w + meta_lr * (nw - w) for w, nw in zip(meta_weights, new_weights)]
            meta_model.set_weights(meta_weights)
    return meta_model

meta_model = reptile_meta_train(pretrained_model, X_train, y_train_time_limit, y_train_Cmax)
meta_model.save('time_limit_predictor.keras')