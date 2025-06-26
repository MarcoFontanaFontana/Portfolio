################################ NEURAL NETWORK BACKTESTER ################################

import random
import joblib
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from RALLY_global_variables import global_variables as gv
from RALLY_NN_trainer import NN_train

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if gv.TRAIN_BEFORE_TESTING:
    NN_train()

start_time = time.time()

# List of unseen data CSV filenames
test_files = gv.TEST_DATA
all_rel_errors = []  # To store average relative error for each CSV


def create_sequences(df, window_size, horizon=1):
    X, y = [], []
    max_i = len(df) - window_size - (horizon - 1)
    for i in range(max_i):
        seq = df.iloc[i : i + window_size][["openPrice", "highPrice", "lowPrice", "lastPrice", "volume"]].values
        X.append(seq)
        y.append(df.iloc[i + window_size + (horizon - 1)]["lastPrice"])
    return np.array(X), np.array(y)

for file in test_files:
    print(f"\n######################## Testing on {file} ########################")
    # Load unseen data
    
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    
    # Create sequences
    X, y_actual = create_sequences(df, gv.WINDOW_SIZE, gv.HORIZON)
    
    if len(X) == 0 or len(y_actual) == 0:
        print(f"❌ Not enough data in {file} to create sequences.")
        continue

    # Load saved scalers and model
    scaler_X = joblib.load("scaler_X.save")
    scaler_y = joblib.load("scaler_y.save")
    model = load_model(gv.NEURAL_NETWORK)
    
    # Scale input sequences
    X_scaled = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Predict and inverse-transform predictions
    predictions_scaled = model.predict(X_scaled, verbose=0)
    y_predicted = scaler_y.inverse_transform(predictions_scaled).flatten()


    # Compute errors
    errors = np.abs(y_predicted - y_actual)
    rel_errors = errors / y_actual
    mean_error = np.mean(errors)
    mean_rel_error = np.mean(rel_errors)
    mean_actual = np.mean(y_actual)
    mean_predicted = np.mean(y_predicted)
    
    print(f'Neural Network: {gv.NEURAL_NETWORK}  - Test set: {file}  -  Total predictions: {len(y_actual)}')  
    print(f'Av. actual price:                   ${mean_actual:.2f}')
    print(f'Av. predicted price:                ${mean_predicted:.2f}')
    print(f'Av. absolute error:                 ${mean_error:.2f}')
    print(f'Av. relative error:                 {(mean_rel_error)*100:.2f}%\n')
    

    all_rel_errors.append(mean_rel_error)

# Compute overall performance: arithmetic mean of relative errors from all files
overall_mean_rel_error = np.mean(all_rel_errors)
print(f"\nOverall average relative prediction error: {(overall_mean_rel_error)*100:.2f}%\n")

end_time = time.time()
elapsed_time = (end_time - start_time) / 60
print(f"Testing time: {elapsed_time:.2f} minutes")

