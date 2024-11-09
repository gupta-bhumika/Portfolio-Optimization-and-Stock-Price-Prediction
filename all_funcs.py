import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, GRU
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import keras_tuner as kt
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch and preprocess stock data
def fetch_and_preprocess_stock_data():
    companies = ['INFY.NS', 'TCS.NS', 'TATAMOTORS.NS', 'MARUTI.NS',
                 'SUNPHARMA.NS', 'CIPLA.NS', 'ITC.NS', 'MARICO.NS', 'GOLDBEES.NS', 'BAJAJ-AUTO.NS']
    end_date = datetime.date.today()
    start_date = datetime.date(end_date.year - 10, end_date.month, end_date.day)
    stock_data = fetch_stock_data(companies, start_date, end_date)
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(stock_data), columns=stock_data.columns, index=stock_data.index)
    df = normalized_data.dropna()
    return df

# Fetch stock data
def fetch_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)['Close']
    return data

# Create dataset for model
def create_dataset(data_array, timesteps):
    data_array = data_array.to_numpy()
    X_3d = np.array([data_array[i-timesteps:i, 0:] for i in range(timesteps, len(data_array))])
    y_3d = np.array([data_array[i, 0:] for i in range(timesteps, len(data_array))])
    return X_3d, y_3d

# Save and load model and metrics
def save_model_and_metrics(model, data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_model_and_metrics(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Calculate metrics function
# def calculate_metrics(y_true, y_pred):
    # Ensure 1D shapes for y_true and y_pred
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.reshape(-1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.reshape(-1)

    if len(y_true) != len(y_pred):
        logger.error(f"Shape mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)} samples.")
        y_pred = y_pred[:len(y_true)]  # Adjust y_pred to match y_true if necessary
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R-squared': r_squared}
def calculate_metrics(y_true, y_pred):
    # Log shapes for debugging
    logger.info(f"Calculating metrics: y_true shape = {y_true.shape}, y_pred shape = {y_pred.shape}")

    # Ensure consistent length between y_true and y_pred
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]

    # Reshape if necessary to make them 1D
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R-squared': r_squared}

# Define CNN-LSTM model with EarlyStopping and caching
def cnn_lstm_model(X_train, y_train, X_test, y_test, company):
    logger.info(f"Starting CNN-LSTM model training for {company}...")

    model_file = f'cnn_lstm_model_{company}.pkl'
    cached_data = load_model_and_metrics(model_file)
    if cached_data:
        logger.info(f"Loaded cached CNN-LSTM model for {company}.")
        if 'predictions' in cached_data and 'metrics' in cached_data:
            return cached_data['predictions'], cached_data['metrics']
        else:
            logger.warning("Cached data does not contain expected keys. Retraining the model...")

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    try:
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping])
        logger.info(f"Training completed for CNN-LSTM model for {company}.")
    except Exception as e:
        logger.error(f"Error during training for {company}: {e}")
        return None, None

    predictions = model.predict(X_test)
    if predictions.shape != y_test.shape:
        logger.warning(f"Shape mismatch after prediction for {company}: Adjusting predictions.")
        predictions = predictions[:len(y_test)].reshape(y_test.shape)

    metrics = calculate_metrics(y_test, predictions)
    save_model_and_metrics(model, {'predictions': predictions, 'metrics': metrics}, model_file)
    return predictions, metrics

# Define GRU-CNN model with EarlyStopping and caching
def gru_cnn_model(X_train, y_train, X_test, y_test, company):
    logger.info(f"Starting GRU-CNN model training for {company}...")

    model_file = f'gru_cnn_model_{company}.pkl'
    cached_data = load_model_and_metrics(model_file)
    if cached_data:
        logger.info(f"Loaded cached GRU-CNN model for {company}.")
        if 'predictions' in cached_data and 'metrics' in cached_data:
            return cached_data['predictions'], cached_data['metrics']
        else:
            logger.warning("Cached data does not contain expected keys. Retraining the model...")

    model = Sequential()
    model.add(GRU(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    try:
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping])
        logger.info(f"Training completed for GRU-CNN model for {company}.")
    except Exception as e:
        logger.error(f"Error during training for {company}: {e}")
        return None, None

    predictions = model.predict(X_test)
    if predictions.shape != y_test.shape:
        logger.warning(f"Shape mismatch after prediction for {company}: Adjusting predictions.")
        predictions = predictions[:len(y_test)].reshape(y_test.shape)

    metrics = calculate_metrics(y_test, predictions)
    save_model_and_metrics(model, {'predictions': predictions, 'metrics': metrics}, model_file)
    return predictions, metrics

# Define XGBoost-LSTM model
# Define XGBoost-LSTM model
def xgboost_lstm_model(X_train, y_train, X_test, y_test, company):
    logger.info(f"Starting XGBoost-LSTM model training for {company}...")

    model_file = f'xgboost_lstm_model_{company}.pkl'
    cached_data = load_model_and_metrics(model_file)
    if cached_data:
        logger.info(f"Loaded cached XGBoost-LSTM model for {company}.")
        if 'predictions' in cached_data and 'metrics' in cached_data:
            return cached_data['predictions'], cached_data['metrics']
        else:
            logger.warning("Cached data does not contain expected keys. Retraining the model...")

    # LSTM Part
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm_model.add(Dense(y_train.shape[1]))
    lstm_model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    try:
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping])
        logger.info(f"LSTM part of XGBoost-LSTM training completed for {company}.")
    except Exception as e:
        logger.error(f"Error during LSTM training in XGBoost-LSTM model for {company}: {e}")
        return None, None

    lstm_predictions = lstm_model.predict(X_train)

    # XGBoost Part
    try:
        xgb_model = XGBRegressor()
        xgb_model.fit(lstm_predictions, y_train)
        predictions = xgb_model.predict(lstm_model.predict(X_test))
        if predictions.shape != y_test.shape:
            logger.warning(f"Shape mismatch after prediction for {company}: Adjusting predictions.")
            predictions = predictions[:len(y_test)].reshape(y_test.shape)

        metrics = calculate_metrics(y_test, predictions)
        save_model_and_metrics(xgb_model, {'predictions': predictions, 'metrics': metrics}, model_file)
        logger.info(f"XGBoost part of XGBoost-LSTM model completed for {company}.")
        return predictions, metrics
    except Exception as e:
        logger.error(f"Error during XGBoost training for {company}: {e}")
        return None, None

# Define Random Forest-LSTM model
def rf_lstm_model(X_train, y_train, X_test, y_test, company):
    logger.info(f"Starting RF-LSTM model training for {company}...")

    model_file = f'rf_lstm_model_{company}.pkl'
    cached_data = load_model_and_metrics(model_file)
    if cached_data:
        logger.info(f"Loaded cached RF-LSTM model for {company}.")
        if 'predictions' in cached_data and 'metrics' in cached_data:
            return cached_data['predictions'], cached_data['metrics']
        else:
            logger.warning("Cached data does not contain expected keys. Retraining the model...")

    # LSTM Part
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm_model.add(Dense(y_train.shape[1]))
    lstm_model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    try:
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping])
        logger.info(f"LSTM part of RF-LSTM training completed for {company}.")
    except Exception as e:
        logger.error(f"Error during LSTM training in RF-LSTM model for {company}: {e}")
        return None, None

    lstm_predictions = lstm_model.predict(X_train)

    # Random Forest Part
    try:
        rf_model = RandomForestRegressor()
        rf_model.fit(lstm_predictions, y_train)
        predictions = rf_model.predict(lstm_model.predict(X_test))
        if predictions.shape != y_test.shape:
            logger.warning(f"Shape mismatch after prediction for {company}: Adjusting predictions.")
            predictions = predictions[:len(y_test)].reshape(y_test.shape)

        metrics = calculate_metrics(y_test, predictions)
        save_model_and_metrics(rf_model, {'predictions': predictions, 'metrics': metrics}, model_file)
        logger.info(f"Random Forest part of RF-LSTM model completed for {company}.")
        return predictions, metrics
    except Exception as e:
        logger.error(f"Error during Random Forest training for {company}: {e}")
        return None, None

# Define function for building LSTM model for Keras Tuner with EarlyStopping
def build_lstm_model(hp, input_shape, output_units):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), input_shape=input_shape))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mse')
    return model

# Perform hyperparameter tuning with Keras Tuner
def perform_hyperparameter_tuning(X_train, y_train, X_test, y_test, company):
    logger.info(f"Starting hyperparameter tuning for LSTM model for {company}...")
    try:
        tuner_file = f'tuned_lstm_model_{company}.pkl'
        cached_data = load_model_and_metrics(tuner_file)
        if cached_data:
            logger.info(f"Loaded cached tuned LSTM model for {company}.")
            if 'model' in cached_data and 'metrics' in cached_data:
                return cached_data['model'], cached_data['metrics']

        input_shape = (X_train.shape[1], X_train.shape[2])
        output_units = y_train.shape[1]

        tuner = kt.RandomSearch(
            lambda hp: build_lstm_model(hp, input_shape, output_units),
            objective='val_loss',
            max_trials=5,
            executions_per_trial=3,
            directory='tuner_dir',
            project_name=f'lstm_hyperparam_tuning_{company}'
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_predictions = best_model.predict(X_test)

        if best_predictions.shape != y_test.shape:
            logger.warning(f"Shape mismatch after best model prediction for {company}: Adjusting predictions.")
            best_predictions = best_predictions[:, :y_test.shape[1]]

        best_metrics = calculate_metrics(y_test, best_predictions)

        save_model_and_metrics(best_model, {'hyperparameters': best_hyperparameters.values, 'metrics': best_metrics}, tuner_file)
        logger.info(f"Hyperparameter tuning completed for {company}.")
        return best_model, best_metrics
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning for {company}: {e}")
        return None, None

