import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from all_funcs import (
    fetch_and_preprocess_stock_data, create_dataset, cnn_lstm_model,
    gru_cnn_model, xgboost_lstm_model, rf_lstm_model, calculate_metrics,
    perform_hyperparameter_tuning
)
import tensorflow as tf
import numpy as np
import random
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_seed()  # Call at the start of the script

# Fetch and preprocess data
st.title("Stock Price Prediction and Optimization Dashboard")

# Step 1: Display Dataset Head and Tail
data = fetch_and_preprocess_stock_data()
st.write("### Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.write("#### Data Head")
    st.dataframe(data.head(), use_container_width=True)
with col2:
    st.write("#### Data Tail")
    st.dataframe(data.tail(), use_container_width=True)

# Step 2: Line Graph of Stock Price Trends
st.write("### Stock Price Trend")
selected_companies = st.multiselect("Select Companies to View Trends", options=data.columns.tolist(), default=data.columns[:3].tolist())
if selected_companies:
    st.line_chart(data[selected_companies])

# Step 3: Create Dataset
timesteps = 30
companies = data.columns.tolist()

# Initialize dictionaries to store training and testing metrics
train_metrics_dict = {
    'MSE': {model: [] for model in ['CNN-LSTM', 'GRU-CNN', 'XGBoost-LSTM', 'RF-LSTM']},
    'RMSE': {model: [] for model in ['CNN-LSTM', 'GRU-CNN', 'XGBoost-LSTM', 'RF-LSTM']},
    'MAE': {model: [] for model in ['CNN-LSTM', 'GRU-CNN', 'XGBoost-LSTM', 'RF-LSTM']},
    'R-squared': {model: [] for model in ['CNN-LSTM', 'GRU-CNN', 'XGBoost-LSTM', 'RF-LSTM']}
}

metrics_dict = {
    'MSE': {model: [] for model in ['CNN-LSTM', 'GRU-CNN', 'XGBoost-LSTM', 'RF-LSTM']},
    'RMSE': {model: [] for model in ['CNN-LSTM', 'GRU-CNN', 'XGBoost-LSTM', 'RF-LSTM']},
    'MAE': {model: [] for model in ['CNN-LSTM', 'GRU-CNN', 'XGBoost-LSTM', 'RF-LSTM']},
    'R-squared': {model: [] for model in ['CNN-LSTM', 'GRU-CNN', 'XGBoost-LSTM', 'RF-LSTM']}
}

# Train and evaluate models for each company
for company in companies:
    # st.write(f"### Training and Evaluating for {company}")
    company_data = data[[company]]
    X, y = create_dataset(company_data, timesteps)
    
    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train and evaluate models
    models = {
        'CNN-LSTM': cnn_lstm_model,
        'GRU-CNN': gru_cnn_model,
        'XGBoost-LSTM': xgboost_lstm_model,
        'RF-LSTM': rf_lstm_model
    }
    
    for model_name, model_func in models.items():
        # Get predictions and metrics for testing data
        test_predictions, test_metrics = model_func(X_train, y_train, X_test, y_test, company)

        # Calculate training predictions and metrics
        train_predictions = model_func(X_train, y_train, X_train, y_train, company)[0]

        # Adjust training predictions to match y_train's length
        if len(train_predictions) != len(y_train):
            logger.warning(f"Adjusting training predictions from {len(train_predictions)} to {len(y_train)} for {company}")
            train_predictions = train_predictions[:len(y_train)]

        # Adjust test predictions to match y_test's length
        if len(test_predictions) != len(y_test):
            logger.warning(f"Adjusting test predictions from {len(test_predictions)} to {len(y_test)} for {company}")
            test_predictions = test_predictions[:len(y_test)]

        # Calculate training and testing metrics
        train_metrics = calculate_metrics(y_train, train_predictions)
        # st.write(f"#### {model_name} Training Metrics for {company}")
        # st.write(train_metrics)

        # st.write(f"#### {model_name} Testing Metrics for {company}")
        # st.write(test_metrics)

        # Store training and testing metrics in the dictionaries for further analysis
        for metric_name in train_metrics_dict.keys():
            train_metrics_dict[metric_name][model_name].append(train_metrics[metric_name])
            metrics_dict[metric_name][model_name].append(test_metrics[metric_name])

# Step 4: Display company-wise metrics tables for training metrics
st.write("## Training Metrics")
train_metric_tables = {}
for metric_name in train_metrics_dict.keys():
    train_metric_df = pd.DataFrame(train_metrics_dict[metric_name], index=companies)
    train_metric_df.index.name = 'Company'
    train_metric_tables[metric_name] = train_metric_df
    st.write(f"### {metric_name} Training Table")
    st.dataframe(train_metric_df, use_container_width=True)

# Display company-wise metrics tables for testing metrics
st.write("## Testing Metrics")
metric_tables = {}
for metric_name in metrics_dict.keys():
    metric_df = pd.DataFrame(metrics_dict[metric_name], index=companies)
    metric_df.index.name = 'Company'
    metric_tables[metric_name] = metric_df
    st.write(f"### {metric_name} Testing Table")
    st.dataframe(metric_df, use_container_width=True)


# Step 5: Plotting functions for metrics
def plot_bar_graph(metric_name, metric_df):
    fig = go.Figure()
    for model in metric_df.columns:
        fig.add_trace(go.Bar(
            x=metric_df.index,
            y=metric_df[model],
            name=model
        ))

    fig.update_layout(
        title=f'{metric_name} Comparison',
        xaxis_title='Company',
        yaxis_title=metric_name,
        barmode='group',
        width=1200,
        xaxis_tickangle=-45
    )
    return fig

# Plot bar graphs for MSE, RMSE, MAE, and R-squared
for metric_name, metric_df in metric_tables.items():
    st.write(f"### {metric_name} Bar Graph")
    bar_fig = plot_bar_graph(metric_name, metric_df)
    st.plotly_chart(bar_fig)

# Step 6: Hyperparameter Tuning
st.write("### Hyperparameter Tuning for LSTM Model")

# Call the tuning function
best_model, best_metrics = perform_hyperparameter_tuning(X_train, y_train, X_test, y_test, company)

# Check if the best model is None or invalid
if best_model is None:
    st.error("Hyperparameter tuning did not return a valid model. Please check the tuning process.")
else:
    # Display best hyperparameters if the model is valid
    best_hyperparameters = best_metrics.get('hyperparameters', {})
    st.write("Best Hyperparameters found and Tuned Model Metrics:")
    st.write(best_metrics)

    # Display or use `best_model` for predictions
    y_pred = best_model.predict(X_test).reshape(-1)  # Flatten if necessary
    st.write("Predictions using the Tuned Model:")
    st.write(y_pred)

    # Plot actual vs. predicted values
    st.write("### Actual vs. Predicted Values")
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.reshape(-1), label='Actual Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red')
    plt.title('Actual vs. Predicted Stock Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)

# Display Comparison of Hyperparameters Before and After Tuning
if best_model:
    st.write("### Hyperparameter Comparison: Before vs After Tuning")
    initial_hyperparameters = {
        'units': 50,  # Example initial hyperparameters
        'learning_rate': 0.001
    }

    # Ensure `best_hyperparameters` is available for comparison
    if 'units' in best_hyperparameters:
        hyperparam_comparison_df = pd.DataFrame({
            'Hyperparameter': ['Units', 'Learning Rate'],
            'Initial Value': [initial_hyperparameters['units'], initial_hyperparameters['learning_rate']],
            'Tuned Value': [best_hyperparameters['units'], best_hyperparameters.get('learning_rate', 0.001)]
        })
        st.table(hyperparam_comparison_df)
    else:
        st.warning("No hyperparameters found for comparison.")
