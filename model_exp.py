import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# Function to create model
def create_experiment_model(seq_length, architecture):
    model = Sequential([Input(shape=(seq_length, 2))])
    for neurons in architecture:
        model.add(LSTM(neurons, return_sequences=True, activation='tanh'))
    model.layers[-1].return_sequences = False  # Last LSTM should not return sequences
    model.add(Dense(2))  # Output layer
    model.compile(optimizer='adam', loss='mse')  # Mean Squared Error loss
    return model