import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# Defining model arch.
def create_model(seq_length):
    model = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(200, return_sequences=True, activation='tanh'),
        LSTM(100, return_sequences=True, activation='tanh'),
        LSTM(50, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model