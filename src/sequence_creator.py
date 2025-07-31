import pandas as pd
import numpy as np
import torch

def create_sequences(data, window_size):
    xs, ys = [], []
    has_targets = all(col in data.columns for col in ['EIR_true', 'incall'])  # Check if target columns exist

    for i in range(len(data) - window_size):
        if i < window_size:
            # Pad beginning of sequence
            pad_size = window_size - i
            first_values = data.iloc[0][['prev_true']].values
            replicated_values = np.tile(first_values, (pad_size, 1))
            x_values = np.concatenate((replicated_values, data.iloc[0:i + window_size + 1][['prev_true']].values), axis=0)
        else:
            x_values = data.iloc[i - window_size:i + window_size + 1][['prev_true']].values

        xs.append(x_values.flatten())

        if has_targets:
            y = data.iloc[i][['EIR_true', 'incall']].values
            ys.append(y)

    xs = np.array(xs, dtype=np.float32)

    if has_targets:
        ys = np.array(ys, dtype=np.float32)
        return torch.tensor(xs), torch.tensor(ys)
    else:
        return torch.tensor(xs), None  # Return None for ys if targets are missing





# def create_sequences(data, window_size):
#     xs, ys = [], []
#     for run, run_df in data.groupby('run'):
#         run_df = run_df.reset_index(drop=True)
#         prev_true = run_df['prev_true'].to_numpy()
#         max_i = len(prev_true) - window_size
#         for i in range(max_i):
#             left_window = prev_true[max(0, i - window_size):i]
#             pad_count = window_size - len(left_window)
#             left_window = np.pad(left_window, (pad_count, 0), 'constant', constant_values=prev_true[i])
#             sequence = np.concatenate((left_window, [prev_true[i]], prev_true[i + 1:i + window_size + 1]))
#             xs.append(sequence)
#             ys.append(run_df.loc[i, ['EIR_true', 'incall']].to_numpy())

#     # Convert lists to NumPy arrays before creating tensors
#     xs = np.array(xs, dtype=np.float32)
#     ys = np.array(ys, dtype=np.float32)

#     return torch.tensor(xs), torch.tensor(ys)


