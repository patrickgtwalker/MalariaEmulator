import numpy as np

def create_sequences(data, window_size):
    xs, ys = [], []
    
    for i in range(len(data) - window_size):
        if i < window_size:
            # When i is less than the window size, pad the beginning of the sequence
            pad_size = window_size - i
            # Retrieve and replicate both 'prev_true' and 'prev_2to10' in one step
            first_values = data.iloc[0][['prev_true', 'prev_2to10']].values
            replicated_values = np.tile(first_values, (pad_size, 1))
            # Combine padded values with actual data
            x_values = np.concatenate((replicated_values, data.iloc[0:i + window_size + 1][['prev_true', 'prev_2to10']].values), axis=0)
        else:
            # Extract values for both 'prev_true' and 'prev_2to10' without padding
            x_values = data.iloc[i - window_size:i + window_size + 1][['prev_true', 'prev_2to10']].values
        
        # Target variables: EIR_true and inc_true
        y = data.iloc[i][['EIR_true', 'incall']].values
        
        xs.append(x_values)
        ys.append(y)
    
    return np.array(xs), np.array(ys)
