import numpy as np

#Sequence Creation
def create_sequences(data, window_size):
    xs, ys = [], []
    
    half_window_after = int(np.ceil(window_size / 2)) 
    
    # Iterating over the data at area of interest
    for i in range(len(data) - half_window_after):
        
        # When i is less than the window size, padding is done
        if i < window_size:
            pad_size = window_size - i
            first_values = data.iloc[0][['prev_true']].values
            replicated_values = np.tile(first_values, (pad_size, 1))
            
            # Portion of the sequence before and after the current timestep
            before_t = data.iloc[0:i][['prev_true']].values
            current_t = data.iloc[i][['prev_true']].values
            after_t = data.iloc[i+1:i + half_window_after + 1][['prev_true']].values
            
            # reshaping current_t to 2D 
            current_t = current_t.reshape(1, -1)  # Convert to 2D

            # Concatenating values
            x = np.concatenate((replicated_values, before_t, current_t, after_t), axis=0)
        else:
            #window before timestep `t`
            before_t = data.iloc[i-window_size:i][['prev_true']].values
            #current timestep `t`
            current_t = data.iloc[i][['prev_true']].values
            #window after timestep `t`
            after_t = data.iloc[i+1:i + half_window_after + 1][['prev_true']].values
            
            current_t = current_t.reshape(1, -1)  # Convert to 2D

            # Concatenating sequences
            x = np.concatenate((before_t, current_t, after_t), axis=0)

        # Target value `y` at the current timestep `t`
        y = data.iloc[i]['EIR_true']
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)