import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import torch


# Function to calculate metrics for each model
def calculate_metrics(model, test_data, window_size, device):
    log_transform = lambda x: np.log(x + 1e-8)
    inverse_log_transform = lambda x: np.exp(x) - 1e-8
    
    scaled_test_data = test_data[['prev_true', 'EIR_true', 'incall']].apply(log_transform)
    X_test_scaled, y_test_scaled = create_sequences(scaled_test_data, window_size)
    
    X_test_scaled = X_test_scaled.to(device)
    with torch.no_grad():
        test_predictions_scaled = model(X_test_scaled.unsqueeze(-1)).cpu().numpy()
    
    #y_test_unscaled = inverse_log_transform(y_test_scaled.numpy())
    #test_predictions_unscaled = inverse_log_transform(test_predictions_scaled)
    
    mse = mean_squared_error(y_test_scaled, test_predictions_scaled)
    mae = mean_absolute_error(y_test_scaled, test_predictions_scaled)
    r2 = r2_score(y_test_scaled, test_predictions_scaled)
    
    return mse, mae, r2


# Function to plot metrics per model
def plot_performance_metrics(test_data, window_size, model_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    metrics = {'Model': [], 'MSE': [], 'MAE': [], 'R²': []}
    
    for model_name, model_path in model_paths.items():
        architecture = model_architectures[model_name]
        model = LSTMModel(input_size=1, architecture=architecture)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        mse, mae, r2 = calculate_metrics(model, test_data, window_size, device)
        
        metrics['Model'].append(model_name)
        metrics['MSE'].append(mse)
        metrics['MAE'].append(mae)
        metrics['R²'].append(r2)
    
    df = pd.DataFrame(metrics)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = ['MSE', 'MAE', 'R²']
    
    for i, metric in enumerate(metrics_to_plot):
        sns.boxplot(x='Model', y=metric, data=df, ax=axes[i])
        axes[i].set_title(f'{metric} Across Models on Test Set: Equilibrium')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    #fig.suptitle("Metrics Comparison on Test Set", fontsize=16)
    plt.savefig("plot/model_performance/box_plot_test_predictions_at_Equilibrium.png")
    plt.show()



