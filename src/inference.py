import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function for making predictions and plotting results
def test_model(model_path, test_data, window_size, num_runs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=1, architecture=[256, 128, 64, 32])  # 3-layer model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    log_transform = lambda x: np.log(x + 1e-8)
    inverse_log_transform = lambda x: np.exp(x) - 1e-8

    unique_runs = np.random.choice(test_data['run'].unique(), num_runs, replace=False)

    # Calculate subplot grid dimensions
    num_plots = len(unique_runs)
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), sharey='row')

    axs = axs.flatten() if num_rows * num_cols > 1 else [axs]
    handles, labels = [], []

    for i, run in enumerate(unique_runs):
        run_data = test_data[test_data['run'] == run]
        scaled_run_data = run_data[['prev_true', 'EIR_true', 'incall']].apply(log_transform)
        X_test_scaled, y_test_scaled = create_sequences(scaled_run_data, window_size)

        X_test_scaled = X_test_scaled.to(device)
        with torch.no_grad():
            test_predictions_scaled = model(X_test_scaled.unsqueeze(-1)).cpu().numpy()

        # Unscale the predictions for plotting
        y_test_unscaled = inverse_log_transform(y_test_scaled.numpy())
        test_predictions_unscaled = inverse_log_transform(test_predictions_scaled)

        # Extract the time column and scale it
        time_column = run_data['t'].values[:len(test_predictions_scaled)]
        time_in_years = time_column / 365.25

        # Discard points before 10 years
        valid_indices = np.where(time_in_years >= 10)[0]
        time_in_years = time_in_years[valid_indices]
        y_test_scaled = y_test_scaled.numpy()[valid_indices]
        y_test_unscaled = y_test_unscaled[valid_indices]
        test_predictions_scaled = test_predictions_scaled[valid_indices]
        test_predictions_unscaled = test_predictions_unscaled[valid_indices]

        # Compute error metrics for **scaled** test data
        mse_eir = mean_squared_error(y_test_scaled[:, 0], test_predictions_scaled[:, 0])
        mae_eir = mean_absolute_error(y_test_scaled[:, 0], test_predictions_scaled[:, 0])
        r2_eir = r2_score(y_test_scaled[:, 0], test_predictions_scaled[:, 0])

        mse_inc = mean_squared_error(y_test_scaled[:, 1], test_predictions_scaled[:, 1])
        mae_inc = mean_absolute_error(y_test_scaled[:, 1], test_predictions_scaled[:, 1])
        r2_inc = r2_score(y_test_scaled[:, 1], test_predictions_scaled[:, 1])

        # Define a professional color palette
        true_color = "black"
        pred_color_eir = "#ff7f0e"  # Bright Orange for EIR_true prediction
        pred_color_inc = "#d62728"  # Muted Red for Incidence prediction

        # Plot true and predicted values with distinct colors (using unscaled predictions for plot)
        axs[i].plot(time_in_years, y_test_unscaled[:, 0], label="True EIR_true", color=true_color, linestyle='-')
        axs[i].plot(time_in_years, test_predictions_unscaled[:, 0], label="Predicted EIR_true", color=pred_color_eir, linestyle='--')

        axs[i].plot(time_in_years, y_test_unscaled[:, 1], label="True Incidence", color=true_color, linestyle='-')
        axs[i].plot(time_in_years, test_predictions_unscaled[:, 1], label="Predicted Incidence", color=pred_color_inc, linestyle='--')

        # Set y-axis to log scale
        axs[i].set_yscale('log')

        # Set x-axis label
        axs[i].set_xlabel("Years")

        # Display title with metrics beside it
        axs[i].set_title(
            f'Run {run}  |  MSE: {mse_eir:.3f}, MAE: {mae_eir:.3f}, R²: {r2_eir:.3f}',
            fontsize=10, loc="left"
        )

        # Enable only vertical grid lines
        axs[i].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

        # Collect handles and labels for universal legend
        if i == 0:
            handles, labels = axs[i].get_legend_handles_labels()

    # Universal plot title
    fig.suptitle("EIR and Incidence Predictions for Observation Period:Starting_at_Equilibrium", fontsize=16, fontweight="bold", y=0.96)

    # Remove any empty subplots
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    # Add universal legend
    fig.legend(handles, labels, loc='upper center', ncol=4)

    # Adjust layout and display/save plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plots/test_predictions_at_Equilibrium_4layers.png")
    plt.show()



