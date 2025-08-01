


#Inferencing with uncertainty estimation - Not yet examined


def enable_mc_dropout(model):
    """ Enables dropout during inference """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def monte_carlo_dropout(model, X_test, num_samples=50):
    """ Perform MC Dropout to estimate uncertainty """
    enable_mc_dropout(model)
    preds = torch.stack([model(X_test.unsqueeze(-1)).detach().cpu() for _ in range(num_samples)])
    mean_pred = preds.mean(dim=0)  # Mean prediction
    std_pred = preds.std(dim=0)  # Standard deviation (uncertainty)
    return mean_pred, std_pred

def test_model(model_path, test_data, window_size, num_runs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=1, architecture=[256, 128, 64, 32])  # 4-layer model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    log_transform = lambda x: np.log(x + 1e-8)
    inverse_log_transform = lambda x: np.exp(x) - 1e-8

    unique_runs = np.random.choice(test_data['run'].unique(), num_runs, replace=False)
    num_plots = len(unique_runs)
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), sharey='row')
    axs = axs.flatten() if num_rows * num_cols > 1 else [axs]
    handles, labels = [], []

    for i, run in enumerate(unique_runs):
        run_data = test_data[test_data['run'] == run]
        scaled_run_data = run_data[['prev_true', 'EIR_true', 'incall']].apply(log_transform)
        X_test_scaled, y_test_scaled = create_sequences(scaled_run_data, window_size)

        X_test_scaled = X_test_scaled.to(device)
        with torch.no_grad():
            mean_pred_scaled, std_pred_scaled = monte_carlo_dropout(model, X_test_scaled, num_samples=50)

        y_test_unscaled = inverse_log_transform(y_test_scaled.numpy())
        mean_pred_unscaled = inverse_log_transform(mean_pred_scaled.numpy())
        std_pred_unscaled = inverse_log_transform(std_pred_scaled.numpy())

        time_column = run_data['t'].values[:len(mean_pred_scaled)]
        time_in_years = time_column / 365.25
        valid_indices = np.where(time_in_years >= 10)[0]
        time_in_years = time_in_years[valid_indices]
        y_test_unscaled = y_test_unscaled[valid_indices]
        mean_pred_unscaled = mean_pred_unscaled[valid_indices]
        std_pred_unscaled = std_pred_unscaled[valid_indices]

        true_color = "black"
        pred_color_eir = "#ff7f0e"
        pred_color_inc = "#d62728"

        axs[i].plot(time_in_years, y_test_unscaled[:, 0], label="True EIR_true", color=true_color, linestyle='-')
        axs[i].plot(time_in_years, mean_pred_unscaled[:, 0], label="Predicted EIR_true", color=pred_color_eir, linestyle='--')
        axs[i].fill_between(time_in_years, mean_pred_unscaled[:, 0] - std_pred_unscaled[:, 0],
                            mean_pred_unscaled[:, 0] + std_pred_unscaled[:, 0], color=pred_color_eir, alpha=0.2)
        
        axs[i].plot(time_in_years, y_test_unscaled[:, 1], label="True Incidence", color=true_color, linestyle='-')
        axs[i].plot(time_in_years, mean_pred_unscaled[:, 1], label="Predicted Incidence", color=pred_color_inc, linestyle='--')
        axs[i].fill_between(time_in_years, mean_pred_unscaled[:, 1] - std_pred_unscaled[:, 1],
                            mean_pred_unscaled[:, 1] + std_pred_unscaled[:, 1], color=pred_color_inc, alpha=0.2)
        
        axs[i].set_yscale('log')
        axs[i].set_xlabel("Years")
        axs[i].set_title(f'Run {run}', fontsize=10, loc="left")
        axs[i].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        
        if i == 0:
            handles, labels = axs[i].get_legend_handles_labels()
    
    fig.suptitle("EIR and Incidence Predictions with Uncertainty", fontsize=16, fontweight="bold", y=0.96)
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])
    
    fig.legend(handles, labels, loc='upper center', ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plots/test_prediction/test_predictions_with_uncertainty_4layers.png")
    plt.show()

test_model("src/trained_model/4_layers_model.pth", test_data, window_size=10, num_runs=20)