import numpy as np
import pandas as pd
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Process inferences in batches (avoiding GPU memory issue)
def predict_in_batches(model, X, batch_size=512, device='cpu'):
    """Generate model predictions in manageable batches."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size].to(device)
            if X_batch.ndim == 2:  # Backwards compatibility with 1-D features
                X_batch = X_batch.unsqueeze(-1)
            preds = model(X_batch).cpu().numpy()
            predictions.append(preds)
    return np.vstack(predictions)


#Function to Plot Evaluation Metrics
def plot_performance_metrics(results, X_train, y_train, X_eval, y_eval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_train_pred_list = [
        predict_in_batches(result['model'], X_train, batch_size=512, device=device) for result in results
    ]

    y_eval_pred_list = [
        predict_in_batches(result['model'], X_eval, batch_size=512, device=device) for result in results
    ]

    metrics_data = []
    for idx, result in enumerate(results):
        y_train_pred = y_train_pred_list[idx]
        y_eval_pred = y_eval_pred_list[idx]

        train_r2 = r2_score(y_train, y_train_pred)
        eval_r2 = r2_score(y_eval, y_eval_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        eval_mse = mean_squared_error(y_eval, y_eval_pred)
        train_rmse = np.sqrt(train_mse)
        eval_rmse = np.sqrt(eval_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        eval_mae = mean_absolute_error(y_eval, y_eval_pred)

        metrics_data.append({
            "Model": result['name'],
            "Train R²": train_r2, "Eval R²": eval_r2,
            "Train MSE": train_mse, "Eval MSE": eval_mse,
            "Train RMSE": train_rmse, "Eval RMSE": eval_rmse,
            "Train MAE": train_mae, "Eval MAE": eval_mae
        })

    metrics_df = pd.DataFrame(metrics_data)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # R² Plot with value annotations
    ax = sns.barplot(data=metrics_df.melt(id_vars="Model", value_vars=["Train R²", "Eval R²"]),
                     x="Model", y="value", hue="variable", palette="viridis", ax=axes[0, 0])
    axes[0, 0].set_title("R² Score Comparison:Starting_at_Equilibrium")
    axes[0, 0].set_ylabel("R² Score")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=-25, color='white', fontsize =10)

    # MSE Plot
    sns.barplot(data=metrics_df.melt(id_vars="Model", value_vars=["Train MSE", "Eval MSE"]),
                x="Model", y="value", hue="variable", palette="magma", ax=axes[0, 1])
    axes[0, 1].set_title("Mean Squared Error (MSE) Comparison:Starting_at_Equilibrium")
    axes[0, 1].set_ylabel("MSE")

    # RMSE Plot
    sns.barplot(data=metrics_df.melt(id_vars="Model", value_vars=["Train RMSE", "Eval RMSE"]),
                x="Model", y="value", hue="variable", palette="coolwarm", ax=axes[1, 0])
    axes[1, 0].set_title("Root Mean Squared Error (RMSE) Comparison:Starting_at_Equilibrium")
    axes[1, 0].set_ylabel("RMSE")

    # MAE Plot
    sns.barplot(data=metrics_df.melt(id_vars="Model", value_vars=["Train MAE", "Eval MAE"]),
                x="Model", y="value", hue="variable", palette="cividis", ax=axes[1, 1])
    axes[1, 1].set_title("Mean Absolute Error (MAE) Comparison:Starting_at_Equilibrium")
    axes[1, 1].set_ylabel("MAE")

    fig.tight_layout()
    plt.savefig("plots/test_prediction/performance_metrics_25000runs_at_Equilibrium.png")
    plt.show()
