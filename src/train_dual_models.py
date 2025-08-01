import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.preprocessing import process_dataframe
from src.sequence_creator import create_sequences
from src.model_exp import LSTMModel
from src import util, evaluate


def create_sequences_with_baseline(data: pd.DataFrame, window_size: int):
    """Create sequences using ANC prevalence and the run's starting prevalence."""
    xs, ys = [], []
    seq_len = 2 * window_size + 1
    has_targets = {'EIR_true', 'incall'}.issubset(data.columns)

    for run in data['run'].unique():
        run_df = data[data['run'] == run].reset_index(drop=True)
        start_prev = run_df.loc[0, 'prev_true']
        prev_values = run_df['prev_true'].values
        for i in range(len(run_df) - window_size):
            if i < window_size:
                pad = np.repeat(prev_values[0], window_size - i)
                seq_vals = np.concatenate((pad, prev_values[: i + window_size + 1]))
            else:
                seq_vals = prev_values[i - window_size : i + window_size + 1]
            baseline_feat = np.full(seq_len, start_prev)
            seq = np.stack((seq_vals, baseline_feat), axis=1)
            xs.append(seq)
            if has_targets:
                ys.append(run_df.loc[i, ['EIR_true', 'incall']].values)

    xs = torch.tensor(np.array(xs, dtype=np.float32))
    if has_targets:
        ys = torch.tensor(np.array(ys, dtype=np.float32))
        return xs, ys
    return xs, None


def train(model, train_loader, eval_loader, epochs=25, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    loss_hist, eval_hist = [], []
    start = time.time()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        loss_hist.append(running / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xv, yv in eval_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                val_loss += criterion(model(Xv), yv).item()
        eval_hist.append(val_loss / len(eval_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_hist[-1]:.4f}, Eval Loss: {eval_hist[-1]:.4f}")

    duration = time.time() - start
    return loss_hist, eval_hist, duration


def main():
    parser = argparse.ArgumentParser(description="Train baseline and start-prevalence LSTM models")
    parser.add_argument("--csv", required=True, help="CSV file of simulation output")
    parser.add_argument("--window", type=int, default=10, help="Sequence window size")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.csv)
    df = process_dataframe(raw_df)

    log_transform = lambda x: np.log(x + 1e-8)
    df[['prev_true', 'EIR_true', 'incall']] = df[['prev_true', 'EIR_true', 'incall']].apply(log_transform)

    baseline_X, targets = create_sequences(df, args.window)
    baseline_X = baseline_X.unsqueeze(-1)
    start_X, _ = create_sequences_with_baseline(df, args.window)

    idx = np.arange(len(targets))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)

    def loaders(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch, shuffle=False)
        return train_loader, val_loader, X_train, X_val, y_train, y_val

    bl_train_loader, bl_val_loader, X_tr_b, X_val_b, y_tr, y_val = loaders(baseline_X)
    sp_train_loader, sp_val_loader, X_tr_s, X_val_s, _, _ = loaders(start_X)

    baseline_model = LSTMModel(input_size=1, architecture=[256, 128, 64, 32])
    bl_loss, bl_eval, bl_duration = train(baseline_model, bl_train_loader, bl_val_loader, epochs=args.epochs)
    torch.save(baseline_model.state_dict(), "src/trained_model/baseline_model.pth")

    start_model = LSTMModel(input_size=2, architecture=[256, 128, 64, 32])
    sp_loss, sp_eval, sp_duration = train(start_model, sp_train_loader, sp_val_loader, epochs=args.epochs)
    torch.save(start_model.state_dict(), "src/trained_model/start_prev_model.pth")

    results = [
        {"name": "Baseline", "model": baseline_model, "loss_history": bl_loss, "duration": bl_duration},
        {"name": "With Start Prev", "model": start_model, "loss_history": sp_loss, "duration": sp_duration},
    ]

    util.plot_training_metrics(results)
    util.plot_model_comparison(results)
    evaluate.plot_performance_metrics(results, X_tr_b.squeeze(-1), y_tr, X_val_b.squeeze(-1), y_val)


if __name__ == "__main__":
    main()
