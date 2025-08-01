import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.sequence_creator import create_sequences
from src.create_sequences_with_baseline import create_sequences_with_baseline
from src.model_exp import LSTMModel, train_model
from src.evaluate import plot_performance_metrics


def prepare_loaders(X, y, batch_size=512):
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(X_eval, y_eval), batch_size=batch_size)
    return train_loader, eval_loader, X_train, y_train, X_eval, y_eval


def main(args):
    df = pd.read_csv(args.csv)
    log_transform = lambda x: np.log(x + 1e-8)

    df_scaled = df[['run', 'prev_true', 'EIR_true', 'incall']].copy()
    df_scaled[['prev_true', 'EIR_true', 'incall']] = df_scaled[['prev_true', 'EIR_true', 'incall']].apply(log_transform)

    X_base, y_base = create_sequences(df_scaled[['prev_true', 'EIR_true', 'incall']], args.window)
    X_base_loaders = prepare_loaders(X_base, y_base, args.batch)

    X_with_base, y_with_base = create_sequences_with_baseline(df_scaled, args.window)
    X_with_loaders = prepare_loaders(X_with_base, y_with_base, args.batch)

    model_base = LSTMModel(input_size=1, architecture=[256, 128, 64, 32])
    model_base, _, _, _ = train_model(
        model_base, X_base_loaders[0], X_base_loaders[1], "baseline", epochs=args.epochs
    )

    model_with = LSTMModel(input_size=1, architecture=[256, 128, 64, 32])
    model_with, _, _, _ = train_model(
        model_with, X_with_loaders[0], X_with_loaders[1], "with_baseline", epochs=args.epochs
    )

    results = [
        {"name": "Baseline", "model": model_base},
        {"name": "With Start Prev", "model": model_with},
    ]

    plot_performance_metrics(results, X_base_loaders[2], X_base_loaders[3], X_base_loaders[4], X_base_loaders[5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--window", type=int, default=5, help="Half window size for sequences")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    main(parser.parse_args())
