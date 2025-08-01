import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

sys.path.append(os.path.dirname(__file__))
from sequence_creator import create_sequences


def load_example(window_size: int = 10, test_size: float = 0.2, random_state: int = 42):
    """Create train/test tensors from the sample compendium.

    Parameters
    ----------
    window_size : int
        How many time steps to include before the prediction point.
    test_size : float
        Fraction of data reserved for evaluation.
    random_state : int
        Seed for the random split.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``X_train``, ``X_test``, ``y_train``, ``y_test`` tensors ready for model training.
    """
    path = "data/sim_compendia_test/sims_compendium_test_vol_0.1.csv"
    df = pd.read_csv(path)

    # keep only required columns
    df = df[["run", "prev_true", "EIR_true", "incall"]]

    # log-transform to match model training conventions
    log = lambda x: np.log(x + 1e-8)
    df[["prev_true", "EIR_true", "incall"]] = df[["prev_true", "EIR_true", "incall"]].apply(log)

    X_all, y_all = [], []
    for run_id in df["run"].unique():
        run_df = df[df["run"] == run_id].reset_index(drop=True)
        X_run, y_run = create_sequences(run_df, window_size)
        X_all.append(X_run)
        y_all.append(y_run)

    X = torch.cat(X_all)
    y = torch.cat(y_all)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)


def main():
    X_train, X_test, y_train, y_test = load_example()
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))


if __name__ == "__main__":
    main()
