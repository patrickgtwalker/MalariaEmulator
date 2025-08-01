import numpy as np
import torch
import pandas as pd


def create_sequences_with_baseline(data: pd.DataFrame, window_size: int):
    """Create sequences with baseline prevalence.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing at least ``prev_true`` and ``run`` columns. Optional
        target columns ``EIR_true`` and ``incall`` will be returned if present.
    window_size : int
        Number of timesteps before and after the centre point to include.

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor | None)
        Sequence tensor of shape ``(n_samples, 2 * window_size + 2)`` including
        the starting prevalence appended to each sequence and target tensor if
        target columns exist.
    """

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    has_targets = {'EIR_true', 'incall'}.issubset(data.columns)

    for run, run_df in data.groupby('run'):
        run_df = run_df.reset_index(drop=True)
        baseline = run_df.loc[0, 'prev_true']
        for i in range(len(run_df) - window_size):
            if i < window_size:
                pad = window_size - i
                first = run_df.loc[0, 'prev_true']
                left = np.full(pad, first)
                seq = np.concatenate((left, run_df.loc[0:i + window_size, 'prev_true'].to_numpy()))
            else:
                seq = run_df.loc[i - window_size:i + window_size, 'prev_true'].to_numpy()
            seq_with_base = np.concatenate((seq, [baseline]))
            xs.append(seq_with_base.astype(np.float32))

            if has_targets:
                target = run_df.loc[i, ['EIR_true', 'incall']].to_numpy(dtype=np.float32)
                ys.append(target)

    x_tensor = torch.tensor(np.array(xs, dtype=np.float32))
    if has_targets:
        y_tensor = torch.tensor(np.array(ys, dtype=np.float32))
        return x_tensor, y_tensor
    return x_tensor, None
