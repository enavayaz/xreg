import numpy as np
import jax
from typing import Optional, List, Tuple, Dict


def pred(
        Y_test: List[np.ndarray],
        model,
        n_learn: int = 1,
        n_pred: int = 1,
        iterative: bool = False,
        ensemble_strategy=None,
        prnt: bool = True
) -> Tuple[List[np.ndarray], dict]:
    """
    Predict manifold-valued trajectories using a fitted model.

    Parameters
    ----------
    Y_test : list of arrays, each shape (n_obs, point_dim)
    model  : fitted model with fit() and predict() methods
    n_learn: minimum observations before first prediction
    n_pred : steps ahead to predict
    iterative : whether to use iterative (closed-loop) prediction
    ensemble_strategy : optional post-processing strategy
    prnt   : print summary if True

    Returns
    -------
    Y_pred  : list of predicted arrays
    metrics : dict with keys 'Y_pred', 'MAE', 'mae'
                - MAE : list of per-track error arrays
                - mae : float, mean over all forecasts
    """
    n_test   = len(Y_test)
    M        = model.M
    dist_fn  = jax.jit(jax.vmap(M.metric.dist, in_axes=(0, 0)))
    strategy = ensemble_strategy if ensemble_strategy is not None else model.ensemble_strategy

    Y_pred = [None] * n_test
    MAE    = [None] * n_test

    for k in range(n_test):
        y_test      = Y_test[k]
        n_forecasts = len(y_test) - n_learn - n_pred + 1
        y_pred      = np.empty((n_forecasts,) + M.point_shape)
        y_preds_track = []

        for n in range(n_learn, len(y_test) - n_pred + 1):
            y_learn = y_test[:n]
            len_x   = n + n_pred
            x       = np.linspace(0.0, 1.0, len_x)

            fitted        = model.fit(x[:n], y_learn)
            p            = np.array(fitted.predict(x[n:len_x], iterative=iterative))
            target_pred  = p[n_pred - 1]

            if strategy is not None:
                history_length = getattr(strategy, 'min_history', 2)
                if len(y_learn) >= history_length:
                    target_pred = strategy.adjust(
                        model.M,
                        y_learn[-history_length:],
                        target_pred,
                        np.array(y_preds_track)
                    )

            y_pred[n - n_learn] = target_pred
            y_preds_track.append(target_pred)

        y_true     = y_test[n_learn + n_pred - 1:]
        MAE[k]     = dist_fn(y_true, y_pred)
        Y_pred[k]  = y_pred

    all_mae = np.concatenate(MAE)
    mean_mae = float(np.mean(all_mae))
    std_mae = float(np.std(all_mae))
    if prnt:
        print('=' * 60)
        print(f'MAE: {mean_mae:.4f} +/- {std_mae:.4f}')
        print('=' * 60)

    metrics = {
        'MAE': MAE,
        'mae': mean_mae,
        'std': std_mae
    }
    return Y_pred, metrics


def pred_grid_search(
        Y_val, model_fn, lambda_grid,
        n_learn=1, n_pred=1, iterative=False, prnt=True
):
    results = {}
    for lam in lambda_grid:
        model = model_fn(lam)
        _, metrics = pred(Y_val, model, n_learn=n_learn, n_pred=n_pred,
                          iterative=iterative, prnt=False)
        results[lam] = metrics
        if prnt:
            print(f'  lambda={lam:.0e}  val MAE={metrics["mae"]:.4f}')
    return results