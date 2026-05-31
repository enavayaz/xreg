import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#%env JAX_PLATFORM_NAME=gpu
#%env XLA_PYTHON_CLIENT_PREALLOCATE=false

import numpy as np
import numpy.linalg as lg
from morphomatics.manifold import Sphere, PowerManifold
from morphomatics.stats import ExponentialBarycenter
from timeseries.main import pred
from timeseries.model import Reg, RidgeReg, AVGEnsemble
from helpers.util_pred import cov_mat, fit_poly_dc
from helpers.util import load_hur, save_hur

np.random.seed(42)
M                           = Sphere()
DEGREE                      = 5
NOISE_STD = 0.05
DIAG_LOAD                   = 0.0
TEMPLATES                   = ['Exp2'] #['Exp1', 'Exp2']
LAM                         = [4.0, 0.6]  # set to None to run grid search (takes ~60 minutes)
LAG                         = True  # False = non-lagged regression (slightly lower MAE, but slower)
AVG                         = None  # None = pure ridge (alpha=1); AVGEnsemble(0.5) = post-process averaging
PRED_ARGS                   = {'n_learn': DEGREE + 1, 'n_pred': 1, 'iterative': True}
FAST                        = True  # if True, use fast projection of Euclidean mean, else intrinsic mean
LOAD_PATH                   = ROOT / 'datasets'  # reference datasets used in paper
SAVE_PATH                   = LOAD_PATH / 'generated'  # newly generated data saved here
print(f'covariance: sample + {DIAG_LOAD}*I (diagonal loading)')

def split_trjs(tmp, B, Y):
    if tmp == 'Exp1':
        # Train on all 31 trajectories from 2020
        Btrain, Ytrain = B[:31], Y[:31]

        # Validate on the last 21 trajectories of that 2020 pool (indices 10 to 30)
        Yval = Ytrain[10:]

        # Test on all 21 trajectories from 2021 (indices 31 to the end)
        Ytest = Y[31:]

    elif tmp == 'Exp2':
        # Train on the first 11 trajectories of 2021
        Btrain, Ytrain = B[31:42], Y[31:42]

        # Validate on the subsequent 5 trajectories of 2021
        Yval = Y[42:47]

        # Test on the final 5 trajectories of 2021
        Ytest = Y[47:]

    return Ytrain, Yval, Ytest, Btrain

def coarse_to_fine_grid_search(Y_val, model_fn, coarse_grid, n_refine=3, factor=4.0):
    """Two-stage coarse-to-fine grid search over lambda (ridge only)."""
    all_results = {}

    # Stage 1: coarse pass
    print(f'  Coarse pass: [{", ".join(f"{l:.1e}" for l in coarse_grid)}]')
    for lam in coarse_grid:
        _, m = pred(Y_val, model_fn(lam), **PRED_ARGS, prnt=False)
        all_results[lam] = m['mae']
        print(f'    lambda={lam:.1e}  val MAE={m["mae"]:.4f}')

    lam_coarse_best = min(all_results, key=all_results.get)

    # Stage 2: fine pass around best coarse lambda
    log_center = np.log10(lam_coarse_best)
    log_range  = np.log10(factor)
    fine_grid  = np.logspace(log_center - log_range, log_center + log_range, n_refine).tolist()
    fine_grid  = [l for l in fine_grid
                  if not any(abs(l - ev) / ev < 0.01 for ev in all_results)]

    if fine_grid:
        print(f'  Fine pass around lambda={lam_coarse_best:.1e}: [{", ".join(f"{l:.2e}" for l in fine_grid)}]')
        for lam in fine_grid:
            _, m = pred(Y_val, model_fn(lam), **PRED_ARGS, prnt=False)
            all_results[lam] = m['mae']
            print(f'    lambda={lam:.2e}  val MAE={m["mae"]:.4f}')

    return min(all_results, key=all_results.get), all_results

def compute_mean(B_train, P):
    if FAST:
        # Fast vectorized fallback: Compute Euclidean mean across curves, then project back onto Sphere
        B_arr = np.asarray(B_train)
        euclidean_mean = np.mean(B_arr, axis=0)  # Shape: (n_cp, 3)
        # Project each control point back to the surface of the unit sphere (normalize rows)
        norms = np.linalg.norm(euclidean_mean, axis=1, keepdims=True)
        mean_b = euclidean_mean / norms
    else:
        # Use intrinsic mean of B_train using Riemannian optimization
        Ex = ExponentialBarycenter()
        mean_b = Ex.compute(P, B_train, max_iter=20)
    return mean_b

def run_experiment(template, Y_train, Y_val, Y_test, B_train):
    print(f"\n{'='*60}")
    print(f'Experiment: {template}')
    print(f"{'='*60}")

    n_cp = np.shape(B_train[0])[0]
    dim  = 3
    P    = PowerManifold(M, n_cp)
    PRED_ARGS['n_learn'] = n_cp
    # covariance — diagonal loading for numerical safety before cov_intrinsic
    mean_b  = compute_mean(B_train, P)
    cov_b   = cov_mat(P.metric.log, B_train, mean_b) + DIAG_LOAD * np.eye(n_cp * dim)
    eigvals = np.sort(lg.eigvalsh(cov_b))[::-1]
    print(f'Eigenvalues: {np.round(eigvals, 6)}')

    # scale: noise^2 * M.dim / mean_dominant_eig
    # mean_dominant_eig already used for precondition of loss function
    # makes ridge_const = lam * scale, so lam=1 ~ theoretically balanced
    # dominant = above DIAG_LOAD (structural zeros lifted to DIAG_LOAD, still dominant below)
    # dominant eigenvalues: above numerical noise floor regardless of DIAG_LOAD
    dominant_eigs = eigvals[eigvals > 1e-6]
    scale_lam = NOISE_STD**2 * M.dim  # / np.mean(dominant_eigs)
    print(f'{len(dominant_eigs)} dominant eigs, mean_eig = {np.mean(dominant_eigs):.3e}')

    # model factory: ridge_const = lam * scale_lam passed to RidgeRegression
    def model_fn(lam):
        if lam == 0:
            return Reg(M, lag=LAG, degree=DEGREE)
        return RidgeReg(M, mean_b, cov_b, lam * scale_lam, lag=LAG, degree=DEGREE)

    # OLS on val and test
    _, m_ols_val = pred(Y_val,  model_fn(0), **PRED_ARGS, prnt=False)
    _, m_ols     = pred(Y_test, model_fn(0), **PRED_ARGS, prnt=False)
    print(f'  OLS  MAE: {m_ols["mae"]:.4f} +/- {m_ols["std"]:.4f}')
    if LAM is None:
        # grid search — lam_balanced ~ mean_eig^2 * n_cp / (M.dim * dist_to_mean^2) ~ 1e-3
        COARSE_GRID = [0.1, 0.5, 1.0, 5.0, 10.0]
        lam_star, val_results = coarse_to_fine_grid_search(
            Y_val, model_fn, COARSE_GRID, n_refine=3, factor=4.0
        )
        # fall back to OLS if ridge does not improve on val
        if m_ols_val['mae'] <= val_results[lam_star]:
            lam_star = 0
        print(f'  lambda* = {lam_star:.2e}  (ridge_const = {lam_star * scale_lam:.2e})')
    else:
        idx = TEMPLATES.index(template)
        lam_star = LAM[idx]
    # ridge on test
    _, m_ridge  = pred(Y_test, model_fn(lam_star), **PRED_ARGS, ensemble_strategy=AVG, prnt=False)
    improvement = 100 * (m_ols['mae'] - m_ridge['mae']) / m_ols['mae']
    print(f'  Ridge MAE: {m_ridge["mae"]:.4f} +/- {m_ridge["std"]:.4f}  (improvement: {improvement:.1f}%)')
    return {
        'ols_mean':    m_ols['mae'],
        'ols_std':     m_ols['std'],
        'ridge_mean':  m_ridge['mae'],
        'ridge_std':   m_ridge['std'],
        'lam_star':    lam_star,
        'scale_mu':    scale_lam,
        'improvement': improvement,
    }

B, Y = load_hur('hur', target_dir=LOAD_PATH)
all_data = {'Y': Y, 'B': B}
DEGREE = len(B[0]) - 1
print("Data ready to run experiments.")

d = all_data
results = {}
for template in TEMPLATES:
    Y_train, Y_val, Y_test, B_train = split_trjs(template, d['B'], d['Y'])
    results[template] = run_experiment(
        template,
        Y_train,
        Y_val,
        Y_test,
        B_train
    )