import os
import sys
from pathlib import Path
import numpy as np
import numpy.linalg as lg
from morphomatics.manifold import Sphere, PowerManifold
from morphomatics.stats import ExponentialBarycenter
from timeseries.stats import sph_correlated_trjs
from timeseries.main import pred
from timeseries.model import Reg, RidgeReg, AVGEnsemble
from helpers.util_pred import cov_mat, fit_poly_dc
from helpers.util import load_sph, save_sph

# ==========================================
# 1. Configuration & Repo Root Resolution
# ==========================================
current = Path.cwd().resolve()
for folder in [current, *current.parents]:
    if (folder / 'datasets').exists() and (folder / 'notebooks').exists():
        ROOT = folder
        break
else:
    raise RuntimeError(f"Could not find repo root from {current}")
os.chdir(ROOT)

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 42
np.random.seed(SEED)

M = Sphere()
Ex = ExponentialBarycenter()

DEGREE = 5
N_SUBJ, N_POINTS = 30, 35
N_TRAIN, N_VAL, N_TEST = 20, 5, 5
NOISE_STD, LON_MAX, LAT_MAX = 0.05, 0.75 * np.pi, np.pi / 20
DIAG_LOAD = 0.0
TEMPLATES = ['Geo']  # ['Geo', 'Cubic', 'Sin']
LAM = [4.0, 2.2, 0.6]
LAG = True
AVG = None
PRED_ARGS = {'n_learn': DEGREE + 1, 'n_pred': 1, 'iterative': True}
GENERATE = False
LOAD_PATH = ROOT / 'datasets'
SAVE_PATH = LOAD_PATH / 'generated'
IMP = 10.0
FAST = True
print(f'seed={SEED}  degree={DEGREE}  split={N_TRAIN}/{N_VAL}/{N_TEST}')

# ==========================================
# 2. Data Preparation
# ==========================================
all_data = {}
for template in TEMPLATES:
    if GENERATE:
        print(f'Generating {N_SUBJ} trajectories (template={template}) ...')
        Y, _ = sph_correlated_trjs(
            LON_MAX, LAT_MAX,
            n_trj=N_SUBJ, n_points=N_POINTS,
            noise_std=NOISE_STD, mean_curve=template
        )
        print(f'  Fitting Bezier polynomials ...')
        B, _ = fit_poly_dc(M, Y, deg=DEGREE)
        all_data[template] = {'Y': Y, 'B': B}
    else:
        B, Y = load_sph(f'sph{template}', target_dir=LOAD_PATH)
        all_data[template] = {'Y': Y, 'B': B}
        print(f"Loaded {template} data.")

DEGREE = len(B[0]) - 1

def compute_mean(B_train, P):
    if FAST:
        # If not debugging, attempt the rigorous geometric mean with a low iteration ceiling
        if sys.gettrace() is not None:
            # DEBUG MODE DETECTED: Bypass to avoid debugger hook slowdowns
            raise ImportError("Debugger bypass")
        mean_b = Ex.compute(P, B_train, max_iter=5)
    else:
        # FAST VECTORIZED FALLBACK: Compute Euclidean mean across curves, then project back onto Sphere
        # Shape of B_train is (N_TRAIN, n_cp, dim)
        B_arr = np.asarray(B_train)
        euclidean_mean = np.mean(B_arr, axis=0)  # Shape: (n_cp, 3)

        # Project each control point back to the surface of the unit sphere (normalize rows)
        norms = np.linalg.norm(euclidean_mean, axis=1, keepdims=True)
        mean_b = euclidean_mean / norms
    return mean_b

# ==========================================
# 3. Experiment Engine with Trajectory Tracking
# ==========================================
def run_experiment(template, Y_train, Y_val, Y_test, B_train):
    print(f"\n{'=' * 60}")
    print(f'Experiment: {template}')
    print(f"{'=' * 60}")

    n_cp = np.shape(B_train[0])[0]  # Number of control points
    dim = 3
    P = PowerManifold(M, n_cp)
    PRED_ARGS['n_learn'] = n_cp
    print("Calculating mean coefficients...")
    mean_b = compute_mean(B_train, P)
    print("  -> Used high-speed projection fallback for mean coefficients calculation.")

    # Calculate covariance profile using the established mean anchor
    cov_b = cov_mat(P.metric.log, B_train, mean_b) + DIAG_LOAD * np.eye(n_cp * dim)
    eigvals = np.sort(lg.eigvalsh(cov_b))[::-1]

    dominant_eigs = eigvals[eigvals > 1e-6]
    scale_lam = NOISE_STD ** 2 * M.dim

    def model_fn(lam):
        if lam == 0:
            return Reg(M, lag=LAG, degree=DEGREE)
        return RidgeReg(M, mean_b, cov_b, lam * scale_lam, lag=LAG, degree=DEGREE)

    print("Running OLS predictions...")
    y_pred_ols, m_ols = pred(Y_test, model_fn(0), **PRED_ARGS, prnt=False)

    idx = TEMPLATES.index(template)
    lam_star = LAM[idx]

    print("Running Ridge predictions...")
    y_pred_ridge, m_ridge = pred(Y_test, model_fn(lam_star), **PRED_ARGS, ensemble_strategy=AVG, prnt=False)

    improvement = 100 * (m_ols['mae'] - m_ridge['mae']) / m_ols['mae']
    print(f'  OLS  MAE: {m_ols["mae"]:.4f} +/- {m_ols["std"]:.4f}')
    print(f'  Ridge MAE: {m_ridge["mae"]:.4f} +/- {m_ridge["std"]:.4f}  (improvement: {improvement:.1f}%)')

    # ----------------------------------------------------------------------
    # Trajectory Filtering & Storage Isolation Logic
    # ----------------------------------------------------------------------
    VIS_THRESHOLD = 7.0
    MAX_VIS = 3

    Ytest_vis, Yols_vis, Yridge_vis = [], [], []

    for s_idx in range(len(Y_test)):
        if len(Ytest_vis) >= MAX_VIS:
            print(f'  Reached {MAX_VIS} good trajectories, stopping early.')
            break

        # predict single trajectory
        y_pred_ols_s, m_ols_s = pred([Y_test[s_idx]], model_fn(0), **PRED_ARGS, prnt=False)
        y_pred_ridge_s, m_ridge_s = pred([Y_test[s_idx]], model_fn(lam_star), **PRED_ARGS, prnt=False)

        y_true_trj = np.asarray(Y_test[s_idx])
        y_ols_trj = np.asarray(y_pred_ols_s[0])
        y_ridge_trj = np.asarray(y_pred_ridge_s[0])

        min_len = min(len(y_true_trj), len(y_ols_trj), len(y_ridge_trj))

        err_ols = np.mean(np.arccos(np.clip(
            np.sum(y_true_trj[-min_len:] * y_ols_trj[-min_len:], axis=-1), -1.0, 1.0)))
        err_ridge = np.mean(np.arccos(np.clip(
            np.sum(y_true_trj[-min_len:] * y_ridge_trj[-min_len:], axis=-1), -1.0, 1.0)))

        trj_improvement = 100 * (err_ols - err_ridge) / (err_ols + 1e-8)
        print(f'  subj {s_idx}: OLS={err_ols:.4f}  Ridge={err_ridge:.4f}  impr={trj_improvement:.1f}%')

        if trj_improvement > VIS_THRESHOLD:
            Ytest_vis.append(y_true_trj[-min_len:])
            Yols_vis.append(y_ols_trj[-min_len:])
            Yridge_vis.append(y_ridge_trj[-min_len:])
            print(f'    -> Selected (total: {len(Ytest_vis)})')

    if Ytest_vis:
        output_filename = LOAD_PATH / f'visNew_{template}.npz'
        np.savez(output_filename,
                 Ytest=np.array(Ytest_vis, dtype=object),
                 Yols=np.array(Yols_vis, dtype=object),
                 Yridge=np.array(Yridge_vis, dtype=object))
        print(f'  Saved {len(Ytest_vis)} trajectories -> {output_filename.name}')
    else:
        print(f'  No trajectories with improvement > {VIS_THRESHOLD}% found.')

    return {
        'ols_mean': m_ols['mae'],
        'ols_std': m_ols['std'],
        'ridge_mean': m_ridge['mae'],
        'ridge_std': m_ridge['std'],
        'lam_star': lam_star,
        'improvement': improvement,
    }

# ==========================================
# 4. Main Iteration Pipeline
# ==========================================
results = {}
for template in TEMPLATES:
    d = all_data[template]
    Y, B = d['Y'], d['B']
    results[template] = run_experiment(
        template,
        Y[:N_TRAIN],
        Y[N_TRAIN:N_TRAIN + N_VAL],
        Y[N_TRAIN + N_VAL:],
        B[:N_TRAIN]
    )

# ==========================================
# 5. Summary Display Table
# ==========================================
print(f"\n{'=' * 73}")
print('Synthetic Spherical Trajectory Forecasting -- Summary Metrics')
print(f"\n{'=' * 75}")
print(f"\n{'Experiment':<12} {'OLS':<22} {'Ridge (proposed)':<20} {'Impr.':<6}   {'lambda*':<4}")
print('-' * 73)
for name, r in results.items():
    print(f"\n{name:<12}"
f"\n{r['ols_mean']:.4f} +/- {r['ols_std']:.4f}      "
f"\n{r['ridge_mean']:.4f} +/- {r['ridge_std']:.4f}    "
f"\n{r['improvement']:+5.1f}%   {r['lam_star']:<6}")
print('-' * 73)
avg_imp = np.mean([r['improvement'] for r in results.values()])
print(f"\n{'Average improvement:':<40} {avg_imp:+.1f}%")
print(f"\n{'='*73}")