import numpy as np
from numpy import linalg as lg
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import Array
from morphomatics.manifold import Manifold
from typing import Tuple, List
from timeseries.reg import PolyRegression as PolyReg

def cov_mat(log, y, mean_y):
    n = len(y)
    w = jax.vmap(jax.jit(log), (None, 0))(mean_y, y)  # Map data to mean tangent space
    #w = [log(mean_y, y) for k in range(n)] #same
    w_vec = w.reshape(n, -1)
    return 1 / n * w_vec.T @ w_vec

def mahal(p, Q=None, mean=None, cov=None):
    if Q is None:
        u, cov = p - mean, cov
    else:
        u, cov = p - np.mean(Q, axis=0), np.cov(Q)

    # from scipy.spatial import distance
    # distance.mahalanobis(p, mean, lg.inv(np.cov(Q)))  # same
    return np.sqrt(u @ lg.inv(cov) @ u)

def fit_poly_dc(M: Manifold, trjs, deg=3, x=None):
    Coeff, Y = [], []
    for trj in trjs:
        trend = PolyReg(M, jnp.array(trj), jnp.linspace(0., 1., len(trj)), deg).trend
        Coeff += [np.array(trend.control_points)]
        #if x is None:
        x = np.linspace(0.0, 1.0, len(trj))
        #Y += [np.array([trend.eval(t) for t in x])]
        Y += [jax.vmap(trend.eval)(x)]
    return jnp.array(Coeff), Y

def eval_poly_dc(M: Manifold, P: jnp.array, x: jnp.array) -> jnp.array:
    """Evaluates the Bézier spline at time t."""
    ev = jax.jit(lambda t: decasteljau(M, P, t)[0])
    return jax.vmap(ev)(x)

def decasteljau(M: Manifold, P: Array, t: float) -> Tuple[Array, List[Array]]:
    """Generalized de Casteljau algorithm
    :param M: manifold
    :param P: control points of curve beta
    :param t: scalar in [0,1]
    :return  beta(t), (B): result of the de Casteljau algorithm with control points P, (intermediate points Bf in the algorithm)
    """
    # number of control points
    k = len(P)

    # init linearized tree of control points
    B = jnp.concatenate([jnp.asarray(P)[i:] for i in range(k)])
    # for lower-level control points: indices of parent ones w.r.t Bf
    offset = [(2 * k * n - n * n + n) // 2 for n in range(k - 1)]
    idx = np.concatenate([np.arange(k - 1 - i) + o for i, o in enumerate(offset)])
    # compute lower-level points
    f = lambda B, io: (B.at[io[1]].set(M.connec.geopoint(B[io[0]], B[io[0] + 1], t)), None)
    B = lax.scan(f, B, np.c_[idx, k + np.arange(len(idx))])[0]

    return B[-1], [B[o:o + k - i] for i, o in enumerate(offset)]

def diff(M: Manifold, y, ref=None):
    if ref is None:
        ref = np.array([y[k] for k in range(len(y)-1)])
    return np.array([M.metric.log(ref[k], y[k+1]) for k in range(len(y)-1)])