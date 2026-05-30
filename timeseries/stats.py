import numpy as np
import numpy.linalg as lg
import jax
import jax.numpy as jnp
from jax import random as rnd
from morphomatics.stats import ExponentialBarycenter as Mean
from helpers.util import bez_sph
from helpers.util_pred import eval_poly_dc

eps = 1e-8

def map2D3D(x, y, uniform=True):
    #Z = np.zeros((n_points, 3))
    if uniform:
        Z = np.sqrt(1-y**2)*np.cos(x), np.sqrt(1-y**2)*np.sin(x), y
    else:
        Z = np.cos(y)*np.cos(x), np.cos(y)*np.sin(x), np.sin(y)  # central distribution
    return Z


def generate_polynomial_series(M,
                               n_points: int = 30,
                               deg: int = 3,
                               noise_level: float = 0.1,
                               key=None,
                               ) -> np.ndarray:
    """
    Generate manifold time series with polynomial (Bézier) trend

    Creates smooth, predictable evolution that polynomial regression
    should capture well.

    Parameters
    ----------
    M : Manifold
        The manifold (e.g., SPD, Sphere)
    n_points : int
        Number of time points
    deg : int
        Polynomial degree (1=linear, 2=quadratic, 3=cubic)
    noise_level : float
        Noise level (0=deterministic, 0.5=very noisy)
        key : jax.random.PRNGKey, optional
        Random key for reproducibility. If None, generates a random one.

    Returns
    -------
    np.ndarray, shape (n_points, *M.point_shape)
        Manifold time series
    """
    # Initialize random keys
    if key is None:  # ADD THIS CHECK
        key = rnd.PRNGKey(np.random.randint(0, 2 ** 32))

    master_key = key  # CHANGE THIS LINE (was: random.PRNGKey(0))
    init_key, noise_key = rnd.split(key)

    # Generate time parameter
    t = np.linspace(0, 1, n_points)

    # Generate deg+1 random control points on the manifold
    ctl_pts = np.empty((deg + 1,) + M.point_shape)
    init_keys_array = rnd.split(init_key, deg + 1)

    for i in range(deg + 1):
        ctl_pts[i] = M.rand(init_keys_array[i])

    # Evaluate Bézier polynomial curve
    Y = eval_poly_dc(M, ctl_pts, t)

    # Add noise if requested
    if noise_level > 0:
        Y_noisy = add_correlated_noise_TS(M, Y, noise_key, noise_level, correlation=.8)
        #Y_noisy = add_gauss_noise(M, Y, noise_key, noise_level)
    else:
        Y_noisy = Y

    # Convert JAX array to NumPy array
    return np.asarray(Y_noisy)

# ============================================================================
# Add Noise to Trajectories
# ============================================================================

#@partial(jax.jit, static_argnums=(3,))
def add_correlated_noise_TS(
    M,
    Y: np.ndarray,
    key,
    noise_level: float,
    correlation: float = 0.8
) -> np.ndarray:
    """
    Add temporally correlated noise to time series on manifold.

    Creates realistic observation noise with temporal correlation,
    mimicking real-world measurement processes where consecutive
    observations have correlated errors.

    If Y is not a time series, just use add_gauss_noise

    Parameters
    ----------
    M : Manifold
        The manifold
    Y : np.ndarray
        Clean trajectory
    key : JAX random key
        Random seed
    noise_level : float
        Base noise magnitude
    correlation : float in [0, 1]
        Temporal correlation strength
        - 0 = independent noise (like add_noise_TS)
        - 1 = fully correlated (random walk noise)
        - 0.8 = realistic (recommended)

    Returns
    -------
    Y_noisy : np.ndarray
        Trajectory with correlated noise
    """
    n = len(Y)
    Y_noisy = np.empty_like(Y)
    Y_noisy[0] = Y[0]

    # Initialize noise direction
    key, subkey = rnd.split(key)
    noise_direction = M.randvec(Y[0], subkey)
    noise_direction = noise_direction / (np.linalg.norm(noise_direction) + eps)

    for i in range(1, n):
        key, subkey = rnd.split(key)

        # Update noise direction with correlation
        new_random = M.randvec(Y[i], subkey)
        new_random = new_random / (np.linalg.norm(new_random) + eps)

        # Blend: correlation * old + (1-correlation) * new
        noise_direction = correlation * M.metric.transp(Y[i-1], Y[i], noise_direction) + (1 - correlation) * new_random
        noise_direction = noise_direction / (np.linalg.norm(noise_direction) + eps)

        # Apply correlated noise
        noise = noise_level * noise_direction
        Y_noisy[i] = M.metric.exp(Y[i], noise)

    return Y_noisy


#@partial(jax.jit, static_argnums=(3,))
def add_gauss_noise(
        M,
        Y: np.ndarray,
        key,
        noise_level: float
) -> np.ndarray:
    """
    Add independent Riemannian Gaussian noise to each sample in Y on manifold.

    Creates i.i.d. observation noise where each error is independent
    of the previous one. This corresponds to the case where correlation = 0.

    Parameters
    ----------
    M : Manifold
        The manifold
    Y : np.ndarray
        Clean trajectory
    key : JAX random key
        Random seed
    noise_level : float
        Standard deviation of the noise magnitude

    Returns
    -------
    Y_noisy : np.ndarray
        Trajectory with independent Gaussian noise
    """
    n = len(Y)
    Y_noisy = np.empty_like(Y)

    for i in range(n):
        key, subkey = rnd.split(key)

        # Sample a random tangent vector (innovation)
        # Note: M.randvec usually samples from a standard normal in the tangent space
        noise_direction = M.randvec(Y[i], subkey)

        # Normalize to ensure isotropic direction if desired,
        # or leave as is for true Gaussian scaling.
        # Here we follow the convention of fixed-magnitude directional noise:
        noise_direction = noise_direction / (np.linalg.norm(noise_direction) + eps)

        # Apply independent noise
        noise = noise_level * noise_direction
        Y_noisy[i] = M.metric.exp(Y[i], noise)

    return Y_noisy

# ============================================================================
# Generate CORRELATED trajectories
# ============================================================================

def sph_correlated_trjs(lon_max, lat_max, n_trj=30, n_points=40, noise_std=0.03,
                        mean_curve='Sin', correlation=0.95):
    """
    Generate highly correlated trajectories by perturbing a shared template
    with temporally correlated (AR(1)) Gaussian noise mapped to the sphere
    via the exponential map.

    The shared template induces between-trajectory correlation. Under the
    tangent-space approximation, this correlation is:

        rho ~ var_template / (var_template + var_noise)

    where var_noise = noise_std^2 / (1 - correlation^2) is the stationary
    AR(1) variance.  For noise_std=0.05, correlation=0.95, and typical
    template variance of order 0.1-0.3 on the unit sphere, this gives rho
    approximately in the range 0.80-0.95.

    Lower noise_std or higher correlation => higher rho => ridge helps more.
    Try: noise_std = 0.02 (very high correlation, rho > 0.90)
         noise_std = 0.05 (moderate-high correlation, rho ~ 0.80)
         noise_std = 0.10 (lower correlation - like hurricanes)

    Parameters
    ----------
    lon_max : float
        Maximum longitude range for template generation.
    lat_max : float
        Maximum latitude range for template generation.
    n_trj : int
        Number of trajectories to generate.
    n_points : int
        Number of time points per trajectory.
    noise_std : float
        Individual within-trajectory noise standard deviation.
        Controls both noise magnitude and, via var_noise, the
        between-trajectory correlation rho.
    mean_curve : str
        Template shape: 'Geo' (geodesic), 'Sin' (sinusoidal), or
        anything else for a random Bezier curve.
    correlation : float
        Within-trajectory AR(1) temporal correlation coefficient.
        Higher values produce smoother individual tracks.
    """
    from morphomatics.manifold import Sphere
    M = Sphere()
    # Create template trajectory
    if mean_curve == 'Geo':
        start_point, end_point = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        template = np.array([M.metric.geopoint(start_point, end_point, t) for t in np.linspace(0, 1, n_points)])
    elif mean_curve == 'Sin':  # perturbated sine curve
        n_template = n_points
        x_template = np.linspace(0, lon_max, n_template)
        y_max = np.sin(lat_max)
        y_template = 0.5 * y_max * np.sin(np.pi * x_template / lon_max)
        z_template = map2D3D(x_template, y_template, uniform=True)
        template = np.array([z_template[0], z_template[1], z_template[2]]).T
    else:
        template = bez_sph(n_points)
    # Generate correlated trajectories
    Y = []
    for i in range(n_trj):
        template_i, n_points_i = template.copy(), n_points
        # Add small correlated noise
        noisy_trj = np.zeros((n_points_i, 3))
        prev_noise_u, prev_noise_v = 0, 0

        for j in range(n_points_i):
            p = template_i[j]
            u = np.array([-p[1], p[0], 0])
            u = u / lg.norm(u)
            v = np.cross(p, u)

            # Spatially correlated noise
            noise_u = np.random.normal(0, noise_std)
            noise_v = np.random.normal(0, noise_std)

            if j > 0:
                # AR(1) temporal smoothing with coefficient `correlation` (default 0.95).
                # Stationary variance = noise_std^2 / (1 - correlation^2).
                # Combined with the shared template this yields between-trajectory
                # correlation rho ~ var_template / (var_template + noise_std^2 / (1 - correlation^2))
                noise_u = correlation * prev_noise_u + np.sqrt(1 - correlation ** 2) * noise_u
                noise_v = correlation * prev_noise_v + np.sqrt(1 - correlation ** 2) * noise_v

            prev_noise_u, prev_noise_v = noise_u, noise_v
            noisy_trj[j] = M.metric.exp(p, noise_u * u + noise_v * v)

        Y.append(noisy_trj)

    return Y, template

# Generate list of random trajectories
def sph_rand_trjs(lon_max, lat_max, n_trj=30, n_points=30, uniform=True):
    Y = []
    for i in range(n_trj):
        n_points = np.random.randint(30, 70)
        # Generate x coordinates: random but sorted (increasing)
        x = np.sort(np.random.uniform(0, lon_max, n_points))
        #x =np.linspace(0,lon_max,n_points)

        # Generate y coordinates: random
        if uniform:
            y_max = np.sin(lat_max)
            y = np.random.uniform(-y_max, y_max, n_points)
        else:
            y = np.random.uniform(-lat_max, lat_max, n_points)
        y = np.sort(y)
        # map to sphere
        z = map2D3D(x, y, uniform=uniform)
        Y.append(np.array([z[0], z[1], z[2]]).T)
    return Y

#==========================================
# PGA
#==========================================

class PrincipalGeodesicAnalysis(object):
    """
    Principal Geodesic Analysis (PGA) as introduced by
    Fletcher et al. (2003): Statistics of manifold via principal geodesic analysis on Lie groups.
    """

    def __init__(self, mfd, data, mu=None):
        """
        Setup PGA.

        :arg mfd: underlying data space (Assumes that mfd#inner(...) supports list of vectors)
        :arg data: list of data points
        :arg mu: intrinsic mean of data
        """
        assert mfd.connec and mfd.metric
        self.mfd = mfd
        N = len(data)

        # assure mean
        if mu is None:
            mu = Mean.compute(mfd, data)
        self._mean = mu

        ################################
        # inexact PGA, aka tangent PCA
        ################################

        # map data to tangent space at mean
        v = jax.vmap(jax.jit(mfd.connec.log), (None, 0))(mu, data)

        # setup covariance operator
        v_vec = v.reshape(N, -1)
        C = 1/N * v_vec.T @ v_vec

        variances, modes, coeffs = self.compute_cov(C, v_vec)
        self.cov = C

        self._variances = variances
        self._modes = modes
        self._coeffs = coeffs

    def compute_cov(self, C, v):
        d = self.mfd.dim
        # decompose
        vals, vecs = jnp.linalg.eigh(C)

        # set variance and modes
        n = jnp.sum(vals > 1e-6)
        e = d - n - 1 if n<d else -d-1
        variances = vals[:e:-1]
        modes = vecs[:, :e:-1].T.reshape((n,) + self.mfd.point_shape)

        coeffs = v @ vecs[:,:e:-1]

        return variances, modes, coeffs