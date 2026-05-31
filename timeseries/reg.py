from functools import partial, cached_property
from timeseries.bezier_polynom import BezierPolynom
from morphomatics.manifold import Manifold, PowerManifold
from morphomatics.opt import RiemannianSteepestDescent
from morphomatics.stats import ExponentialBarycenter
import jax
import jax.numpy as jnp


class PolyRegression(object):
    """
    Higher-order regression for estimation of relationship between
    single explanatory and manifold-valued dependent variable.

    The relationship is modeled via intrinsic Bezier splines (morphomatics.manifold.BezierSpline).

    See:
    Martin Hanik, Hans-Christian Hege, Anja Hennemuth, Christoph von Tycowicz:
    Nonlinear Regression on Manifolds for Shape Analysis using Intrinsic Bézier Splines.
    Proc. Medical Image Computing and Computer Assisted Intervention (MICCAI), 2020.
    """

    def __init__(self, M: Manifold, Y: jnp.array, param: jnp.array, degree: int = 3, P_init=None,
                 maxiter=100, mingradnorm=1e-6):
        """Compute regression with Bézier splines for data in a manifold M.

        :param M: manifold
        :param Y: array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        inY. The integer part determines the segment to which the data point belongs.
        :param P_init: initial guess
        :param maxiter: maximum number of iterations in steepest descent
        :param mingradnorm: stop iteration when the norm of the gradient is lower than mingradnorm

        :return P: array of control points of the optimal Bézier spline
        """

        self._M = M
        self._Y = Y
        self._param = param

        # initial guess
        if P_init is None:
            P_init = self.initControlPoints(M, Y, param, degree)
        # P_init = indep_set(P_init)

        # fit spline to data
        P = PolyRegression.fit(M, Y, param, P_init, degree, maxiter, mingradnorm)
        self._polynom = BezierPolynom(M, P)

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'degree'])
    def fit(M: Manifold, Y: jnp.array, param: jnp.array, P_init: jnp.array, degree: int,
            maxiter=100, mingradnorm=1e-6) -> jnp.array:
        """Fit Bézier spline to data Y,param in a manifold M using gradient descent.

        :param M: manifold
        :param Y: array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        in Y. The integer part determines the segment to which the data point belongs.
        :param P_init: initial guess (independent ctrl. pts. only, see #indep_set)
        :param degree: degree of the polynom
        :param maxiter: maximum number of iterations in steepest descent
        :param mingradnorm: stop iteration when the norm of the gradient is lower than mingradnorm

        :return P: array of independent control points of the optimal Bézier spline.
        """

        # number of independent control points
        k = degree + 1
        # search space: k-fold product of M
        N = PowerManifold(M, k)

        # Cost
        def cost(P):
            # pts = full_set(M, P, degrees)
            # return sumOfSquared(BezierPolynom(M, pts), Y, param) / len(Y)
            return sumOfSquared(BezierPolynom(M, P), Y, param) / len(Y)

        args = {'stepsize': 1., 'maxiter': maxiter, 'mingradnorm': mingradnorm}
        # return RiemSteepestDescent.fixedpoint(N, cost, P_init, **args)
        return RiemannianSteepestDescent.fixedpoint(N, cost, P_init, **args)

    @property
    def trend(self) -> BezierPolynom:
        """
        :return: Estimated trajectory encoding relationship between
            explanatory and manifold-valued dependent variable.
        """
        return self._polynom

    @cached_property
    def unexplained_variance(self) -> float:
        """Variance in the data set that is not explained by the regressed Bézier spline.
        """
        cost = sumOfSquared(self.trend, self._Y, self._param)
        return cost / len(self._Y)

    @property
    def R2statistic(self) -> float:
        """ Computes Fletcher's generalized R2 statistic for Bézier spline regression. For the definition see
                        Fletcher, Geodesic Regression on Riemannian Manifolds (2011), Eq. 7.

        :return: generalized R^2 statistic (in [0, 1])
        """

        # total variance
        total_var = ExponentialBarycenter.total_variance(self._M, self._Y)

        return 1 - self.unexplained_variance / total_var

    @staticmethod
    def initControlPoints(M: Manifold, Y: jnp.array, param: jnp.array,
                          degree: int) -> jnp.array:
        """Computes an initial choice of control points for the gradient descent steps in polynomial
        regression.
        The control points are initialized "along geodesics" near the data
        points such that the differentiability conditions hold.

        :param M:  manifold
        :param Y:  array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        in Y. The integer part determines the segment to which the data point belongs.
        :param degrees: vector of length L; the l-th entry is the degrees of the l-th segment of the spline. All entries
        must be positive. For a closed spline, L > 1, degrees[0] > 2 and degrees[-1] > 2 must hold.

        :return P: list of length L containing arrays of control points. The l-th entry is an
               array with degrees(l)+1 elements of M, that are ordered along the first dimension.
        """

        P, d = [], degree
        for i in range(0, d + 1):
            P.append(M.connec.geopoint(Y[0], Y[-1], i / d))
        return jnp.array(P)
        # return np.polyfit(param, Y, d)


def sumOfSquared(B: BezierPolynom, Y: jnp.array, param: jnp.array) -> float:
    """Computes sum of squared distances between the spline
    defined by P and data Y.
    :param B: Bézier spline
    :param Y: array with data points along first axis
    :param param: vector with corresponding parameter values
    :return: non-negative scalar
    """

    return jnp.sum(jax.vmap(lambda y, t: B._M.metric.squared_dist(B.eval(t), y))(Y, param))


def gradSumOfSquared(B: BezierPolynom, Y: jnp.array, param: jnp.array) -> jnp.array:
    """Compute the gradient of the sum of squared distances from a manifold-valued Bézier spline to time labeled data
    points.
    :param B: Bézier spline with K segments
    :param Y: array that contains data in the manifold where B is defined (along first axis).
    :param param: vector with the sorted parameter values that correspond to the data in Y. All values must be
    in [0, B.nsegments].
    :return: gradients at the control points of B
    """

    M = B._M

    grad_i = lambda y, t: -2 * B.adjDpB(t, M.connec.log(B.eval(t), y))
    grad_E = jnp.sum(jax.vmap(grad_i)(Y, param), axis=0)

    # Taking care of C1/cycle conditions
    # return RiemannianRegression.grad_constraints(B, grad_E)
    return grad_constraints(B, grad_E)


def grad_constraints(B: BezierPolynom, grad_E: jnp.array) -> jnp.array:
    """Compute the gradient of the sum of squared distances from a manifold-valued Bézier spline to time labeled data
    points.
    :param B: Bézier spline with K segments
    :param grad_E: gradients at the control points for each segment
    :return: corrected gradients s.t. C1/cycle conditions are accounted for
    """

    M = B._M

    P = B.control_points

    L = 1

    # Taking care of C1 conditions
    for l in range(1, L):
        k = B.degree

        X_plus = grad_E[l][1]  # gradient w.r.t. p_l^+
        X_l = M.connec.adjDygeo(P[l - 1, -2], P[l, 0], k, X_plus)
        X_minus = M.connec.adjDxgeo(P[l - 1, -2], P[l, 0], k, X_plus)

        # Final gradients at p_l and p_l^-
        grad_E = grad_E.at[l - 1, -1].set(grad_E[l - 1, -1] + grad_E[l, 0] + X_l)
        grad_E = grad_E.at[l - 1, -2].set(grad_E[l - 1, -2] + X_minus)

    return jax.lax.cond(False, lambda g: g, lambda g: g, grad_E)


class RidgeRegression(object):
    """
    Higher-order regression for estimation of relationship between
    single explanatory and manifold-valued dependent variable.

    The relationship is modeled via intrinsic Bezier splines (morphomatics.manifold.BezierSpline).

    See:
    Martin Hanik, Hans-Christian Hege, Anja Hennemuth, Christoph von Tycowicz:
    Nonlinear Regression on Manifolds for Shape Analysis using Intrinsic Bézier Splines.
    Proc. Medical Image Computing and Computer Assisted Intervention (MICCAI), 2020.
    """

    def __init__(self, M: Manifold, Y: jnp.array, param: jnp.array, mean: jnp.array, cov: jnp.array, ridge_const: jnp.array, degree: int = 3,
                 P_init=None, maxiter=100, mingradnorm=1e-6):
        """Compute regression with Bézier splines for data in a manifold M.

        :param M: manifold
        :param Y: array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        inY. The integer part determines the segment to which the data point belongs.
        :param degree: degree of each segment of the polynom
        :param P_init: initial guess
        :param maxiter: maximum number of iterations in steepest descent
        :param mingradnorm: stop iteration when the norm of the gradient is lower than mingradnorm

        :return P: array of control points of the optimal Bézier spline
        """

        self._M = M
        self._Y = Y
        self._param = param

        # initial guess
        if P_init is None:
            P_init = self.initControlPoints(M, Y, param, degree)
        #P_init = indep_set(P_init)

        # Precompute Lambda_inv and V_full from covariance matrix (only once!)
        dim_eff = (degree + 1) * M.dim
        Lambda_inv, V_full = cov_intrinsic(cov, dim_eff)

        # fit spline to data
        P = RidgeRegression.fit(M, Y, param, P_init, mean, cov, ridge_const, degree, maxiter, mingradnorm, Lambda_inv, V_full)
        self._Polynom = BezierPolynom(M, P)

    @staticmethod
    @partial(jax.jit, static_argnames=['M', 'degree'])
    def fit(M: Manifold, Y: jnp.array, param: jnp.array, P_init: jnp.array, mean: jnp.array, cov: jnp.array, ridge_const: jnp.array, degree: int,
            maxiter=100, mingradnorm=1e-6, Lambda_inv=None, V_full=None) -> jnp.array:
        """Fit Bézier spline to data Y,param in a manifold M using gradient descent.

        :param M: manifold
        :param Y: array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        in Y. The integer part determines the segment to which the data point belongs.
        :param P_init: initial guess (independent ctrl. pts. only, see #indep_set)
        :param degree: degree of the polynom
        :param maxiter: maximum number of iterations in steepest descent
        :param mingradnorm: stop iteration when the norm of the gradient is lower than mingradnorm

        :return P: array of independent control points of the optimal Bézier spline.
        """

        # number of independent control points
        k = degree + 1
        # search space: k-fold product of M
        N = PowerManifold(M, k)

        # Compute Lambda_inv and V_full if not provided (for backward compatibility)
        if Lambda_inv is None or V_full is None:
            dim_eff = len(P_init) * M.dim
            Lambda_inv, V_full = cov_intrinsic(cov, dim_eff)
        scale_factor = jnp.max(Lambda_inv)  # jnp.sum(Lambda_inv)
        # Cost
        def cost(P):
            reg = sumOfSquared(BezierPolynom(M, P), Y, param) #/ len(Y)
            v = N.metric.log(mean, P)

            # Flatten v to match covariance matrix dimensions
            v_flat = v.reshape(-1)
            #scale_factor = jnp.max(Lambda_inv)  #jnp.sum(Lambda_inv)
            mahal = mahal_squared_eff(v_flat, Lambda_inv, V_full) / scale_factor
            return (reg + ridge_const * mahal) / len(v_flat) #/ (len(Y) + len(cov_inv))

        args = {'stepsize': 1., 'maxiter': maxiter, 'mingradnorm': mingradnorm}
        return RiemannianSteepestDescent.fixedpoint(N, cost, P_init, **args)

    @property
    def trend(self) -> BezierPolynom:
        """
        :return: Estimated trajectory encoding relationship between
            explanatory and manifold-valued dependent variable.
        """
        return self._Polynom

    @cached_property
    def unexplained_variance(self) -> float:
        """Variance in the data set that is not explained by the regressed Bézier spline.
        """
        cost = sumOfSquared(self.trend, self._Y, self._param)
        return cost / len(self._Y)

    @property
    def R2statistic(self) -> float:
        """ Computes Fletcher's generalized R2 statistic for Bézier spline regression. For the definition see
                        Fletcher, Geodesic Regression on Riemannian Manifolds (2011), Eq. 7.

        :return: generalized R^2 statistic (in [0, 1])
        """

        # total variance
        total_var = ExponentialBarycenter.total_variance(self._M, self._Y)

        return 1 - self.unexplained_variance / total_var

    @staticmethod
    def initControlPoints(M: Manifold, Y: jnp.array, param: jnp.array,
                          degree: jnp.array) -> jnp.array:
        """Computes an initial choice of control points for the gradient descent steps in polynomial
        regression.
        The control points are initialized "along geodesics" near the data
        points such that the differentiability conditions hold.

        :param M:  manifold
        :param Y:  array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        in Y. The integer part determines the segment to which the data point belongs.
        :param degrees: vector of length L; the l-th entry is the degrees of the l-th segment of the spline. All entries
        must be positive. For a closed spline, L > 1, degrees[0] > 2 and degrees[-1] > 2 must hold.

        :return P: list of length L containing arrays of control points. The l-th entry is an
               array with degrees(l)+1 elements of M, that are ordered along the first dimension.
        """

        P, d = [], degree
        for i in range(0, d + 1):
            P.append(M.connec.geopoint(Y[0], Y[-1], i / d))
        return jnp.array(P)
        # return np.polyfit(param, Y, d)


def sumOfSquared(B: BezierPolynom, Y: jnp.array, param: jnp.array) -> float:
    """Computes sum of squared distances between the spline
    defined by P and data Y.
    :param B: Bézier spline
    :param Y: array with data points along first axis
    :param param: vector with corresponding parameter values
    :return: non-negative scalar
    """

    return jnp.sum(jax.vmap(lambda y, t: B._M.metric.squared_dist(B.eval(t), y))(Y, param))


def gradSumOfSquared(B: BezierPolynom, Y: jnp.array, param: jnp.array) -> jnp.array:
    """Compute the gradient of the sum of squared distances from a manifold-valued Bézier spline to time labeled data
    points.
    :param B: Bézier spline with K segments
    :param Y: array that contains data in the manifold where B is defined (along first axis).
    :param param: vector with the sorted parameter values that correspond to the data in Y. All values must be
    in [0, B.nsegments].
    :return: gradients at the control points of B
    """

    M = B._M

    grad_i = lambda y, t: -2 * B.adjDpB(t, M.connec.log(B.eval(t), y))
    grad_E = jnp.sum(jax.vmap(grad_i)(Y, param), axis=0)

    # Taking care of C1/cycle conditions
    # return RiemannianRegression.grad_constraints(B, grad_E)
    return grad_constraints(B, grad_E)


def grad_constraints(B: BezierPolynom, grad_E: jnp.array) -> jnp.array:
    """Compute the gradient of the sum of squared distances from a manifold-valued Bézier spline to time labeled data
    points.
    :param B: Bézier spline with K segments
    :param grad_E: gradients at the control points for each segment
    :return: corrected gradients s.t. C1/cycle conditions are accounted for
    """

    M = B._M

    P = B.control_points

    L = 1

    # Taking care of C1 conditions
    for l in range(1, L):
        k = B.degree

        X_plus = grad_E[l][1]  # gradient w.r.t. p_l^+
        X_l = M.connec.adjDygeo(P[l - 1, -2], P[l, 0], k, X_plus)
        X_minus = M.connec.adjDxgeo(P[l - 1, -2], P[l, 0], k, X_plus)

        # Final gradients at p_l and p_l^-
        grad_E = grad_E.at[l - 1, -1].set(grad_E[l - 1, -1] + grad_E[l, 0] + X_l)
        grad_E = grad_E.at[l - 1, -2].set(grad_E[l - 1, -2] + X_minus)

    return jax.lax.cond(False, lambda g: g, lambda g: g, grad_E)

def cov_intrinsic(C, dim_eff, eigenvalue_threshold=1e-8):
    # Eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(C)
    # Sort eigenvalues in descending order
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Create mask for first m components
    n_features = eigenvectors.shape[1]
    mask = (jnp.arange(n_features) < dim_eff) & (eigenvalues > eigenvalue_threshold)

    # Apply mask to actually truncate (not zero out)
    V_full = eigenvectors[:, mask]
    Lambda_full = eigenvalues[mask]

    # Compute inverse eigenvalues (epsilon just for numerical safety)
    Lambda_inv = 1.0 / (Lambda_full + 1e-12)

    return Lambda_inv, V_full

def mahal_squared_eff(b, Lambda_inv, V_full):
    """
    Compute Mahalanobis distance between point b and samples in effective subspace E of dimension dim_eff.

    Parameters:
    -----------
    b : array-like, query point in ambient space, centered at mean (b = log(mean, b))
    Lambda_inv : inv of covariance matrix on E
    V_full : projection to effective subspace E, where data live
    eigenvalue_threshold : float, relative threshold for considering eigenvalues as non-zero

    Returns:
    --------
    distance : float, Mahalanobis squared distance from b to samples
    """
    # Project to effective subspace E
    b_proj = V_full.T @ b

    # Compute Mahalanobis distance in subspace
    # Only the first m components will contribute (others are multiplied by 0)
    mahal_squared = jnp.sum(b_proj * Lambda_inv * b_proj)

    return mahal_squared