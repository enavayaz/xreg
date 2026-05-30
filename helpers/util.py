import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LightSource
matplotlib.use('TkAgg')  # matplotlib.use("Agg")  # NOQA
import jax.numpy as jnp
import numpy as np
from pathlib import Path

eps = 1e-8
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def visSph(points_list, color_list, segment_list=None, size_list=None, intrinsic=True, surf=True):
    """
    Visualize groups of points on the 2D-sphere.

    Parameters
    ----------
    points_list : list of array-like, each (N, 3)
        One entry per group; each entry is an array of 3D unit vectors.
        Shapes (1, N, 3) are squeezed automatically.
    color_list : list of color specs
        One matplotlib color per group (str, hex, RGB tuple, …).
    segment_list : list of bool/int, optional
        One flag per group — if truthy, a polyline is drawn through the
        group's points in the same color as its scatter. Defaults to
        drawing a line only for the first group.
    size_list : list of float, optional
        Marker size (scatter `s`) per group. Defaults to 10 for all groups.
    intrinsic : bool, default True
        Whether to use the intrinsic SLERP interpolation. If False, Euclidean
    surf : bool, default True
        Whether to render the translucent sphere surface.
    """
    _ = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111, projection="3d")
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.computed_zorder = False

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ls = plt.matplotlib.colors.LightSource(azdeg=45, altdeg=45)
    ax.plot_surface(x, y, z, color='#E8F4F8', alpha=0.25,
                    linewidth=0, antialiased=surf, shade=True, lightsource=ls)
    ax.set_axis_off()
    ax.grid(False)

    n = len(points_list)
    if segment_list is None:
        draw_line = [i == 0 for i in range(n)]
    else:
        draw_line = [bool(s) for s in segment_list]

    for i in range(n):
        pts = np.asarray(points_list[i])
        s = size_list[i] if size_list is not None else 10
        if pts.ndim == 3:
            pts = pts.squeeze(0)

        color = color_list[i]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, color=color, marker=".")
        if draw_line[i]:
            if intrinsic:
                for j in range(len(pts) - 1):
                    arc = slerp_arc(pts[j], pts[j + 1])
                    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=color, linewidth=1.0, alpha=0.7)
            else:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.0, alpha=0.7)

    plt.show()


def slerp_arc(p1, p2, n=50):
    """Interpolate a great-circle arc between two unit vectors."""
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    t = np.linspace(0, 1, n)
    # SLERP
    omega = np.arccos(np.clip(p1 @ p2, -1.0, 1.0))
    if omega < 1e-10:
        return np.outer(1 - t, p1) + np.outer(t, p2)  # nearly identical points
    return (np.outer(np.sin((1 - t) * omega), p1) +
            np.outer(np.sin(t * omega), p2)) / np.sin(omega)


# Earth Science
def coord_2D3D(lat, lon, h=0.0):
    """
    this function converts latitude,longitude and height above sea level
    to earthcentered xyx coordinates in wgs84, lat and lon in decimal degrees
    e.g. 52.724156(West and South are negative), heigth in meters
    for algoritm see https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    for values of a and b see https://en.wikipedia.org/wiki/Earth_radius#Radius_of_curvature
    """
    #a = 1  # 6378137.0             #radius a of earth in meters cfr WGS84
    #b = 1  # 6356752.3             #radius b of earth in meters cfr WGS84
    #e2 = 1 - (b ** 2 / a ** 2)
    latr = np.pi*lat/180  # latitude in radians
    lonr = np.pi*lon/180  # longituede in radians
    #Nphi = a / sqrt(1 - e2 * sin(latr) ** 2)
    x = np.cos(latr) * np.cos(lonr)  # (Nphi + h) * cos(latr) * cos(lonr)
    y = np.cos(latr) * np.sin(lonr)  # (Nphi + h) * cos(latr) * sin(lonr)
    z = np.sin(latr)  # (b ** 2 / a ** 2 * Nphi + h) * sin(latr)
    return x, y, z


def coord_3D2D(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    lat = np.sign(z)*180*np.arctan(z/np.sqrt(x**2 + y**2))/np.pi
    lon = 180*np.arctan2(y, x)/np.pi # West is negative
    return lat, lon


def bez_sph(n_points):

    # 1. Define Control Points and Parameters
    P = np.array([
        [-4, 0, -1],
        [2, 1, 1],
        [-2, 0, -1],
        [0, 1, -1]
    ])
    t_values = np.linspace(0, 1, n_points + 1)
    deg = len(P) - 1

    # 2. Normalize Control Points to unit vectors (on the unit sphere)
    # This forces the points P_i to be "spherical."
    P_normalized = P / np.linalg.norm(P, axis=1, keepdims=True)

    # 3. Define the SLERP (Spherical Linear Interpolation) function
    def slerp(v0, v1, t):
        """
        Computes the Spherical Linear Interpolation between two unit vectors.
        """
        # Angle between v0 and v1
        dot_product = np.dot(v0, v1)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        omega = np.arccos(dot_product)

        if omega < 1e-6: # Handle near-collinear vectors
            return (1 - t) * v0 + t * v1

        sin_omega = np.sin(omega)

        # SLERP formula
        slerp_point = (np.sin((1 - t) * omega) / sin_omega) * v0 + \
                      (np.sin(t * omega) / sin_omega) * v1

        # Re-normalize for numerical stability
        return slerp_point / np.linalg.norm(slerp_point)

    # 4. Implement the Spherical De Casteljau (SLERP) Algorithm for the Spherical Bézier Curve
    def sph_bez_curve_pt(t, P):
        """
        Computes a point on the Spherical Bézier Curve of any degree using
        the De Casteljau algorithm with SLERP.
        """
        points = list(P)

        # Iterate for each level of the De Casteljau algorithm
        for j in range(1, deg + 1):
            new_points = []
            for i in range(deg - j + 1):
                # SLERP is used instead of linear interpolation
                b_j_i = slerp(points[i], points[i+1], t)
                new_points.append(b_j_i)
            points = new_points
        return points[0]

    # 5. Compute the Curve Points
    y = np.zeros((len(t_values), 3))
    for i, t in enumerate(t_values):
        y[i] = sph_bez_curve_pt(t, P_normalized)

    #visSph([y, y], ['b', 'r'])
    return y


def generate_on_vec(M, p, u, key):
    """
    Generates a unit-length tangent vector V at point S that is
    orthogonal to vector U under the Riemannian metric of manifold M.

    Parameters:
    -----------
    M : Manifold object (e.g., from geomstats or pymanopt)
    S : array-like
        The point on the manifold where the tangent space resides.
    U : array-like
        The reference tangent vector (e.g., current velocity).

    Returns:
    --------
    V : array-like
        A unit-length tangent vector orthogonal to U.
    """
    z = M.randvec(p, key)
    inner_uz = M.metric.inner(p, u, z)
    inner_uu = M.metric.inner(p, u, u)
    v_raw = z - (inner_uz / (inner_uu + eps)) * u
    # Normalization
    # Scale V_raw so that its Riemannian norm is exactly 1.0
    norm_v = jnp.sqrt(M.metric.inner(p, v_raw, v_raw))
    v = v_raw / (norm_v + 1e-10)
    return v


def save_sph(B, Y, name, target_dir=DEFAULT_DATA_DIR):
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f'{name}.npz'
    np.savez(path, B=B, Y=np.array(Y, dtype=object))


def load_sph(name, target_dir=DEFAULT_DATA_DIR):
    path = target_dir / f'{name}.npz'
    if not path.exists():
        raise FileNotFoundError(f'No file found at {path}')
    data = np.load(path, allow_pickle=True)
    B, Y = data['B'], data['Y'].tolist()
    return B, [np.array(y) for y in Y]