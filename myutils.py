from collections.abc import Iterable
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def iterable(obj):
    return isinstance(obj, Iterable)

def isclose(a, b, tol=(1e-4, 1e-13)):
    a = np.reshape(a, -1)
    b = np.reshape(b, -1)
    assert (a.dtype == np.float32 or a.dtype == np.float64) and (b.dtype == np.float32 or b.dtype == np.float64), "This function handles numpy float32 and float64 dtypes only"
    if a.dtype == np.float32 or b.dtype == np.float32:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        tol = tol[0] if iterable(tol) else tol
    else:
        tol = tol[1] if iterable(tol) else tol
    if a.size > 1:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        namb = np.linalg.norm(a - b)
    else:
        na = np.abs(a)
        nb = np.abs(b)
        namb = np.abs(a - b)
    if (na == 0 and nb == 0) or namb == 0:
        return True, 0
    return tol > namb * 2 / (na + nb), namb * 2 / (na + nb)

def get_one_hot(indices, depth):
    one_hot = np.eye(depth)[np.array(indices).reshape(-1)]
    return one_hot.reshape(list(indices.shape)+[depth])

def maxabs(a):
	return np.max(np.abs(a))

def minabs(a):
	return np.min(np.abs(a))

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)