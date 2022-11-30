# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from pyDOE import lhs
from matplotlib import cm
from scipy.stats import gamma
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler


# Sampling strategies
# -----------------------------------------------------------------------------
def stratified_sampling(
    x,
    strata
):
    x_split = np.array_split(x, strata)
    x_1 = np.array([np.random.choice(i) for i in x_split])
    x_2 = np.setdiff1d(x, x_1)
    assert np.intersect1d(x_1, x_2).size == 0
    return x_2, x_1

def sample_lhs(
    bounds,
    nb_samples
):
    dim = bounds.shape[1]
    # Contructing design matrix
    x_norm = lhs(dim, nb_samples)
    x = x_norm * (bounds[1] - bounds[0]) + bounds[0]
    # Sorting w.r.t. the projection norm
    proj = np.dot(x_norm, np.ones((dim,1))).squeeze()
    samples = x[proj.argsort()]
    # Returning initial conditions
    return samples

def sample_conditional_lhs(
    bounds,
    nb_samples,
    indices=None
):
    dim = bounds.shape[1]
    indices = indices + [dim-1]
    # Contructing design matrix
    X = lhs(dim+1, nb_samples)
    Y = qdirichlet(np.take(X, indices, 1), np.repeat(2,len(indices)))
    for i, idx in enumerate(indices[:-1]):
        X[:,idx] = Y[:,i]
    x_norm = X[:,:-1]
    x = x_norm * (bounds[1] - bounds[0]) + bounds[0]
    # Sorting w.r.t. the projection norm
    proj = np.dot(x_norm, np.ones((dim,1))).squeeze()
    samples = x[proj.argsort()]
    # Returning initial conditions
    return samples

def qdirichlet(X, alpha):
    # dirichlet_ppf is not an exact quantile function since the quantile of a
    #  multivariate distribtion is not unique
    # dirichlet_ppf is also not the quantiles of the marginal distributions since
    #  those quantiles do not sum to one
    # dirichlet_ppf is the quantile of the underlying gamma functions, normalized
    # This has been tested to show that dirichlet_ppf approximates the dirichlet
    #  distribution well and creates the correct marginal means and variances
    #  when using a latin hypercube sample
    #
    # Python translation of qdirichlet function by  R. Carnell
    # original: https://stats.stackexchange.com/a/476433/244679
    assert alpha.ndim == 1, "parameter alpha must be a vector"
    assert X.ndim == 2, "parameter X must be an array with samples as rows and variables as columns"
    assert X.shape[1] == alpha.shape[0], "number of variables in each row of X and length of alpha must be equal"
    assert not (np.any(np.isnan(X)) or np.any(np.isnan(alpha))), "NAN values are not allowed in dirichlet_ppf"
    
    Y = np.zeros(shape=X.shape)
    for idx, a in enumerate(alpha):
        if a != 0. :
            Y[:, idx] = gamma.ppf(X[:, idx], a)
    
    return Y / Y.sum(axis=1)[:, np.newaxis]

def check_samples(samples, bounds):
    n = len(samples)
    dT = -np.diff(np.take(samples, [-2,-1], axis=1))
    samples = np.hstack([samples, dT])
    cond = (samples >= bounds[0]) * (samples <= bounds[1])
    cond = np.product(cond, axis=1)
    idx  = np.where(cond)[0]
    samples = samples[idx]
    return samples, n - len(idx)

def lhs_kde_combined_sampling(
    nb,
    bounds,
    bounds_raw,
    bounds_prim,
    kde,
    scaler,
    kde_samples_perc
):
    # Define number of samples
    nb_samples_lhs = int(nb*(1-kde_samples_perc))
    nb_samples_kde = nb - nb_samples_lhs
    # LHS
    samples = []
    n = nb_samples_lhs
    while n > 0:
        samples_n = sample_lhs(
            bounds=bounds_prim,
            nb_samples=n
        )
        samples_n, n = check_samples(samples_n, bounds)
        samples.append(samples_n)
    samples_lhs = np.vstack(samples)
    # KDE
    samples = []
    n = nb_samples_kde
    while n > 0:
        samples_n = sample_kde(
            kde,
            nb_samples=n,
            scaler=scaler,
            bounds=bounds_prim
        )
        samples_n, n = check_samples(samples_n, bounds)
        samples.append(samples_n)
    samples_kde = np.vstack(samples)
    # Concatenate
    samples = np.vstack([samples_lhs, samples_kde])
    samples = np.take(samples, np.arange(len(bounds_raw)), 1)
    return samples

# Bounds and scaling
# -----------------------------------------------------------------------------
def construct_scaler(
    X
):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler

def compute_bounds(
    X,
    phys_bounds=None,
    extra_perc=0.1
):
    scaler = construct_scaler(X)
    # Set minimum values
    min_values = []
    for v in scaler.data_min_:
        if v >= 0.:
            min_values.append(v*(1.-extra_perc))
        else:
            min_values.append(v*(1.+extra_perc))
    # Set maximum values
    max_values = []
    for v in scaler.data_max_:
        if v >= 0.:
            max_values.append(v*(1.+extra_perc))
        else:
            max_values.append(v*(1.-extra_perc))
    bounds = np.stack([np.array(min_values), np.array(max_values)], 0)
    # Check physical bounds
    if phys_bounds is not None:
        d = phys_bounds.shape[1]
        bounds[0,:d] = np.maximum(bounds[0,:d], phys_bounds[0])
        bounds[1,:d] = np.minimum(bounds[1,:d], phys_bounds[1])
    # Round bounds
    order = np.floor(np.log10(np.absolute(bounds)))
    magn = np.power(10, order)
    bounds[0] = np.floor(bounds[0]/magn[0]) * magn[0]
    bounds[1] = np.ceil(bounds[1]/magn[1]) * magn[1]
    return bounds

# Kernel-density estimate
# -----------------------------------------------------------------------------
def construct_kde(
    X,
    bandwidth=0.02,
    kernel='gaussian',
    scaler=None
):
    if scaler is not None:
        X = scaler.transform(X)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
    return kde

def sample_kde_once(
    kde,
    nb_samples=100,
    scaler=None
):
    samples = kde.sample(nb_samples)
    if scaler is not None:
        samples = scaler.inverse_transform(samples)
    return samples

def sample_kde(
    kde,
    nb_samples=100,
    scaler=None,
    bounds=None,
    resampling=False
):
    if bounds is not None:
        if resampling:
            samples = []
            n = nb_samples
            while n > 0:
                samples_n = sample_kde_once(kde, n, scaler)
                cond = (samples_n >= bounds[0]) * (samples_n <= bounds[1])
                cond = np.product(cond, axis=1)
                idx  = np.where(cond)[0]
                samples.append(samples_n[idx])
                n = n - len(idx)
            samples = np.vstack(samples)
        else:
            samples = sample_kde_once(kde, nb_samples, scaler)
            samples = np.absolute(samples - bounds[0]) + bounds[0]
            samples = -np.absolute(-samples + bounds[1]) + bounds[1]
    else:
        samples = sample_kde_once(kde, nb_samples, scaler)
    return samples

# Plotting
# -----------------------------------------------------------------------------
def plot_hist_1d(x, xlabel, figname='hist_1d.pdf', nb_bins=40, show=False):
    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(x, bins=nb_bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Num. samples')
    if show:
        plt.show()
    plt.savefig(figname)
    plt.close()
    
def plot_hist_2d(x, y, labels, figname='hist_2d.pdf', nb_bins=40, show=False, cmap='jet'):
    fig, ax = plt.subplots(tight_layout=True)
    ax.hist2d(x, y, bins=nb_bins, cmap=cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(figname)
    if show:
        plt.show()
    plt.close()
    
def plot_hist_3d(x, y, labels, figname='hist_3d.pdf', nb_bins=40, show=False, cmap='jet'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=(nb_bins,nb_bins))
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    cmap = cm.get_cmap(cmap)
    max_height = np.max(dz)
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz]
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(figname)
    if show:
        plt.show()
    plt.close()
