'''
Adaptation of the pTFCE algorithm to MEG data.

Probabilistic TFCE (pTFCE) was originally published in:

    Spisák T, Spisák Z, Zunhammer M, Bingel U, Smith S, Nichols T, & Kincses T
    (2019). Probabilistic TFCE: A generalized combination of cluster size and
    voxel intensity to increase statistical power. NeuroImage, 185, 12–26.
    https://doi.org/10.1016/j.neuroimage.2018.09.078

The original implementations (in R and MATLAB) are here:

- https://github.com/spisakt/pTFCE
- https://github.com/spisakt/pTFCE_spm

As a result of the original authors' interest in MRI analysis, much of the
implementation relies on Gaussian Random Field Theory (GRFT) which, while
appropriate for MRI (where voxel measurements are independent and image values
are routinely z-scored) are not suitable for M/EEG-based distributed source
estimates (where the number of source vertices is much greater than the data
rank, and where activation values are usually restricted to be non-negative).

This code adapts pTFCE to the M/EEG source imaging case, by empirically
generating the necessary null distributions of suprathreshold source
activations and cluster sizes, instead of deriving them from GRFT.

author: Daniel McCloy <dan@mccloy.info>
license: BSD 3-clause
'''

# from functools import partial
from time import perf_counter
from contextlib import contextmanager
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.stats import norm, gaussian_kde
import mne

# prevent NaN propogation / careless array munging
import warnings
warnings.filterwarnings('error', 'invalid value encountered in')
warnings.filterwarnings('error', 'divide by zero encountered in')
warnings.filterwarnings('error', 'Creating an ndarray from ragged nested')


@contextmanager
def timer(description: str) -> None:
    """Simple context manager for timing code execution piecemeal."""
    if description:
        print(description)
    start = perf_counter()
    yield
    elapsed_time = perf_counter() - start
    space = ' ' if description else ''
    print(f'elapsed time{space}{description}: {elapsed_time:.4f} sec.')


def calc_thresholds(data, n_thresh=100):
    """Compute pTFCE thresholds (equidistant in log-space)."""
    min_logp_thresh = 0.
    max_logp_thresh = -1 * norm.logsf(data.max())
    logp_thresholds = np.linspace(min_logp_thresh, max_logp_thresh, n_thresh)
    delta_logp_thresh = np.diff(logp_thresholds[:2])[0]
    all_thresholds = norm.isf(np.exp(-1 * logp_thresholds))
    # convert first thresh (-inf) to zero, or just below our lowest data value
    all_thresholds[0] = min(0., data.min() - np.finfo(data.dtype).eps)
    return all_thresholds, delta_logp_thresh


def _find_clusters(data, threshold, adjacency):
    """Find indices of vertices that form clusters at the given threshold."""
    suprathresh = (data > threshold)
    # XXX      ↓↓↓ THIS IS THE TIE-IN TO EXISTING MNE-PYTHON CLUSTERING CODE
    clusters = mne.stats.cluster_level._get_components(suprathresh, adjacency)
    return clusters  # list of arrays of vertex numbers


def _get_cluster_sizes(clusters):
    """Get cluster sizes from the _find_clusters output (helper function)."""
    return np.array([len(clust) for clust in clusters], dtype=int)


def _cluster_size_density_factory(sizes, max_cluster_size):
    """Find empirically the distribution (density func) of cluster sizes."""
    unique_sizes = np.unique(sizes)
    if len(unique_sizes) == 0:
        return lambda x: np.atleast_1d(np.zeros_like(x, float))
    elif len(unique_sizes) == 1:
        # can't use gaussian_kde (LinAlgError); make unimodal prob mass func:
        return lambda x: np.atleast_1d(x == unique_sizes[0]).astype(float)
    else:
        counts = np.bincount(sizes)
        x = np.nonzero(counts)[0]
        y = counts[x]
        # we need to interp first before normalizing
        _x = np.arange(max_cluster_size) + 1
        _y = interp1d(x=x, y=y, fill_value=tuple(y[[0, -1]]),
                      bounds_error=False)(_x)
        _y = _y / _y.sum()
        return interp1d(x=_x, y=_y)


def _suprathresh_density_given_cluster_size(
        thresholds, all_thresholds, observed_cluster_size,
        source_activation_density_func,
        threshold_specific_cluster_size_density_func):
    """PDF of threshold or activation value, given an observed cluster size.

    Equivalent in pTFCE source code is dvox.clust(); Spisák et al. equation 2
    """
    numer = (source_activation_density_func(thresholds) *  # p(hᵢ)
             threshold_specific_cluster_size_density_func(
                 thresholds, observed_cluster_size))  # p(c|hᵢ)
    y = (source_activation_density_func(all_thresholds)  # p(h)
         * threshold_specific_cluster_size_density_func(
             all_thresholds, observed_cluster_size))  # p(c|h)
    assert np.isfinite(y).all()
    denom = trapezoid(x=all_thresholds, y=y)  # integral
    assert np.isfinite(denom)
    return numer / denom


def _prob_suprathresh_given_cluster_size(
        threshold, all_thresholds, observed_cluster_size,
        source_activation_density_func,
        threshold_specific_cluster_size_density_func):
    """pvox.clust()"""
    thresh_ix = all_thresholds.tolist().index(threshold)
    x = all_thresholds[thresh_ix:]
    y = _suprathresh_density_given_cluster_size(
        x, all_thresholds, observed_cluster_size,
        source_activation_density_func,
        threshold_specific_cluster_size_density_func)
    integral = trapezoid(x=x, y=y)
    return integral


def _aggregate_logp_vals(unaggregated_probs, delta_logp_thresh):
    """Perform p-value enhancement by aggregating across thresholds."""
    # avoid underflow
    finfo = np.finfo(unaggregated_probs.dtype)
    unaggregated_probs[unaggregated_probs == 0] = finfo.eps
    unaggregated_probs[unaggregated_probs == 1] = 1 - finfo.epsneg
    # S(x) = ∑ᵢ -log(P(V ≥ hᵢ|cᵢ)) at voxel position x   (Spisák et al. eq. 10)
    neglogp = np.sum(-np.log(unaggregated_probs), axis=0)
    # (sqrt(Δk * (8S(x) + Δk)) - Δk) / 2   (Spisák et al. eq. 9)
    radicand = delta_logp_thresh * (8 * neglogp + delta_logp_thresh)
    enhanced = (np.sqrt(radicand) - delta_logp_thresh) / 2
    # neglogp → regular p-values
    return np.exp(-1 * enhanced)


def ptfce(data, adjacency, noise, max_cluster_size, seed=None):
    """Perform pTFCE.

    Parameters
    ----------

    data : array-like, shape (n_points,)
        The input data, reshaped or raveled into a one-dimensional vector.
    adjacency :
        Matrix describing the adjacency of points in the data vector (for the
        purposes of cluster formation).
    noise : array-like, shape (n_iter, n_points)
        Simulated noise to use when constructing the null distributions.
    max_cluster_size : int
        Largest allowed cluster size (usually the number of vertices in a
        hemisphere).
    seed : None | int | np.random.Generator
        Source of randomness for the noise simulations.
    """
    # compute pTFCE thresholds
    all_thresholds, delta_logp_thresh = calc_thresholds(data, n_thresh=100)

    with timer('calculating source activation prior'):
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ this is the p(v) distribution
        source_activation_density_func = gaussian_kde(noise.ravel())

    with timer('finding clusters in noise simulations'):
        all_noise_clusters = list()
        for iter_ix, noise_iter in enumerate(noise):
            print(f'iteration {iter_ix}, threshold ', end='', flush=True)
            this_clusters = list()
            for thresh_ix, threshold in enumerate(all_thresholds):
                # progress bar
                if not thresh_ix % 5:
                    print(f'{thresh_ix} ', end='', flush=True)
                # compute cluster size prior
                clust = _find_clusters(noise_iter, threshold, adjacency)
                this_clusters.append(clust)
            print()
            all_noise_clusters.append(this_clusters)

    with timer('calculating cluster size distribution from noise'):
        # pool obs across epochs & thresholds → total prob of each cluster size
        all_noise_cluster_sizes = _get_cluster_sizes([
            clust for _iter in all_noise_clusters for thresh in _iter
            for clust in thresh])
        # get the PDF of cluster sizes in noise (across all thresholds)
        cluster_size_density_func = _cluster_size_density_factory(
            all_noise_cluster_sizes, max_cluster_size)

        sizes_at_thresh = list()
        for thresh_ix in range(len(all_thresholds)):
            # estimate prob. density of cluster size at each threshold: p(c|h)
            clusts_at_thresh = [
                _iter[thresh_ix] for _iter in all_noise_clusters]
            _sizes_at_thresh = _get_cluster_sizes(
                [clust for _iter in clusts_at_thresh for clust in _iter])
            sizes_at_thresh.append(_sizes_at_thresh)

    def threshold_specific_cluster_size_density_func(
            thresholds, observed_cluster_size):
        """PDF of cluster size, given threshold.

        Equivalent in pTFCE source code is dclust() which is derived from the
        Euler Characteristic Density of a gaussian field of given dimension.
        """
        this_thresholds = np.array(thresholds)
        thresh_ixs = np.nonzero(np.in1d(all_thresholds, this_thresholds))[0]
        densities = list()
        for thresh_ix in thresh_ixs:
            noise_cluster_sizes = sizes_at_thresh[thresh_ix]
            density_func = _cluster_size_density_factory(
                noise_cluster_sizes, max_cluster_size)
            density = np.atleast_1d(density_func(observed_cluster_size))[0]
            densities.append(density)
        return np.array(densities)

    # apply to the real data
    with timer('finding clusters in real data'):
        print('threshold number: ', end='', flush=True)
        unaggregated_probs = np.ones(
            (len(all_thresholds), *data.shape), dtype=float)
        all_data_clusters_by_thresh = list()
        all_data_cluster_sizes_by_thresh = list()
        for thresh_ix, threshold in enumerate(all_thresholds):
            # progress bar
            if not thresh_ix % 5:
                print(f'{thresh_ix} ', end='', flush=True)
            # find clusters in data STC
            data_clusters = _find_clusters(data, threshold, adjacency)
            data_cluster_sizes = _get_cluster_sizes(data_clusters)
            all_data_clusters_by_thresh.append(data_clusters)
            all_data_cluster_sizes_by_thresh.append(data_cluster_sizes)
            uniq_data_cluster_sizes = np.unique(data_cluster_sizes)
            # compute unaggregated probs. (the call to
            # _prob_suprathresh_given_cluster_size is slow, so do it only once
            # for each unique cluster size)
            uniq_data_cluster_probs = {
                size: _prob_suprathresh_given_cluster_size(
                    threshold, all_thresholds, size,
                    source_activation_density_func,
                    threshold_specific_cluster_size_density_func)
                for size in uniq_data_cluster_sizes}
            # prepare prob array that will zip with clusters
            data_cluster_probs = np.array(
                [uniq_data_cluster_probs[size] for size in data_cluster_sizes])
            # assign probs to vertices in thresh-appropriate slice of big array
            for clust, prob in zip(data_clusters, data_cluster_probs):
                # make sure we're not overwriting anything
                assert np.all(unaggregated_probs[thresh_ix][clust] == 1.)
                unaggregated_probs[thresh_ix][clust] = prob
        print()

    with timer('aggregating and adjusting probabilities'):
        _ptfce = _aggregate_logp_vals(unaggregated_probs, delta_logp_thresh)

    return (_ptfce,
            all_thresholds,
            unaggregated_probs,
            source_activation_density_func,
            all_noise_cluster_sizes,
            cluster_size_density_func,
            all_data_clusters_by_thresh,
            all_data_cluster_sizes_by_thresh)


def calc_thresholded_source_prior(threshold, noise):
    """Find empirically the probability of a source being suprathreshold.

    Vectorized over thresholds.
    """
    noise = np.atleast_2d(noise.ravel())          # (1, noise.size)
    thresh = np.atleast_2d(threshold).T           # (thresh.size, 1)
    suprathresh = (noise > thresh)                # (thresh.size, noise.size)
    n_suprathresh_src = suprathresh.sum(axis=-1)  # (thresh.size,)
    assert n_suprathresh_src.shape[0] == thresh.size
    return n_suprathresh_src / noise.size


def plot_null_distr(noise, n_iter, source_activation_density_func,
                    cluster_size_density_func, all_noise_cluster_sizes):
    import matplotlib.pyplot as plt
    # initialize figure
    fig, axs = plt.subplots(1, 3)
    subtitle = f'\n({n_iter} noise iterations)'
    # first plot: source activation density
    ax = axs[0]
    x = np.linspace(noise.min(), noise.max(), 100)
    y = source_activation_density_func(x)
    ax.plot(x, y)
    ax.set(title=f'source activation density{subtitle}',
           xlabel='activation', ylabel='density')
    # second plot: probability of suprathresholdness
    ax = axs[1]
    y = calc_thresholded_source_prior(threshold=x, noise=noise)
    ax.plot(x, y)
    ax.set(title=f'probability of suprathresholdness{subtitle}',
           xlabel='threshold', ylabel='probability')
    # third plot: cluster size density
    ax = axs[2]
    x = np.arange(all_noise_cluster_sizes.max()) + 1
    y = cluster_size_density_func(x)
    ax.semilogx(x, y)
    ax.set(title=f'cluster size density across all thresholds{subtitle}',
           xlabel='cluster size', ylabel='density')
    # layout
    fig.set_size_inches((12, 4))
    fig.subplots_adjust(bottom=0.15, wspace=0.4, left=0.075, right=0.95)
    return fig
