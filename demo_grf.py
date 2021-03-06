import matplotlib.pyplot as plt
import numpy as np
from FyeldGenerator import generate_field
from mne.stats import combine_adjacency
from rpy2.robjects import r
from rpy2.robjects.numpy2ri import activate, deactivate, numpy2rpy
from scipy.stats import multivariate_normal, zscore

from ptfce import plot_null_distr, ptfce, timer

# configuration variables
rng = np.random.default_rng(seed=15485863)  # the one millionth prime
n_iter = 20
shape = (100, 100)
save_rdata = False


def make_grf_noise(size, n_iter):
    """Simulate gaussian random fields."""
    result = np.empty((n_iter,) + tuple(size))

    def stat(size):
        return rng.normal(size=size) + 1j * rng.normal(size=size)

    def power(freq):
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning,
                       'divide by zero encountered in true_divide')
            return 1 / freq

    for _iter in range(n_iter):
        result[_iter] = generate_field(
            statistic=stat, power_spectrum=power, shape=size)
    return result


# make fake signal
background = make_grf_noise(shape, 1)[0]
indices = np.array(list(np.ndindex(shape))) - np.array(shape) // 2
mean = (0,) * len(shape)
signal = multivariate_normal(mean=mean).pdf(indices / 5).reshape(shape)
data = background + 2 * signal

# make adjacency
adjacency = combine_adjacency(*shape)

# make noise
noise = make_grf_noise(shape, n_iter)

# prep
data = zscore(data, axis=None)
_noise = zscore(noise.reshape(n_iter, -1), axis=-1)

# write data to R-friendly format (for testing/debugging)
if save_rdata:
    activate()
    rdata = numpy2rpy(data)
    r.assign('rdata', rdata)
    r('saveRDS(rdata, "sim-data.rds")')
    del rdata
    deactivate()

# compute pTFCE
with timer('running pTFCE'):
    (_ptfce,
     all_thresholds,
     unaggregated_probs,
     source_activation_density_func,
     all_noise_cluster_sizes,
     cluster_size_density_func,
     all_data_clusters_by_thresh,
     all_data_cluster_sizes_by_thresh
     ) = ptfce(data.ravel(), adjacency, _noise,
               max_cluster_size=np.multiply(*shape), seed=rng)


# convert back to original shape
pvals = _ptfce.reshape(shape)
enhanced_img = -1 * np.log10(pvals)

# # # # # #
# TESTING #
# # # # # #

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
titles = ('original', 'enhanced')
clim = (min(data.min(), enhanced_img.min()),
        max(data.max(), enhanced_img.max()))
for ax, title, array in zip(axs, titles, (data, enhanced_img)):
    ax.imshow(array, cmap='Greys', clim=clim)
    ax.set(title=title)
fig.tight_layout()
fig.colorbar(ax.images[0], ax=axs, shrink=0.5, label='z-score')
fig.savefig('figs/original-and-enhanced-grf-data.png')

fig = plot_null_distr(
    _noise, n_iter, source_activation_density_func, cluster_size_density_func,
    all_noise_cluster_sizes)
fig.savefig('figs/null-distribution-plots-grf.png')
