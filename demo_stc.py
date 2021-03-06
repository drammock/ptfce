import os

import mne
import nibabel as nib
import numpy as np
from matplotlib.pyplot import close, ion
from scipy.stats import f as fdist
from scipy.stats import norm

from ptfce import plot_null_distr, ptfce, timer

rng = np.random.default_rng(seed=15485863)  # the one millionth prime
sample_data_folder = mne.datasets.sample.data_path()

# configuration variables
n_jobs = 6
n_iter = 20
verbose = False
# surf → full timecourse, surface source space
# vol  → full timecourse volume source space
# peak → single timepoint volume (for validation against R)
analysis = 'peak'
volume = analysis in ('peak', 'vol')


def get_sensor_data():
    """Load or compute Evoked."""
    print('Loading sample data')
    sample_data_raw_file = os.path.join(
        sample_data_folder, 'MEG', 'sample', 'sample_audvis_raw.fif')
    raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=verbose)
    events = mne.find_events(raw, stim_channel='STI 014')
    raw.pick(['grad'])  # ditch stim, ECG, EOG
    raw.drop_channels(raw.info['bads'])
    event_dict = {'auditory/left': 1}
    raw, events = raw.resample(sfreq=100, events=events, n_jobs=n_jobs)

    evk_fname = 'ptfce-ave.fif'
    cov_fname = 'ptfce-cov.fif'
    try:
        evoked = mne.read_evokeds(evk_fname)[0]
        noise_cov = mne.read_cov(cov_fname)
    except FileNotFoundError:
        noise_cov = mne.cov.compute_raw_covariance(raw, n_jobs=n_jobs)
        epochs = mne.Epochs(raw, events, event_id=event_dict, preload=False,
                            proj=True)
        evoked = epochs.average()
        mne.write_evokeds(evk_fname, evoked)
        mne.write_cov(cov_fname, noise_cov)
    return raw, evoked, noise_cov


def get_inverse(raw, noise_cov, subject, subjects_dir, n_jobs=1,
                volume=False, verbose=None):
    """Load or compute the inverse operator."""
    fname = 'ptfce-vol-inv.fif' if volume else 'ptfce-inv.fif'
    try:
        inverse = mne.minimum_norm.read_inverse_operator(fname)
        print('Loaded inverse from disk.')
    except FileNotFoundError:
        model = mne.make_bem_model(
            subject=subject, ico=4, subjects_dir=subjects_dir,
            verbose=verbose)
        bem = mne.make_bem_solution(model, verbose=verbose)
        trans = os.path.join(
            sample_data_folder, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
        if volume:
            src = mne.setup_volume_source_space(
                subject, pos=5.0, bem=bem, verbose=verbose)
        else:
            src = mne.setup_source_space(
                subject, spacing='oct6', add_dist='patch',
                subjects_dir=subjects_dir, verbose=verbose)
        fwd = mne.make_forward_solution(
            raw.info, trans=trans, src=src, bem=bem, meg=True, eeg=True,
            mindist=0, n_jobs=n_jobs, verbose=verbose)
        inverse = mne.minimum_norm.make_inverse_operator(
            raw.info, fwd, noise_cov, loose='auto', depth=0.8, verbose=verbose)
        mne.minimum_norm.write_inverse_operator(
            fname, inverse, verbose=verbose)
    return inverse


def make_noise(raw, noise_cov, seed=None, n_jobs=1, verbose=None):
    # instantiate random number generator
    rng = np.random.default_rng(seed=seed)
    # compute colorer
    whitener, ch_names, rank, colorer = mne.cov.compute_whitener(
        noise_cov, info=raw.info, picks=None, rank=None, scalings=None,
        return_rank=True, pca=False, return_colorer=True, verbose=verbose)
    # make appropriately-colored noise
    white_noise = rng.normal(size=(raw.info['nchan'], raw.n_times))
    colored_noise = colorer @ white_noise
    colored_raw = mne.io.RawArray(colored_noise, raw.info, raw.first_samp,
                                  verbose=verbose)
    # make sure it worked
    sim_cov = mne.cov.compute_raw_covariance(colored_raw, n_jobs=n_jobs)
    assert np.corrcoef(
        sim_cov.data.ravel(), noise_cov.data.ravel())[0, 1] > 0.999
    np.testing.assert_allclose(np.linalg.norm(sim_cov.data),
                               np.linalg.norm(noise_cov.data),
                               rtol=1e-2, atol=0.)
    return colored_raw


# # # # # # # # # # # #
# ACTUALLY RUN STUFF  #
# # # # # # # # # # # #

# get the sensor data
raw, evoked, noise_cov = get_sensor_data()

if analysis == 'peak':
    # reduce to 1 timepoint, for quicker comparison
    ch, t_peak = evoked.get_peak()
    evoked.crop(t_peak, t_peak)
    assert evoked.data.shape[1] == 1

# get the inverse operator
print('Creating inverse operator')
snr = 3.
lambda2 = 1. / snr ** 2
inverse_kwargs = dict(lambda2=lambda2, method='dSPM', pick_ori=None,
                      use_cps=True)
subject = 'sample'
subjects_dir = os.path.join(sample_data_folder, 'subjects')
inverse = get_inverse(
    raw, noise_cov, subject, subjects_dir, n_jobs=n_jobs, verbose=verbose,
    volume=volume)
src_adjacency = mne.spatial_src_adjacency(inverse['src'])

# automatically choose fewer noise simulations if there are more timepoints
n_iter = min(n_iter, np.ceil(n_iter / len(evoked.times)).astype(int))

# make STC from data
print('Creating STC from data')
stc = mne.minimum_norm.apply_inverse(
    evoked, inverse, verbose=verbose, **inverse_kwargs)
# extract real data as array
data = stc.data.ravel()
# expand adjacency to temporal dimension
adjacency = mne.stats.combine_adjacency(len(stc.times), src_adjacency)

max_cluster_size = max([len(hemi) for hemi in stc.vertices]) * len(stc.times)

# compute colorer
whitener, ch_names, rank, colorer = mne.cov.compute_whitener(
    noise_cov, info=raw.info, picks=None, rank=None, scalings=None,
    return_rank=True, pca=False, return_colorer=True, verbose=verbose)
# make appropriately-colored noise
white_noise = rng.normal(size=(n_iter, raw.info['nchan'], len(stc.times)))
colored_noise = colorer[np.newaxis, ...] @ white_noise
epochs = mne.EpochsArray(colored_noise, raw.info, tmin=stc.tmin)
# make STCs from noise
noise_stcs = mne.minimum_norm.apply_inverse_epochs(
    epochs, inverse, verbose=verbose, return_generator=False, **inverse_kwargs)
# extract all noise iterations as one array
noise = np.array([_stc.data.ravel() for _stc in noise_stcs])

# compute pTFCE
with timer('running pTFCE', oneline=False):
    (_ptfce,
     all_thresholds,
     unaggregated_probs,
     source_activation_density_func,
     all_noise_cluster_sizes,
     cluster_size_density_func,
     all_data_clusters_by_thresh,
     all_data_cluster_sizes_by_thresh
     ) = ptfce(data, adjacency, noise, max_cluster_size, seed=rng)

# copy results into STC container
stc_ptfce = stc.copy()
stc_ptfce.data = _ptfce.reshape(stc.data.shape)


# # # # # # # # # # # # #
# VISUALIZE THE RESULTS #
# # # # # # # # # # # # #

# convert p-values to -log10 p-values
stc_enh = stc_ptfce.copy()
stc_enh.data = -1 * np.log10(np.maximum(stc_enh.data, 1e-10))
# stc_enh.data = norm.isf(stc_ehn.data)  # better comparison?
# prep colormaps
pval_threshs = np.array([0.05, 0.001, 1e-10])
clim_enh = dict(kind='value', lims=tuple(-np.log10(pval_threshs)))
clim_orig = dict(kind='percent', lims=tuple(100 * (1 - pval_threshs)))
# auto clims yields: 96, 97.5, 99.95

# plot before/after on brains
for title, (_stc, clim) in dict(original=(stc, clim_orig),
                                # enhanced=(stc_ptfce, clim_orig),
                                neglogp=(stc_enh, clim_enh)
                                ).items():
    figname = f'figs/{title}-stc-data-{analysis}.png'
    if analysis == 'surf':
        fig = _stc.plot(title=title, clim=clim,
                        initial_time=_stc.get_peak()[1])
        fig.save_image(figname)
    elif analysis == 'vol':
        assert isinstance(stc, mne.VolSourceEstimate)
        # these nilearn-based plots block execution by default, so use ion
        with ion():
            fig = _stc.plot(src=inverse['src'], clim=clim, mode='glass_brain')
            fig.savefig(figname)
            close(fig)
    else:  # 'peak'
        # save volume as nifti, to compare with R implementation
        fname = f'ptfce_{title}'
        vol = _stc.as_volume(inverse['src'], mri_resolution=True)
        # remove singleton timepoint dimension
        if vol.shape[-1] == 1:
            vol.dataobj.shape = vol.shape[:-1]
        # let's keep everything in subject-coordinates when running in R, then
        # do necessary transforms in Python afterwards
        # (making sure to save the affine first so we can re-load it later)
        affine = vol.affine[:]
        np.save('affine.npy', affine)
        vol = nib.Nifti1Image(vol.dataobj, affine=np.eye(4))
        nib.save(vol, f'{fname}.nii.gz')
        # we also need to convert dSPM's F-values into the z-values that the
        # R implementation expects (and avoid introducing non-finite values
        # along the way)
        probs = fdist.cdf(vol.dataobj, dfn=3, dfd=raw.n_times)
        inf_mask = probs == 0.
        probs = np.nextafter(probs, np.inf, where=inf_mask, out=probs)
        z_vals = norm.ppf(probs)
        vol_z = nib.Nifti1Image(z_vals, vol.affine)
        nib.save(vol_z, f'{fname}_z.nii.gz')
        # make a brain mask (only need one)
        if title == 'original':
            brain_mask = (vol.dataobj > 0).astype(np.uint8)
            nib.save(nib.Nifti1Image(brain_mask, vol.affine),
                     'brain_mask.nii.gz')

# plot distributions and save
fig = plot_null_distr(
    noise, n_iter, source_activation_density_func, cluster_size_density_func,
    all_noise_cluster_sizes)
fig.savefig(f'figs/null-distribution-plots-{analysis}-stc.png')
