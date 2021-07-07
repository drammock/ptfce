import os
import numpy as np
import mne
from ptfce import timer, ptfce, plot_null_distr


rng = np.random.default_rng(seed=15485863)  # the one millionth prime
sample_data_folder = mne.datasets.sample.data_path()

# configuration variables
n_jobs = 4
n_iter = 20
verbose = False


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
                verbose=None):
    """Load or compute the inverse operator."""
    fname = 'ptfce-inv.fif'
    try:
        inverse = mne.minimum_norm.read_inverse_operator(fname)
        print('Loaded inverse from disk.')
    except FileNotFoundError:
        src = mne.setup_source_space(
            subject, spacing='oct6', add_dist='patch',
            subjects_dir=subjects_dir, verbose=verbose)
        model = mne.make_bem_model(
            subject='sample', ico=4, subjects_dir=subjects_dir,
            verbose=verbose)
        bem = mne.make_bem_solution(model, verbose=verbose)
        trans = os.path.join(
            sample_data_folder, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
        fwd = mne.make_forward_solution(
            raw.info, trans=trans, src=src, bem=bem, meg=True, eeg=True,
            mindist=0, n_jobs=n_jobs, verbose=verbose)
        inverse = mne.minimum_norm.make_inverse_operator(
            raw.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=verbose)
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

# get the inverse operator
print('Creating inverse operator')
snr = 3.
lambda2 = 1. / snr ** 2
inverse_kwargs = dict(lambda2=lambda2, method='dSPM', pick_ori=None,
                      use_cps=True)
subject = 'sample'
subjects_dir = os.path.join(sample_data_folder, 'subjects')
inverse = get_inverse(
    raw, noise_cov, subject, subjects_dir, n_jobs=n_jobs, verbose=verbose)
src_adjacency = mne.spatial_src_adjacency(inverse['src'])
adjacency = None

# reduce to 1 timepoint, for simplicity for now
ch, t_peak = evoked.get_peak()
evoked.crop(t_peak, t_peak)
assert evoked.data.shape[1] == 1

# make STC from data
print('Creating STC from data')
stc = mne.minimum_norm.apply_inverse(
    evoked, inverse, verbose=verbose, **inverse_kwargs)
# extract real data as array
data = stc.data.ravel()
# expand adjacency to temporal dimension
adjacency = mne.stats.combine_adjacency(len(stc.times), src_adjacency)

max_cluster_size = max([len(hemi) for hemi in stc.vertices])

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
with timer('running pTFCE'):
    (_ptfce,
     all_thresholds,
     unaggregated_probs,
     source_activation_density_func,
     all_noise_cluster_sizes,
     cluster_size_density_func,
     all_data_clusters_by_thresh,
     all_data_cluster_sizes_by_thresh
     ) = ptfce(data, adjacency, noise, max_cluster_size, seed=rng)

# convert back into STC
stc_ptfce = stc.copy()
stc_ptfce.data = _ptfce.reshape(stc.data.shape)


# # # # # #
# TESTING #
# # # # # #

foo = stc_ptfce.copy()
foo.data = -1 * np.log10(np.maximum(foo.data, 1e-10))
fig1 = stc.plot(title='original')
fig2 = foo.plot(title='enhanced')
fig1.save_image('figs/original-stc-data.png')
fig2.save_image('figs/enhanced-stc-data.png')

fig = plot_null_distr(
    noise, n_iter, source_activation_density_func, cluster_size_density_func,
    all_noise_cluster_sizes)
fig.savefig('figs/null-distribution-plots-stc.png')
