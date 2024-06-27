"""
detect.py - Detection module of DAS4Whales package

This module provides functions for detecting whale calls in DAS strain data.

Author: Quentin Goestchel
Date: 2023-2024
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy.stats as st
from tqdm import tqdm
from das4whales.plot import import_roseus 
import concurrent.futures

## Matched filter detection functions:
def gen_linear_chirp(fmin, fmax, duration, sampling_rate):
    """Generate a linear chirp signal.

    Parameters
    ----------
    fmin : float
        The starting frequency of the chirp signal.
    fmax : float
        The ending frequency of the chirp signal.
    duration : float
        The duration of the chirp signal in seconds.
    sampling_rate : int
        The sampling rate of the chirp signal in Hz.

    Returns
    -------
    numpy.ndarray
        The generated linear chirp signal.
    """
    t = np.arange(0, duration, 1/sampling_rate)
    y = sp.chirp(t, f0=fmax, f1=fmin, t1=duration, method='linear')
    return y


def gen_hyperbolic_chirp(fmin, fmax, duration, sampling_rate):
    """Generate a hyperbolic chirp signal.

    Parameters
    ----------
    fmin : float
        The starting frequency of the chirp signal.
    fmax : float
        The ending frequency of the chirp signal.
    duration : float
        The duration of the chirp signal in seconds.
    sampling_rate : int
        The sampling rate of the chirp signal in Hz.

    Returns
    -------
    numpy.ndarray
        The generated hyperbolic chirp signal.
    """
    t = np.arange(0, duration, 1/sampling_rate)
    y = sp.chirp(t, f0=fmax, f1=fmin, t1=duration, method='hyperbolic')
    return y


def gen_template_fincall(time, fs, fmin = 15., fmax = 25., duration = 1., window=True):
    """ generate template of a fin whale call

    Parameters
    ----------
    time : numpy.ndarray
        time vector
    fs : float
        sampling rate in Hz
    fmin : float, optional
        Minimum frequency, by default 15
    fmax : float, optional
        Maximum frequency, by default 25
    duration : float, optional
        Duration of the chirp signal in seconds, by default 1.
    """
    # 1 Hz frequency buffer to compensate the windowing
    df = 0
    chirp_signal = gen_hyperbolic_chirp(fmin-df, fmax + df, duration, fs)
    template = np.zeros(np.shape(time))
    #TODO: remove the padding and keep just the short window values
    if window:
        template[:len(chirp_signal)] = chirp_signal * np.hanning(len(chirp_signal))
    else: 
        template[:len(chirp_signal)] = chirp_signal
    return template


def shift_xcorr(x, y):
    """compute the shifted (positive lags only) cross correlation between two 1D arrays

    Parameters
    ----------
    x : numpy.ndarray
        1D array containing signal
    y : numpy.ndarray
        1D array containing signal

    Returns
    -------
    numpy.ndarray
        1D array cross-correlation betweem x and y, only for positive lags
    """
    corr = sp.correlate(x, y, mode='full', method='fft')
    return corr[len(x)-1 :]


def shift_nxcorr(x, y):
    """Compute the shifted (positive lags only) normalized cross-correlation with standard deviation normalization.

    Parameters
    ----------
    x : numpy.ndarray
        first input signal.
    y : numpy.ndarray
        second input signal

    Returns
    -------
    numpy.ndarray
        The normalized cross-correlation between the two signals
    """    
    #TODO: Modify to use with the short window values (mode = 'same' instead of 'full')
    # Compute cross-correlation
    cross_corr = sp.correlate(x, y, mode='full', method='fft')

    # Normalize using standard deviation
    normalized_corr = cross_corr / (np.std(x) * np.std(y) * len(x))
    
    return normalized_corr[len(x)-1 :]


def compute_cross_correlogram(data, template):
    """
    Compute the cross correlogram between the given data and template.

    Parameters
    ----------
    data : numpy.ndarray
        The input data array.
    template : numpy.ndarray
        The template array.

    Returns
    -------
    numpy.ndarray
        The cross correlogram array.
    """    
    # Normalize data along axis 1 by its maximum (peak normalization)
    norm_data = (data - np.mean(data, axis=1, keepdims=True)) / np.max(np.abs(data), axis=1, keepdims=True)
    template = (template - np.mean(template)) / np.max(np.abs(template))

    # Compute correlation along axis 1
    cross_correlogram = np.empty_like(data)

    for i in tqdm(range(data.shape[0])):
        cross_correlogram[i, :] = shift_xcorr(norm_data[i, :], template)

    return cross_correlogram


def pick_times_env(corr_m, threshold):
    # TODO: Parallelize this function
    """Detects the peak times in a correlation matrix.

    This function takes a correlation matrix, computes the Hilbert transform of each correlation,
    and detects the peak times based on a given threshold.

    Parameters
    ----------
    corr_m : numpy.ndarray
        The correlation matrix.
    threshold : float, optional
        The threshold value for peak detection. Defaults to 0.3.

    Returns
    -------
    list
        A list of arrays, where each array contains the peak indexes for each correlation.

    """
    peaks_indexes_m = []

    for corr in tqdm(corr_m, desc="Processing corr_m"):
        peaks_indexes = sp.find_peaks(abs(sp.hilbert(corr)), prominence=threshold)[0]  # Change distance in indexes, ex: 'distance=200'
        peaks_indexes_m.append(peaks_indexes)
    
    return peaks_indexes_m


def process_corr(corr, threshold):
    """Detects the peak times in a correlation serie, kernel for parallelization.

    This function takes a correlation serie, computes the Hilbert transform of the correlation, and detects the peak times based on a given threshold.

    Parameters
    ----------
    corr : np.ndarray
        The correlogram array.
    threshold : float, optional
        The threshold value for peak detection. Defaults to 0.3.

    Returns
    -------
    np.ndarray
        The peak indexes for the given correlation.

    """

    peaks_indexes = sp.find_peaks(abs(sp.hilbert(corr)), prominence=threshold)[0]
    return peaks_indexes


def pick_times_par(corr_m, threshold):
    """Detects the peak times in a correlation matrix using parallel processing.

    This function takes a correlation matrix, computes the Hilbert transform of each correlation,
    and detects the peak times based on a given threshold using parallel processing.

    Parameters
    ----------
    corr_m : numpy.ndarray
        The correlation matrix.
    threshold : float, optional
        The threshold value for peak detection. Defaults to 0.3.

    Returns
    -------
    list
        A list of arrays, where each array contains the peak indexes for each correlation.

    """

    peaks_indexes_m = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(process_corr, corr, threshold) for corr in corr_m]
        for result in concurrent.futures.as_completed(results):
            peaks_indexes_m.append(result.result())
    return peaks_indexes_m


def pick_times(corr_m, threshold):
    """Detects the peak times in a correlation matrix.

    This function takes a correlation matrix, computes the Hilbert transform of each correlation,
    and detects the peak times based on a given threshold.

    Parameters
    ----------
    corr_m : numpy.ndarray
        The correlation matrix.
    threshold : float, optional
        The threshold value for peak detection. Defaults to 0.3.

    Returns
    -------
    list
        A list of arrays, where each array contains the peak indexes for each correlation.

    """
    peaks_indexes_m = []

    for corr in tqdm(corr_m, desc="Processing corr_m"):
        peaks_indexes = sp.find_peaks(corr, prominence=threshold)[0]
        peaks_indexes_m.append(peaks_indexes)
    
    return peaks_indexes_m


def convert_pick_times(peaks_indexes_m):
    """
    Convert pick times from a list of lists to a numpy array.

    Parameters
    ----------
    peaks_indexes_m : list of lists
        A list of lists containing the pick times.

    Returns
    -------
    numpy.ndarray
        A numpy array containing a tuple (time index, spatial index) of the converted pick times.

    """
    peaks_indexes_tp = ([], [])

    for i in range(len(peaks_indexes_m)):
        nb_el = len(peaks_indexes_m[i])
        for j in range(nb_el):
            peaks_indexes_tp[0].append(i)
        for el in peaks_indexes_m[i]:
            peaks_indexes_tp[1].append(el)

    peaks_indexes_tp = np.asarray(peaks_indexes_tp)

    return peaks_indexes_tp


def select_picked_times(idx_tp, tstart, tend, fs):
    """
    Select the picked times within a given time range.

    Parameters
    ----------
    idx_tp : numpy.ndarray
        The time and spatial indexes of the picked times.
    tstart : float
        The starting time of the time range [s].
    tend : float
        The ending time of the time range [s].
    fs : float
        The sampling rate of the data.

    Returns
    -------
    numpy.ndarray
        The selected picked times within the given time range (time index, spatial index).

    """
    idx_tp_selected = (idx_tp[0][(idx_tp[1] >= tstart * fs) & (idx_tp[1] <= tend * fs)],
                        idx_tp[1][(idx_tp[1] >= tstart * fs) & (idx_tp[1] <= tend * fs)])

    return idx_tp_selected

## Spectrogram correlation functions:

def get_sliced_nspectrogram(trace, fs, fmin, fmax, nperseg, nhop, plotflag=False):
    """
    Compute the sliced non-stationary spectrogram of a given trace.

    Parameters
    ----------
    trace : numpy.ndarray
        The input trace signal.
    fs : float
        The sampling rate of the trace signal.
    fmin : float
        The minimum frequency of interest.
    fmax : float
        The maximum frequency of interest.
    nperseg : int
        The length of each segment for the spectrogram computation.
    nhop : int
        The number of samples to advance between segments.
    plotflag : bool, optional
        Whether to plot the spectrogram, defaults to False.

    Returns
    -------
    spectrogram : numpy.ndarray
        The computed spectrogram.
    ff : ndarray
        The frequency values of the spectrogram.
    tt : ndarray
        The time values of the spectrogram.

    Notes
    -----
    This function computes the non-stationary spectrogram of a given trace signal.
    The spectrogram is computed using the Short-Time Fourier Transform (STFT) with
    a specified segment length and hop size. The resulting spectrogram is then sliced
    between the specified minimum and maximum frequencies of interest.

    Examples
    --------
    >>> trace = np.random.randn(1000)
    >>> fs = 1000
    >>> fmin = 10
    >>> fmax = 100
    >>> nperseg = 256
    >>> nhop = 128
    >>> spectrogram, ff, tt = get_sliced_nspectrogram(trace, fs, fmin, fmax, nperseg, nhop, plotflag=True)
    """

    spectrogram = np.abs(librosa.stft(y=trace, n_fft=nperseg, hop_length=nhop))
    # Axis
    nf, nt = spectrogram.shape
    tt = np.linspace(0, len(trace)/fs, num=nt)
    ff = np.linspace(0, fs / 2, num=nf)
    p = spectrogram / np.max(spectrogram)

    # Slice the spectrogram betweem fmin and fmax
    ff_idx = np.where((ff >= fmin) & (ff <= fmax))
    p = p[ff_idx]
    ff = ff[ff_idx]

    if plotflag:
        roseus = import_roseus()
        fig, ax = plt.subplots(figsize=(12,4))
        shw = ax.pcolormesh(tt, ff, 20 * np.log10(p / np.max(p)), cmap=roseus, vmin=None, vmax=None)
        # Colorbar
        bar = fig.colorbar(shw, aspect=20, pad=0.015)
        bar.set_label('Normalized magnitude [-]')
        plt.xlim(0, len(trace)/fs)
        plt.ylim(fmin, fmax)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()
    
    return p, ff, tt


def buildkernel(f0, f1, bdwdth, dur, f, t, samp, fmin, fmax, plotflag=False):
    """
    Calculate kernel and plot.

    Parameters
    ----------
    f0 : float
        Starting frequency.
    f1 : float
        Ending frequency.
    bdwdth : float
        Frequency width of call.
    dur : float
        Call length (seconds).
    f : np.array
        Vector of frequencies returned from plotwav.
    t : np.array
        Vector of times returned from plotwav.
    samp : float
        Sample rate.
    plotflag : bool, optional
        If True, plots kernel. If False, no plot. Default is False.
    kernel_lims : tuple, optional
        Tuple of minimum kernel range and maximum kernel range. Default is finKernelLims.

    Returns
    -------
    tvec : numpy.array
        Vector of kernel times.
    fvec_sub : numpy.array
        Vector of kernel frequencies.
    BlueKernel : 2-d numpy.array
        Matrix of kernel values.

    Key variables:
    -------------
    tvec : numpy.array
        Kernel times (seconds).
    fvec : numpy.array
        Kernel frequencies.
    BlueKernel : numpy.array
        Matrix of kernel values.
    """

    # create a time vector of the same length as the call, with the same number of points as the spectrogram
    tvec = np.linspace(0, dur, np.size(np.nonzero((t < dur*8) & (t > dur*7)))) 
    # another way: int(dur * fs / (nperseg * (1-overlap_pct)) + 1)
    # define frequency span of kernel to match spectrogram
    fvec = f 
    # preallocate kernel array
    Kdist = np.zeros((len(fvec), len(tvec))) 
    ker_min, ker_max = fmin, fmax
    
    for j in range(len(tvec)):
        # calculate hat function that is centered on linearly decreasing
        # frequency values for each time in tvec
        # Linearly decreasing frequency values
        # x = fvec - (f0 + (tvec[j] / dur) * (f1 - f0))
        # Hyperbolic decreasing frequency values
        x = fvec - (f0 * f1 * dur / ((f0 - f1) * (tvec[j]) + f1 * dur))
        Kval = (1 - np.square(x) / (bdwdth * bdwdth)) * np.exp(-np.square(x) / (2 * (bdwdth * bdwdth)))
        # store hat function values in preallocated array
        Kdist[:, j] = Kval 
    BlueKernel = Kdist * np.hanning(len(tvec))[np.newaxis, :]
    # freq_inds = np.where(np.logical_and(fvec >= ker_min, fvec <= ker_max))
    
    # fvec_sub = fvec[freq_inds]
    # BlueKernel = BlueKernel_full[freq_inds, :][0]
    
    if plotflag:
        plt.figure(figsize=(1, 5))
        img = plt.pcolormesh(tvec, fvec, BlueKernel, cmap="RdBu_r", vmin=-np.max(np.abs(BlueKernel)), vmax=np.max(np.abs(BlueKernel)),)
        plt.axis([0, dur, np.min(fvec), np.max(fvec)])
        plt.colorbar(img, format='%.1f')
        plt.clim(-1, 1)
        plt.ylim(ker_min, ker_max)
        plt.title('Fin whale call kernel')
        plt.xlabel('t [s]')
        plt.ylabel('f [Hz]')
        plt.show()
        
    return tvec, fvec, BlueKernel


def buildkernel_from_template(fmin, fmax, dur, fs, nperseg, nhop, plotflag=False):
    """
    Build a kernel from a template.

    Parameters
    ----------
    fmin : float
        The minimum frequency of interest.
    fmax : float
        The maximum frequency of interest.
    dur : float
        The duration of the kernel in seconds.
    fs : float
        The sampling rate of the kernel in Hz.
    nperseg : int
        The length of each segment for the spectrogram computation.
    nhop : int
        The number of samples to advance between segments.
    plotflag : bool, optional
        Whether to plot the kernel, defaults to False.  

    Returns
    -------
    numpy.ndarray
        The computed kernel.

    """

    template = gen_hyperbolic_chirp(fmin, fmax, dur, fs)
    template *= np.hanning(len(template))
    spectro, ff, tt = get_sliced_nspectrogram(template, fs, fmin, fmax, nperseg, nhop, plotflag=False)

    if plotflag:
        roseus = import_roseus()
        fig, ax = plt.subplots(figsize=(2,4))
        shw = ax.pcolormesh(tt, ff, spectro, cmap=roseus, vmin=None, vmax=None)
        # Colorbar
        bar = fig.colorbar(shw, aspect=20, pad=0.015)
        bar.set_label('Normalized magnitude [-]')
        plt.xlim(0, dur)
        plt.ylim(fmin, fmax)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

    return spectro


def nxcorr2d(spectro, kernel):
    """
    Calculate the normalized cross-correlation between a spectrogram and a kernel.

    Parameters
    ----------
    spectro : numpy.ndarray
        The spectrogram array.
    kernel : numpy.ndarray
        The kernel array.

    Returns
    -------
    numpy.ndarray
        The maximum correlation values along the time axis.

    Notes
    -----
    The normalized cross-correlation is calculated using `scipy.signal.correlate2d`.
    The correlation values are normalized by dividing by the standard deviation of the spectrogram and the kernel,
    multiplied by the number of columns in the spectrogram.

    Examples
    --------
    >>> spectro = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> kernel = np.array([[1, 0], [0, 1]])
    >>> nxcorr2d(spectro, kernel)
    array([0.        , 0.33333333, 0.66666667])
    """
    correlation = sp.correlate(spectro, kernel, mode='same', method='fft') / (np.std(spectro) * np.std(kernel) * spectro.shape[1])
    maxcorr_t = np.max(correlation, axis=0)

    return maxcorr_t


def xcorr2d(spectro, kernel):
    """
    Calculate the 2D cross-correlation between a spectrogram and a kernel.

    Parameters
    ----------
    spectro : numpy.ndarray
        The input spectrogram array [frequency x time].

    kernel : numpy.ndarray
        The kernel array used for cross-correlation [frequency x time].

    Returns
    -------
    numpy.ndarray
        The resulting cross-correlation array.

    """
    correlation = sp.fftconvolve(spectro, np.flip(kernel, axis=1), mode='same', axes=1)
    maxcorr_t = np.sum(correlation, axis=0)
    maxcorr_t[maxcorr_t < 0] = 0
    maxcorr_t /= (np.median(spectro) * kernel.shape[1])

    return maxcorr_t


def xcorr(t, f, Sxx, tvec, fvec, BlueKernel):
    """
    Cross-correlate kernel with spectrogram

    Parameters
    ----------
    t : np.array
        Vector of times returned from plotwav
    f : np.array
        Vector of frequencies returned from plotwav
    Sxx : np.array
        2-D array of spectrogram amplitudes
    tvec : np.array
        Vector of times of kernel
    fvec : np.array
        Vector of frequencies of kernel
    BlueKernel : np.array
        2-D array of kernel amplitudes

    Returns
    -------
    t_scale : numpy.array
        Vector of correlation lags
    CorrVal : numpy.array
        Vector of correlation values

    """
    tvec_size = np.size(tvec)
    fvec_size = np.size(fvec)
    CorrVal = np.zeros(np.size(t) - (tvec_size-1))
    corrchunk= np.zeros((fvec_size, tvec_size))

    for ind1 in range(np.size(t) - tvec_size + 1):
        ind2 = ind1 + tvec_size
        corrchunk = Sxx[:fvec_size, ind1:ind2]
        CorrVal[ind1] = np.sum(BlueKernel * corrchunk)

    CorrVal /= (np.median(Sxx)*tvec_size)
    CorrVal[0] = 0
    CorrVal[-1] = 0
    CorrVal[CorrVal < 0] = 0
    t_scale = t[int(tvec_size / 2)-1:-int(np.ceil(tvec_size / 2))]
    return  [t_scale, CorrVal]


def compute_cross_correlogram_spectrocorr(data, fs, flims, kernel, win_size, overlap_pct):
    """Compute the cross-correlogram via spectrogram correlation.

    This function computes the cross-correlogram spectrocorr between the input data and a kernel.
    The cross-correlogram spectrocorr is a measure of similarity between the spectrogram of the input data
    and the kernel.

    Parameters
    ----------
    data : ndarray
        Input data array of shape (n, m), where n is the number of samples and m is the number of channels.
    fs : float
        Sampling frequency of the input data.
    flims : tuple
        Frequency limits (fmin, fmax) for the spectrogram computation.
    kernel : dict
        Dictionary containing the kernel parameters (f0, f1, duration, bandwidth).
    win_size : float
        Window size in seconds for the spectrogram computation.
    overlap_pct : float
        Percentage of overlap between consecutive windows for the spectrogram computation.

    Returns
    -------
    cross_correlogram : ndarray
        Cross-correlogram spectrocorr array of shape (n, p), where n is the number of samples and p is the number of time bins.
    """

    norm_data = (data - np.mean(data, axis=1, keepdims=True)) / np.max(np.abs(data), axis=1, keepdims=True)

    nperseg = int(win_size * fs)
    nhop = int(np.floor(nperseg * (1 - overlap_pct)))
    noverlap = nperseg - nhop
    print(f'nperseg: {nperseg}, noverlap: {noverlap}, hop_length: {nhop}')   
    fmin, fmax = flims

    # get call kernel attributes
    f1 = kernel["f1"] 
    f0 = kernel["f0"] 
    duration = kernel["dur"]
    bandwidth = kernel["bdwidth"]

    # check that hat function is within frequency range of spectrogram
    if fmax-f1 < 2 * bandwidth:
        fmax = f1 + 3 * bandwidth
    if f0-fmin < 2 * bandwidth: 
        fmin = f0 - 3 * bandwidth

    # Compute correlation along axis 1
    spectro, ff, tt = get_sliced_nspectrogram(data[0, :], fs, fmin, fmax, nperseg, nhop, plotflag=False)
    # TODO: Try weighting the spectrogram with the Cable frequency response (channel, bearing dependant)
    cross_correlogram = np.empty((data.shape[0], len(tt)))
    _, _, kernel = buildkernel(f0, f1, bandwidth, duration, ff, tt, fs, fmin, fmax, plotflag=False)
    # kernel = buildkernel_from_template(fmin, fmax, duration, fs, nperseg, nhop, plotflag=False)

    for i in tqdm(range(data.shape[0])):
        spectro, _, _ = get_sliced_nspectrogram(data[i, :], fs, fmin, fmax, nperseg, nhop, plotflag=False)
        cross_correlogram[i, :] = xcorr2d(spectro, kernel)

    return cross_correlogram