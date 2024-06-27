"""
dsp.py - Digital Signal Processing module for DAS4Whales

This module provides various functions for digital signal processing of DAS strain data.

Authors: Léa Bouffaut, Quentin Goestchel
Date: 2023-2024
"""

import numpy as np
import scipy.signal as sp
import librosa
import sparse
from scipy import ndimage
from numpy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift

# Transformations
def get_fx(trace, nfft):
    """
    Apply a fast Fourier transform (FFT) to each channel of the strain data matrix.

    Parameters
    ----------
    trace : np.ndarray
        A 2D array of shape (channel, time sample) containing the strain data in the spatio-temporal domain.
    nfft : int
        Number of time samples used for the FFT.

    Returns
    -------
    ndarray
        A 2D array of shape (channel, freq. sample) containing the strain data in the spatio-spectral domain.
    """

    fx = 2 * (abs(np.fft.fftshift(np.fft.fft(trace, nfft), axes=1)))
    fx /= nfft
    fx *= 10 ** 9
    return fx


def get_spectrogram(waveform, fs, nfft=128, overlap_pct=0.8):
    """
    Get the spectrogram of a single channel

    Parameters
    ----------
    waveform : np.ndarray
        Single channel temporal signal.
    fs : float
        The sampling frequency (Hz).
    nfft : int, optional
        Number of time samples used for the STFT. Default is 128.
    overlap_pct : float, optional
        Percentage of overlap in the spectrogram. Default is 0.8.

    Returns
    -------
    p : np.ndarray
        Spectrogram in dB scale (normalized by max).
    tt : np.ndarray
        Time vector.
    ff : ndarray
        Frequency vector.

    """
    spectrogram = np.abs(librosa.stft(
        y=waveform, n_fft=nfft,
        hop_length=int(np.floor(nfft * (1 - overlap_pct)))))

    # Axis
    height = spectrogram.shape[0]
    width = spectrogram.shape[1]

    tt = np.linspace(0, len(waveform)/fs, num=width)
    ff = np.linspace(0, fs / 2, num=height)
    p = 20 * np.log10(spectrogram / np.max(spectrogram))

    return p, tt, ff


# Filters
# f-k filters design functions
#TODO: uniformize and make a global function with choice of filter
# Version from MATLAB code
def fk_filter_design(trace_shape, selected_channels, dx, fs, cs_min=1400, cp_min=1450, cp_max=3400, cs_max=3500):
    # TODO: mark as deprecated, use hybrid_filter_design instead
    """
    Designs a f-k filter for DAS strain data
    Keeps by default data with propagation speed [1450-3400] m/s

    The transition band is inspired and adapted from Yi Lin's matlab fk function
    https://github.com/nicklinyi/seismic_utils/blob/master/fkfilter.m

    Parameters
    ----------
    trace_shape : tuple
        A tuple with the dimensions of the strain data in the spatio-temporal domain such as
        trace_shape = (trace.shape[0], trace.shape[1]) where dimensions are [channel x time sample].
    selected_channels : list
        A list of the selected channels number [start, end, step].
    dx : float
        Channel spacing (m).
    fs : float
        Sampling frequency (Hz).
    cs_min : float, optional
        Minimum selected sound speeds for the f-k passband filtering (m/s). Default is 1400 m/s.
    cp_min : float, optional
        Minimum selected sound speed for the f-k stopband filtering, values should frame [c_min and c_max] (m/s).
        Default is 1450 m/s.
    cp_max : float, optional
        Maximum selected sound speeds for the f-k passband filtering (m/s). Default is 3400 m/s.
    cs_max : float, optional
        Maximum selected sound speed for the f-k stopband filtering, values should frame [c_min and c_max] (m/s).
        Default is 3500 m/s.

    Returns
    -------
    fk_filter_matrix : ndarray
        A [channel x time sample] numpy array containing the f-k-filter.

    """

    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx

    # Get the dimensions of the trace data
    nnx, nns = trace_shape

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # Supress/hide the warning
    np.seterr(invalid='ignore')

    # Create the filter
    # Wave speed is the ratio between the frequency and the wavenumber
    fk_filter_matrix = np.ndarray(shape=(len(knum), len(freq)), dtype=float, order='F')

    # Going through wavenumbers
    for i in range(len(knum)):
        # Taking care of very small wavenumber to avoid 0 division
        if abs(knum[i]) < 0.005:
            fk_filter_matrix[i, :] = np.zeros(shape=[len(freq)], dtype=float, order='F')
        else:
            filter_line = np.ones(shape=[len(freq)], dtype=float, order='F')
            speed = abs(freq / knum[i])

            # Filter transition band, ramping up from cs_min to cp_min
            selected_speed_mask = ((speed >= cs_min) & (speed <= cp_min))
            filter_line[selected_speed_mask] = np.sin(0.5 * np.pi *
                                                      (speed[selected_speed_mask] - cs_min) / (cp_min - cs_min))
            # Filter transition band, going down from cp_max to cs_max
            selected_speed_mask = ((speed >= cp_max) & (speed <= cs_max))
            filter_line[selected_speed_mask] = 1 - np.sin(0.5 * np.pi *
                                                          (speed[selected_speed_mask] - cp_max) / (cs_max - cp_max))
            # Stopband
            filter_line[speed >= cs_max] = 0
            filter_line[speed < cs_min] = 0

            # Fill the filter matrix
            fk_filter_matrix[i, :] = filter_line
    
    # Display the filter
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(fk_filter_matrix, extent=[min(freq),max(freq),min(knum),max(knum)],aspect='auto')
    # plt.figure()
    # plt.plot(freq, fk_filter_matrix[2000, :])
    # plt.show()

    return fk_filter_matrix

# Infinite wave speed filter, sine tapers
def hybrid_filter_design(trace_shape, selected_channels, dx, fs, cs_min=1400., cp_min=1450., fmin=15., fmax=25., display_filter=False):
    """Designs an infinite wave speed bandpass f-k hybrid filter for DAS strain data
        Keeps by default data with propagation speed above 1450 m/s between [15 - 25] Hz (designed for fin whales)

    Parameters
    ----------
    trace_shape : tuple
        tuple with the dimensions of the strain data in the spatio-temporal domain such as trace_shape = (trace.shape[0], trace.shape[1]) where dimensions are [channel x time sample]
    selected_channels : list
        list of the selected channels number  [start, end, step]
    dx : float
        channel spacing (m)
    fs : float
        sampling frequency (Hz)
    cs_min : float, optional
        lower minimum selected sound speeds for the f-k highpass filtering (m/s), by default 1400 m/s
    cp_min : float, optional
        higher minimum selected sound speed for the f-k highpass filtering, by default 1450 m/s
    fmin : float, optional
        minimum frequency for the passband, by default 15
    fmax : float, optional
        maximum frequency for the passband, by default 25
    display_filter : bool, optional
        option for filter display, by default False

    Returns
    -------
    fk_filter_matrix : array-like
        [channel x time sample] a scipy sparse array containing the f-k-filter
    """    

    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx
    # Get the dimensions of the trace data
    nnx, nns = trace_shape

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # 1st step: frequency bandpass filtering
    H = np.zeros_like(freq)
    # set the width of the frequency range tapers
    df_taper = 4 # Hz
    # Apply it to the frequencies of interest
    fpmax = fmax + df_taper
    fpmin = fmin - df_taper
    # Find the corresponding indexes
    fmin_idx = np.argmax(freq >= fpmin)
    fmax_idx = np.argmax(freq >= fpmax)

    # Filter transition band, ramping up from fpmin to fmin
    rup_mask = ((freq >= fpmin) & (freq <= fmin))
    H[rup_mask] = np.sin(0.5 * np.pi * (freq[rup_mask] - fpmin) / (fmin - fpmin))
    # Filter passband
    H[(freq >= fmin) & (freq <= fmax)] = 1
    # Filter transition band, ramping down from fmax to fpmax
    rdo_mask = ((freq >= fmax) & (freq <= fpmax))
    H[rdo_mask] = np.cos(0.5 * np.pi * (freq[rdo_mask] - fmax) / (fmax - fpmax))

    # Replicate the bandpass frequency response along the k-axis to initialize the filter
    fk_filter_matrix = np.tile(H, (len(knum), 1))
    
    # 2nd step: filtering waves whose speeds are below cmin, with a taper between csmin and cpmin
    # Going through frequencies between the considered range of the bandpass filter
    for i in range(fmin_idx, fmax_idx):
        # Initiating filter column to zeros
        filter_col = np.zeros_like(knum)

        # Filter transition bands, ramping up from cs_min to cp_min
        ks = freq[i] / cs_min
        kp = freq[i] / cp_min
        
        # Avoid zero division
        if ks != kp:
            # f+ k+ quadrant                                             
            selected_k_mask = ((knum >= -ks) & (knum <= -kp))
            filter_col[selected_k_mask] = -np.sin(0.5 * np.pi * (knum[selected_k_mask] + ks) / (kp - ks))

            # f+ k- quadrant
            selected_k_mask = ((-knum >= -ks) & (-knum <= -kp))
            filter_col[selected_k_mask] = np.sin(0.5 * np.pi * (knum[selected_k_mask] - ks) / (kp - ks))

        # Passband
        # Positive frequencies (kp is positive):
        filter_col[(knum < kp) & (knum > -kp)] = 1

        # Fill the filter matrix by multiplication 
        fk_filter_matrix[:, i] *= filter_col 

    # Symmetrize the filter
    fk_filter_matrix += np.fliplr(fk_filter_matrix)

    # Filter display, optional
    if display_filter: 
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Context manager for the plot (to avoid changing the global settings)
        with plt.rc_context():
            # Change the font sizes for plots (if needed)
            # plt.rc('font', size=20) 
            # plt.rc('xtick', labelsize=16)  
            # plt.rc('ytick', labelsize=16)

            fig = plt.figure(figsize=(18, 10))
            gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[6, 2])

            ax1 = plt.subplot(gs[0])
            ax1.imshow(fk_filter_matrix, extent=[min(freq), max(freq), min(knum), max(knum)], aspect='auto')
            ax1.set_ylabel('k [m$^{-1}$]')
            ax1.set_xlabel('f [Hz]')
            
            ax2 = plt.subplot(gs[2], sharex=ax1)
            ax2.plot(freq, fk_filter_matrix[len(knum)//2, :], lw=3)
            ax2.set_xlabel('f [Hz]')
            ax2.set_ylabel('Gain []')
            ax2.set_xlim([min(freq), max(freq)])
            ax2.grid()

            ax3 = plt.subplot(gs[1], sharey=ax1)
            ax3.plot(fk_filter_matrix[:, fmin_idx + 250], knum, lw=3)
            ax3.set_xlabel('Gain []')
            ax3.set_ylabel('k [m$^{-1}$]')
            ax3.yaxis.set_label_position("right")
            ax3.set_ylim([min(knum), max(knum)])
            ax3.invert_xaxis()
            ax3.yaxis.tick_right()
            ax3.grid()
            plt.tight_layout()
            plt.show()

    return sparse.COO.from_numpy(fk_filter_matrix)

# Non-infinite wave speed filter, sine tapers
def hybrid_ninf_filter_design(trace_shape, selected_channels, dx, fs, cs_min=1400., cp_min=1450., cp_max=3400, cs_max=3500, fmin=15., fmax=25., display_filter=False):
    """Designs a bandpass f-k hybrid filter for DAS strain data
        Keeps by default data with propagation speed above 1450 m/s between [15 - 25] Hz (designed for fin whales)

    Parameters
    ----------
    trace_shape : tuple
        tuple with the dimensions of the strain data in the spatio-temporal domain such as trace_shape = (trace.shape[0], trace.shape[1]) where dimensions are [channel x time sample]
    selected_channels : list
        list of the selected channels number  [start, end, step]
    dx : float
        channel spacing (m)
    fs : float
        sampling frequency (Hz)
    cs_min : float, optional
        lower minimum selected sound speeds for the f-k highpass filtering (m/s), by default 1400 m/s
    cp_min : float, optional
        higher minimum selected sound speed for the f-k highpass filtering, by default 1450 m/s
    fmin : float, optional
        minimum frequency for the passband, by default 15
    fmax : float, optional
        maximum frequency for the passband, by default 25
    display_filter : bool, optional
        option for filter display, by default False

    Returns
    -------
    fk_filter_matrix : array-like
        [channel x time sample] a scipy sparse array containing the f-k-filter
    """    

    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx
    # Get the dimensions of the trace data
    nnx, nns = trace_shape

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # Replace the sine/cosine tapers by butterworth filters
    b, a = sp.butter(8, [fmin/(fs/2), fmax/(fs/2)], 'bp')
    H = np.concatenate((np.zeros(len(freq)//2), np.abs(sp.freqz(b, a, worN=len(freq)//2)[1]) ** 2))

    # 1st step: frequency bandpass filtering
    # H = np.zeros_like(freq)
    # set the width of the frequency range tapers, since butterworth filters are smooth
    df_taper = 14 # Hz
    # Apply it to the frequencies of interest
    fpmax = fmax + df_taper
    fpmin = fmin - df_taper
    # Find the corresponding indexes
    fmin_idx = np.argmax(freq >= fpmin)
    fmax_idx = np.argmax(freq >= fpmax)

    # Filter transition band, ramping up from fpmin to fmin, replaced by butterworth filter
    # rup_mask = ((freq >= fpmin) & (freq <= fmin))
    # H[rup_mask] = np.sin(0.5 * np.pi * (freq[rup_mask] - fpmin) / (fmin - fpmin))
    # # Filter passband
    # H[(freq >= fmin) & (freq <= fmax)] = 1
    # # Filter transition band, ramping down from fmax to fpmax
    # rdo_mask = ((freq >= fmax) & (freq <= fpmax))
    # H[rdo_mask] = np.cos(0.5 * np.pi * (freq[rdo_mask] - fmax) / (fmax - fpmax))

    # Replicate the bandpass frequency response along the k-axis to initialize the filter
    fk_filter_matrix = np.tile(H, (len(knum), 1))

    # 2nd step: filtering waves whose speeds are below cmin, with a taper between csmin and cpmin
    # Going through frequencies between the considered range of the bandpass filter
    for i in range(fmin_idx, fmax_idx):
        # Initiating filter column to zeros
        filter_col = np.zeros_like(knum)

        # Filter transition bands, ramping up from cs_min to cp_min
        ks_min = freq[i] / cs_max
        kp_min = freq[i] / cp_max

        ks_max = freq[i] / cs_min
        kp_max = freq[i] / cp_min
        
        # Avoid zero division
        if ks_min != kp_min:
            # f+ k+ quadrant ramp up                                         
            selected_k_mask = ((knum >= ks_min) & (knum <= kp_min))
            filter_col[selected_k_mask] = np.sin(0.5 * np.pi * (knum[selected_k_mask] - ks_min) / (kp_min - ks_min))
        if ks_max != kp_max:
            # f+ k- quadrant ramp down
            selected_k_mask = ((knum >= kp_max) & (knum <= ks_max))
            filter_col[selected_k_mask] = -np.sin(0.5 * np.pi * (knum[selected_k_mask] - ks_max) / (ks_max - kp_max))

        # Passband
        # Positive frequencies (kp_min is positive):
        filter_col[(knum > kp_min) & (knum < kp_max)] = 1

        # Fill the filter matrix by multiplication 
        fk_filter_matrix[:, i] *= filter_col 

    # Symmetrize the filter
    fk_filter_matrix += np.fliplr(fk_filter_matrix)
    fk_filter_matrix += np.flipud(fk_filter_matrix)

    # Filter display, optional
    if display_filter: 
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Context manager for the plot (to avoid changing the global settings)
        with plt.rc_context():
            # Change the font sizes for plots (if needed)
            # plt.rc('font', size=20) 
            # plt.rc('xtick', labelsize=16)  
            # plt.rc('ytick', labelsize=16)

            fig = plt.figure(figsize=(18, 10))
            gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[6, 2])

            # Matrix display
            ax1 = plt.subplot(gs[0])
            ax1.imshow(fk_filter_matrix, extent=[min(freq), max(freq), min(knum), max(knum)], aspect='auto', origin='lower')
            ax1.hlines(knum[len(knum)//2 + 420], min(freq), max(freq), color='tab:orange', lw=2, ls=':')
            ax1.vlines(freq[len(freq)//2 + 1500], min(knum), max(knum), color='tab:blue', lw=2, ls=':')
            # colorbar
            # cbar = plt.colorbar(ax1.imshow(fk_filter_matrix, extent=[min(freq), max(freq), min(knum), max(knum)], aspect='auto', origin='lower'))
            ax1.set_ylabel('k [m$^{-1}$]')
            ax1.set_xlabel('f [Hz]')
            
            # Frequency slice display
            ax2 = plt.subplot(gs[2], sharex=ax1)
            ax2.plot(freq, fk_filter_matrix[len(knum)//2 + 420, :], lw=3, color='tab:orange')
            ax2.set_xlabel('f [Hz]')
            ax2.set_ylabel('Gain []')
            ax2.set_xlim([min(freq), max(freq)])
            ax2.grid()

            # Wavenumber slice display
            ax3 = plt.subplot(gs[1], sharey=ax1)
            ax3.plot(fk_filter_matrix[:, len(freq)//2 + 1500], knum, lw=3, color='tab:blue')
            ax3.set_xlabel('Gain []')
            ax3.set_ylabel('k [m$^{-1}$]')
            ax3.yaxis.set_label_position("right")
            ax3.set_ylim([min(knum), max(knum)])
            ax3.invert_xaxis()
            ax3.yaxis.tick_right()
            ax3.grid()
            plt.tight_layout()
            plt.show()
            
    return sparse.COO.from_numpy(fk_filter_matrix)

# Infinite wave speed filter, gaussian tapers
def hybrid_gs_filter_design(trace_shape, selected_channels, dx, fs, cs_min=1400., cp_min=1450., fmin=15., fmax=25., display_filter=False):
    """Designs a bandpass f-k hybrid filter for DAS strain data
        Keeps by default data with propagation speed above 1450 m/s between [15 - 25] Hz (designed for fin whales)

    Parameters
    ----------
    trace_shape : tuple
        tuple with the dimensions of the strain data in the spatio-temporal domain such as trace_shape = (trace.shape[0], trace.shape[1]) where dimensions are [channel x time sample]
    selected_channels : list
        list of the selected channels number  [start, end, step]
    dx : float
        channel spacing (m)
    fs : float
        sampling frequency (Hz)
    cs_min : float, optional
        lower minimum selected sound speeds for the f-k highpass filtering (m/s), by default 1400 m/s
    cp_min : float, optional
        higher minimum selected sound speed for the f-k highpass filtering, by default 1450 m/s
    fmin : float, optional
        minimum frequency for the passband, by default 15
    fmax : float, optional
        maximum frequency for the passband, by default 25
    display_filter : bool, optional
        option for filter display, by default False

    Returns
    -------
    fk_filter_matrix : array-like
        [channel x time sample] a scipy sparse array containing the f-k-filter
    """    

    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx
    # Get the dimensions of the trace data
    nnx, nns = trace_shape

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # 1st step: frequency bandpass filtering
    H = np.zeros_like(freq)
    # set the width of the frequency range tapers
    df_taper = 4 # Hz
    # Apply it to the frequencies of interest
    fpmax = fmax + df_taper
    fpmin = fmin - df_taper
    # Find the corresponding indexes
    fmin_idx = np.argmax(freq >= fpmin)
    fmax_idx = np.argmax(freq >= fpmax)

    # Filter passband
    H[(freq >= fmin) & (freq <= fmax)] = 1

    # Replicate the bandpass frequency response along the k-axis to initialize the filter
    fk_filter_matrix = np.tile(H, (len(knum), 1))
    
    # 2nd step: filtering waves whose speeds are below cmin, with a taper between csmin and cpmin
    # Going through frequencies between the considered range of the bandpass filter
    for i in range(fmin_idx, fmax_idx):
        # Initiating filter column to zeros
        filter_col = np.zeros_like(knum)

        # Filter transition bands, ramping up from cs_min to cp_min
        ks = freq[i] / cs_min
        kp = freq[i] / cp_min
        
        # Avoid zero division
        if ks != kp:
            # f+ k+ quadrant                                             
            selected_k_mask = ((knum >= -ks) & (knum <= -kp))

            # f+ k- quadrant
            selected_k_mask = ((-knum >= -ks) & (-knum <= -kp))

        # Passband
        # Positive frequencies (kp is positive):
        filter_col[(knum < kp) & (knum > -kp)] = 1

        # Fill the filter matrix by multiplication 
        fk_filter_matrix[:, i] *= filter_col 

    # Symmetrize the filter
    fk_filter_matrix += np.fliplr(fk_filter_matrix)
    fk_filter_matrix = ndimage.gaussian_filter(fk_filter_matrix, 20)

    # Filter display, optional
    if display_filter: 
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        with plt.rc_context():
            # Change the font sizes for plots (if needed)
            # plt.rc('font', size=20) 
            # plt.rc('xtick', labelsize=16)  
            # plt.rc('ytick', labelsize=16)

            fig = plt.figure(figsize=(18, 10))
            gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[6, 2])

            ax1 = plt.subplot(gs[0])
            ax1.imshow(fk_filter_matrix, extent=[min(freq), max(freq), min(knum), max(knum)], aspect='auto')
            ax1.set_ylabel('k [m$^{-1}$]')
            ax1.set_xlabel('f [Hz]')
            
            ax2 = plt.subplot(gs[2], sharex=ax1)
            ax2.plot(freq, fk_filter_matrix[len(knum)//2, :], lw=3)
            ax2.set_xlabel('f [Hz]')
            ax2.set_ylabel('Gain []')
            ax2.set_xlim([min(freq), max(freq)])
            ax2.grid()

            ax3 = plt.subplot(gs[1], sharey=ax1)
            ax3.plot(fk_filter_matrix[:, fmin_idx + 250], knum, lw=3)
            ax3.set_xlabel('Gain []')
            ax3.set_ylabel('k [m$^{-1}$]')
            ax3.yaxis.set_label_position("right")
            ax3.set_ylim([min(knum), max(knum)])
            ax3.invert_xaxis()
            ax3.yaxis.tick_right()
            ax3.grid()
            plt.tight_layout()
            plt.show()

    return sparse.COO.from_numpy(fk_filter_matrix)

# Non-infinite wave speed filter, gaussian tapers
def hybrid_ninf_gs_filter_design(trace_shape, selected_channels, dx, fs, cs_min=1400., cp_min=1450., cp_max=3400, cs_max=3500, fmin=15., fmax=25., display_filter=False):
    """Designs a bandpass f-k hybrid filter for DAS strain data
        Keeps by default data with propagation speed above 1450 m/s between [15 - 25] Hz (designed for fin whales)

    Parameters
    ----------
    trace_shape : tuple
        tuple with the dimensions of the strain data in the spatio-temporal domain such as trace_shape = (trace.shape[0], trace.shape[1]) where dimensions are [channel x time sample]
    selected_channels : list
        list of the selected channels number  [start, end, step]
    dx : float
        channel spacing (m)
    fs : float
        sampling frequency (Hz)
    cs_min : float, optional
        lower minimum selected sound speeds for the f-k highpass filtering (m/s), by default 1400 m/s
    cp_min : float, optional
        higher minimum selected sound speed for the f-k highpass filtering, by default 1450 m/s
    fmin : float, optional
        minimum frequency for the passband, by default 15
    fmax : float, optional
        maximum frequency for the passband, by default 25
    display_filter : bool, optional
        option for filter display, by default False

    Returns
    -------
    fk_filter_matrix : array-like
        [channel x time sample] a scipy sparse array containing the f-k-filter
    """    

    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx
    # Get the dimensions of the trace data
    nnx, nns = trace_shape

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # 1st step: frequency bandpass filtering
    H = np.zeros_like(freq)
    # set the width of the frequency range tapers
    df_taper = 4 # Hz
    # Apply it to the frequencies of interest
    fpmax = fmax + df_taper
    fpmin = fmin - df_taper
    # Find the corresponding indexes
    fmin_idx = np.argmax(freq >= fpmin)
    fmax_idx = np.argmax(freq >= fpmax)

    # Filter passband
    H[(freq >= fmin) & (freq <= fmax)] = 1

    # Replicate the bandpass frequency response along the k-axis to initialize the filter
    fk_filter_matrix = np.tile(H, (len(knum), 1))
    
    # 2nd step: filtering waves whose speeds are below cmin, with a taper between csmin and cpmin
    # Going through frequencies between the considered range of the bandpass filter
    for i in range(fmin_idx, fmax_idx):
        # Initiating filter column to zeros
        filter_col = np.zeros_like(knum)

        # Filter transition bands, ramping up from cs_min to cp_min
        ks_min = freq[i] / cs_min
        kp_min = freq[i] / cp_min

        ks_max = freq[i] / cs_max
        kp_max = freq[i] / cp_max

        # Passband
        # Positive frequencies (kp_min is positive):
        filter_col[(knum > -kp_min) & (knum < -kp_max)] = 1

        # Fill the filter matrix by multiplication 
        fk_filter_matrix[:, i] *= filter_col 

    # Symmetrize the filter
    fk_filter_matrix = ndimage.gaussian_filter(fk_filter_matrix, 20)
    fk_filter_matrix += np.fliplr(fk_filter_matrix)
    fk_filter_matrix += np.flipud(fk_filter_matrix)

    # Filter display, optional
    if display_filter: 
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Context manager for the plot (to avoid changing the global settings)
        with plt.rc_context():
            # Change the font sizes for plots (if needed)
            # plt.rc('font', size=20) 
            # plt.rc('xtick', labelsize=16)  
            # plt.rc('ytick', labelsize=16)

            fig = plt.figure(figsize=(18, 10))
            gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[6, 2])

            ax1 = plt.subplot(gs[0])
            ax1.imshow(fk_filter_matrix, extent=[min(freq), max(freq), min(knum), max(knum)], aspect='auto')
            ax1.set_ylabel('k [m$^{-1}$]')
            ax1.set_xlabel('f [Hz]')
            
            ax2 = plt.subplot(gs[2], sharex=ax1)
            ax2.plot(freq, fk_filter_matrix[len(knum)//2 -50, :], lw=3)
            ax2.set_xlabel('f [Hz]')
            ax2.set_ylabel('Gain []')
            ax2.set_xlim([min(freq), max(freq)])
            ax2.grid()

            ax3 = plt.subplot(gs[1], sharey=ax1)
            ax3.plot(fk_filter_matrix[:, fmin_idx + 500], knum, lw=3)
            ax3.set_xlabel('Gain []')
            ax3.set_ylabel('k [m$^{-1}$]')
            ax3.yaxis.set_label_position("right")
            ax3.set_ylim([min(knum), max(knum)])
            ax3.invert_xaxis()
            ax3.yaxis.tick_right()
            ax3.grid()
            plt.tight_layout()
            plt.show()

    return sparse.COO.from_numpy(fk_filter_matrix)


def taper_data(trace):
    """
    Apply a Tukey window to each line (time series) of the input matrix.

    Parameters
    ----------
    trace : np.ndarray
        2D numpy array, where each column represents a time series.

    Returns
    -------
    np.ndarray
        Tapered matrix with the same shape as the input.
    """
    nt = trace.shape[1]
    # Change alpha to increase the tapering ratio
    trace *= sp.windows.tukey(nt, alpha=0.03)[np.newaxis, :]
    return trace


def fk_filter_filt(trace, fk_filter_matrix, tapering=False):
    """
    Applies a pre-calculated f-k filter to DAS strain data.

    Parameters
    ----------
    trace : np.ndarray
        A [channel x time sample] nparray containing the strain data in the spatio-temporal domain.
    fk_filter_matrix : np.ndarray
        A [channel x time sample] nparray containing the f-k-filter.
    tapering : bool, optional
        Flag indicating whether to apply tapering to the data. Default is False.

    Returns
    -------
    np.ndarray
        A [channel x time sample] nparray containing the f-k-filtered strain data in the spatio-temporal domain.
    """

    if tapering:
        trace = taper_data(trace)

    # Calculate the frequency-wavenumber spectrum
    fk_trace = np.fft.fftshift(np.fft.fft2(trace))

    # Apply the filter
    fk_filtered_trace = fk_trace * fk_filter_matrix

    # Back to the t-x domain
    trace = np.fft.ifft2(np.fft.ifftshift(fk_filtered_trace))

    return trace.real


def fk_filter_sparsefilt(trace, fk_filter_matrix, tapering=False):
    """
    Applies a pre-calculated f-k filter to DAS strain data

    Parameters
    ----------
    trace : np.ndarray
        A [channel x time sample] nparray containing the strain data in the spatio-temporal domain.
    fk_filter_matrix : np.ndarray
        A [channel x time sample] nparray containing the f-k-filter.

    Returns
    -------
    np.ndarray
        A [channel x time sample] nparray containing the f-k-filtered strain data in the spatio-temporal domain.
    """
    if tapering:
        trace = taper_data(trace)

    # Calculate the frequency-wavenumber spectrum
    fk_trace = np.fft.fftshift(np.fft.fft2(trace))

    # Apply the filter
    fk_filtered_trace = fk_trace * fk_filter_matrix
    # Back to the t-x domain
    trace = np.fft.ifft2(np.fft.ifftshift(fk_filtered_trace.todense()))

    return trace.real


def butterworth_filter(filterspec, fs):
    """
    Designs and applies a Butterworth filter.

    Parameters:
    ----------
    filterspec : tuple
        A tuple containing the filter order, critical frequency, and filter type.
    fs : float
        The sampling frequency.

    Returns:
    -------
    filter_sos : np.ndarray
        The second-order sections (SOS) representation of the Butterworth filter.

    Notes:
    ------
    The Butterworth filter is designed using the scipy.signal.butter function.

    Example:
    --------
    filter_order = 4
    filter_critical_freq = 1000
    filter_type_str = 'lowpass'
    filterspec = (filter_order, filter_critical_freq, filter_type_str)
    fs = 44100

    filter_sos = butterworth_filter(filterspec, fs)
    trace_filtered = sp.sosfiltfilt(filter_sos, trace_original, axis=1)
    """

    filter_order, filter_critical_freq, filter_type_str = filterspec
    # Build a filter of the desired type
    wn = np.array(filter_critical_freq) / (fs / 2)  # convert to angular frequency

    filter_sos = sp.butter(filter_order, wn, btype=filter_type_str, output='sos')

    return filter_sos


def instant_freq(channel, fs):
    """Compute the instantaneous frequency

    Parameters
    ----------
    channel : np.ndarray
        1D time series channel trace
    fs : float
        sampling frequency

    Returns
    -------
    np.ndarray
        instantaneous frequency along time[1:]
    """    
    # Compute the instantaneous frequency
    fi = np.diff(np.unwrap(np.angle(sp.hilbert(channel)))) / (2.0 * np.pi) * fs
    # Sliding window filtering to smooth out the fi
    # window_size = 50
    # ffi = np.convolve(fi, np.ones(window_size)/window_size, mode='same')
    # Compute the instantaneous median frequency
    # f, t, Zxx = sp.spectrogram(channel, fs, nperseg=140, noverlap=0.99)
    # cumulative_sum = np.cumsum(Zxx, axis=0)
    # Find the index corresponding to the median frequency at each time point
    # median_index = np.argmax(cumulative_sum >= 0.5 * cumulative_sum[-1], axis=0)
    # fm = f[median_index]
    return fi #, ffi, t, fm


def bp_filt(data,fs,fmin,fmax):
    """bp_filt - perform bandpass filtering on an array of DAS data

    Parameters
    ----------
    data : array-like
        array containing wave signal from DAS data
    fs : float
        sampling frequency
    fmin : float    
        minimum frequency for the passband
    fmax : float
        maximum frequency for the passband

    Returns
    -------
    tr_filt : array-like
        bandpass filtered data
    """
    b, a = sp.butter(8, [fmin/(fs/2), fmax/(fs/2)], 'bp')
    tr_filt = sp.filtfilt(b, a, data, axis=1)
    return tr_filt


def fk_filt(data,tint,fs,xint,dx,c_min,c_max):
    """fk_filt - perform fk filtering on an array of DAS data   

    Parameters
    ----------
    data : array-like
        array containing wave signal from DAS data
    tint : float
        decimation time interval between considered samples
    fs : float
        sampling frequency
    xint : float
        decimation space interval between considered samples
    dx : float
        spatial resolution
    c_min : float
        minimum phase speed for the pass-band filter in f-k domain
    c_max : float
        maximum phase speed for the pass-band filter in f-k domain

    Returns
    -------
    f : array-like
        vector of frequencies

    k : array-like
        vector of wavenumbers   
    g : array-like
        2D designed gaussian filter
    data_fft_g: array-like
        2D Fourier transformed data, filtered by g
    data_g.real: array-like
        Real value of spatiotemporal filtered data
    """    

    # Perform 2D Fourier Transform on the detrended input data
    data_fft = np.fft.fft2(data)
    # Make freq and wavenum vectors
    nx = data_fft.shape[0]
    ns = data_fft.shape[1]
    f = fftshift(fftfreq(ns, d = tint/fs))
    k = fftshift(fftfreq(nx, d = xint*dx))
    ff,kk = np.meshgrid(f,k)

    #  Define a filter in the f-k domain
    # Soundwaves have f/k = c so f = k*c

    g = 1.0*((ff < kk*c_min) & (ff < -kk*c_min))
    g2 = 1.0*((ff < kk*c_max) & (ff < -kk*c_max))

    # Symmetrize the filter
    g += np.fliplr(g)
    # g2 += np.fliplr(g2)
    g -= g2 + np.fliplr(g2) # combine to have g = g - g2
    
    # Apply Gaussian filter to the f-k filter
    # Tuning the standard deviation of the filter can improve computational efficiency
    g = ndimage.gaussian_filter(g, 20)
    # epsilon = 0.0001
    # g = np.exp (-epsilon*( ff-kk*c)**2 )

    # Normalize the filter to values between 0 and 1
    g = (g - np.min(g)) / (np.max(g) - np.min(g))

    # Apply the filter to the 2D Fourier-transformed data
    data_fft_g = fftshift(data_fft) * g
    # Perform inverse Fourier Transform to obtain the filtered data in t-x domain
    data_g = ifft2(ifftshift(data_fft_g))
    
    # return f, k, g, data_fft_g, data_g.real
    return data_g.real


def snr_tr_array(trace, env=False):
    """Calculate the 2D Signal-to-Noise Ratio (SNR) array for a given input trace.

    This function computes the SNR for each element in the input 2D trace array. The SNR
    is calculated as the ratio of the square of the trace values to the square of the
    standard deviation of the trace along the second axis (time).

    Parameters
    ----------
    trace : numpy.ndarray
        The input 2D trace array for which the SNR is to be calculated.

    Returns
    -------
    numpy.ndarray
        A 2D array containing the Signal-to-Noise Ratio (SNR) values for each element
        in the input trace.
    """  
    if env:
        return 10 * np.log10(abs(sp.hilbert(trace, axis=1)) ** 2 / np.std(trace, axis=1, keepdims=True) ** 2)
    return 10 * np.log10(trace ** 2 / np.std(trace, axis=1, keepdims=True) ** 2)
