"""
plot.py - Plotting module for DAS4Whales

This module provides functions for plotting das data and data products in various formats.

Authors: Léa Bouffaut, Quentin Goestchel
Date: 2023-2024
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import scipy.signal as sp
from das4whales.dsp import get_fx, instant_freq
from datetime import datetime

def plot_rawdata(trace, time, dist, fig_size=(12, 10)):
    """
    Plot the raw DAS data.

    Parameters
    ----------
    trace : ndarray
        The DAS trace data.
    time : ndarray
        The time values corresponding to the trace data.
    dist : ndarray
        The distance values corresponding to the trace data.
    fig_size : tuple, optional
        The size of the figure in inches, by default (12, 10).
    """    

    fig = plt.figure(figsize=fig_size)
    wv = plt.imshow(trace * 1e9, aspect='auto', cmap='RdBu', extent=[min(time),max(time),min(dist)*1e-3,max(dist)*1e-3], origin='lower', vmin=-500, vmax=500)
    plt.title('Raw DAS data')
    plt.ylabel('Distance [km]')
    plt.xlabel('Time [s]')
    bar = fig.colorbar(wv, aspect=30, pad=0.015)
    bar.set_label(label='Strain [-] x$10^{-9}$)')
    plt.show()


def plot_tx(trace, time, dist, file_begin_time_utc=0, fig_size=(12, 10), v_min=None, v_max=None):
    """
    Spatio-temporal representation (t-x plot) of the strain data

    Parameters:
    ----------
    trace : np.ndarray
        A [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    time : np.ndarray
        The corresponding time vector
    dist : np.ndarray
        The corresponding distance along the FO cable vector
    file_begin_time_utc : int or datetime.datetime, optional
        The time stamp of the represented file (default is 0)
    fig_size : tuple, optional
        Tuple of the figure dimensions (default is (12, 10))
    v_min : float, optional
        Sets the min nano strain amplitudes of the colorbar (default is None)
    v_max : float, optional
        Sets the max nano strain amplitudes of the colorbar (default is None)

    Returns:
    -------
    None

    Notes:
    ------
    This function plots a spatio-temporal representation (t-x plot) of the strain data. It uses the given strain data,
    time vector, and distance vector to create the plot. The plot shows the strain envelope as a color map, with time
    on the x-axis and distance on the y-axis. The color of each point in the plot represents the strain amplitude at
    that point. The function also supports customizing the figure size, colorbar limits, and title.

    """

    fig = plt.figure(figsize=fig_size)
    #TODO determine if the envelope should be implemented here rather than just abs
    # Replace abs(trace) per abs(sp.hilbert(trace, axis=1)) ? 
    shw = plt.imshow(abs(trace) * 10 ** 9, extent=[time[0], time[-1], dist[0] * 1e-3, dist[-1] * 1e-3, ], aspect='auto',
                     origin='lower', cmap='turbo', vmin=v_min, vmax=v_max)
    plt.ylabel('Distance (km)')
    plt.xlabel('Time (s)')
    bar = fig.colorbar(shw, aspect=30, pad=0.015)
    bar.set_label('Strain Envelope (x$10^{-9}$)')

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')
    plt.tight_layout()
    plt.show()

    return


def plot_fx(trace, dist, fs, file_begin_time_utc=0, win_s=2, nfft=4096, fig_size=(12, 10), f_min=0,
            f_max=100, v_min=None, v_max=None):
    """
    Spatio-spectral (f-k plot) of the strain data

    Parameters
    ----------
    trace : np.ndarray
        A [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    dist : np.ndarray
        The corresponding distance along the FO cable vector
    fs : float
        The sampling frequency (Hz)
    file_begin_time_utc : int or datetime.datetime, optional
        The time stamp of the represented file, by default 0
    win_s : int, optional
        The duration of each f-k plot (s), by default 2
    nfft : int, optional
        Number of time samples used for the FFT, by default 4096
    fig_size : tuple, optional
        Tuple of the figure dimensions, by default (12, 10)
    f_min : int, optional
        Displayed minimum frequency interval (Hz), by default 0
    f_max : int, optional
        Displayed maximum frequency interval (Hz), by default 100
    v_min : float, optional
        Set the min nano strain amplitudes of the colorbar, by default None
    v_max : float, optional
        Set the max nano strain amplitudes of the colorbar, by default None

    Returns
    -------
    None

    Notes
    -----
    This function plots the spatio-spectral (f-k plot) of the strain data.

    - The number of subplots is evaluated based on the duration of each f-k plot.
    - The frequency axis is created using the FFT.
    - The strain data is processed and plotted for each subplot.

    Examples
    --------
    >>> plot_fx(trace, dist, fs, file_begin_time_utc=0, win_s=2, nfft=4096, fig_size=(12, 10), f_min=0,
                f_max=100, v_min=None, v_max=None)
    """

    # Evaluate the number of subplots
    nb_subplots = int(np.ceil(trace.shape[1] / (win_s * fs)))

    # Create the frequency axis
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))

    # Prepare the plot
    rows = 3
    cols = int(np.ceil(nb_subplots/rows))

    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    # Run through the data
    for ind in range(nb_subplots):
        fx = get_fx(trace[:, int(ind * win_s * fs):int((ind + 1) * win_s * fs):1], nfft)
        # fx = np.transpose(fx) - np.mean(fx, axis=1)
        # fx = np.transpose(fx)

        # Plot
        r = ind // cols
        c = ind % cols
        ax = axes[r][c]

        shw = ax.imshow(fx, extent=[freq[0], freq[-1], dist[0] * 1e-3, dist[-1] * 1e-3], aspect='auto',
                        origin='lower', cmap='jet', vmin=v_min, vmax=v_max)

        ax.set_xlim([f_min, f_max])
        if r == rows-1:
            ax.set_xlabel('Frequency (Hz)')
        else:
            ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)

        if c == 0:
            ax.set_ylabel('Distance (km)')
        else:
            ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    # Colorbar
    bar = fig.colorbar(shw, ax=axes.ravel().tolist())
    bar.set_label('Strain (x$10^{-9}$)')
    plt.show()


def plot_spectrogram(p, tt, ff, fig_size=(17, 5), v_min=None, v_max=None, f_min=None, f_max=None):
    """
    Plot a spectrogram.

    Parameters
    ----------
    p : ndarray
        Spectrogram values in dB.
    tt : ndarray
        Associated time vector (s).
    ff : ndarray
        Associated frequency vector (Hz).
    fig_size : tuple, optional
        Tuple of the figure dimensions. Default is (17, 5).
    v_min : float, optional
        Minimum dB strain amplitudes of the colorbar.
    v_max : float, optional
        Maximum dB strain amplitudes of the colorbar.
    f_min : float, optional
        Minimum frequency for the spectrogram display.
    f_max : float, optional
        Maximum frequency for the spectrogram display.

    Returns
    -------
    None

    """
    roseus = import_roseus()
    fig, ax = plt.subplots(figsize=fig_size)

    shw = ax.pcolormesh(tt, ff, p, shading='auto', cmap=roseus, vmin=v_min, vmax=v_max)
    ax.set_ylim(f_min, f_max)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # Colorbar
    bar = fig.colorbar(shw, aspect=30, pad=0.015)
    bar.set_label('dB (strain x$10^{-9}$)')
    plt.show()


def plot_3calls(channel, time, t1, t2, t3):
    """
    Plot the strain channel with 3 calls highlighted.

    Parameters
    ----------
    channel : np.ndarray
        The strain channel.
    time : np.ndarray
        The time values.
    t1 : float
        The time of the first call.
    t2 : float
        The time of the second call.
    t3 : float
        The time of the third call.

    Returns
    -------
    None

    """
    
    plt.figure(figsize=(12,4))

    plt.subplot(211)
    plt.plot(time, channel, ls='-')
    plt.xlim([time[0], time[-1]])
    plt.ylabel('strain [-]')
    plt.grid()
    plt.tight_layout()

    plt.subplot(234)
    plt.plot(time, channel)
    plt.ylabel('strain [-]')
    plt.xlabel('time [s]')
    plt.xlim([t1, t1+2.])
    plt.grid()
    plt.tight_layout()

    plt.subplot(235)
    plt.plot(time, channel)   
    plt.xlim([t2, t2+2.])
    plt.xlabel('time [s]')
    plt.grid()
    plt.tight_layout()

    plt.subplot(236)
    plt.plot(time, channel)   
    plt.xlim([t3, t3+2.])
    plt.xlabel('time [s]')
    plt.grid()
    plt.tight_layout()

    # plt.savefig('3calls.pdf', format='pdf')
    plt.show()

    return


def design_mf(trace, hnote, lnote, th, tl, time, fs):
    """Plot to design the matched filter 

    Parameters
    ----------
    trace : numpy.ndarray
        1D time series channel trace
    hnote : numpy.ndarray
        1D time series high frequency note template
    lnote : numpy.ndarray
        1D time series low frequency note template
    th : float
        start time of the high frequency note
    tl : float
        start time of the low frequency note
    time : numpy.ndarray
        1D vector of time values
    fs : float
        sampling frequency
    """    

    nf = int(th * fs)
    nl = int(tl * fs)
    # Create a dummy channel made of two notes at given times (not robust)
    dummy_chan = np.zeros_like(hnote)
    dummy_chan[nf:] = hnote[:-nf]
    dummy_chan[nl:] = lnote[:-nl]

    # Matched filter instantaneous freq
    fi = instant_freq(trace, fs)
    fi_mf = instant_freq(dummy_chan, fs)

    # Plot the generated linear chirp signal
    plt.figure(figsize=(18, 4))
    plt.subplot(121)
    plt.plot(time, (trace) / (np.max(abs(trace))), label='normalized measured fin call')
    plt.plot(time, (dummy_chan) / (np.max(abs(dummy_chan))), label='template')
    plt.title('fin whale call template - HF note')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.xlim(th-0.5, th+1.5)
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(time[1:], fi, label='measured fin call')
    plt.plot(time[1:], fi_mf, label='template')
    plt.xlim([th-0.5, th+1.5])
    plt.ylim([15., 35])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Instantaneous frequency [Hz]')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.subplot(121)
    plt.plot(time, (trace - np.mean(trace)) / (np.max(abs(trace))), label='normalized measured fin call')
    plt.plot(time, (dummy_chan - np.mean(dummy_chan)) / (np.max(abs(dummy_chan))), label='template')
    plt.title('fin whale call template - LF note')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.xlim([tl-0.5, tl+1.5])
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(time[1:], fi, label='measured fin call')
    plt.plot(time[1:], fi_mf, label='template')
    plt.xlim([tl-0.5, tl+1.5])
    plt.ylim([12., 28.])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Instantaneous frequency [Hz]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return


def detection_mf(trace, peaks_idx_HF, peaks_idx_LF, time, dist, fs, dx, selected_channels, file_begin_time_utc=None):
    """Plot the strain trace matrix [dist x time] with call detection above it

    Parameters
    ----------
    trace : numpy.ndarray
        [channel x time sample] array containing the strain data in the spatio-temporal domain
    peaks_idx_HF : tuple
        tuple of lists containing the detected call indexes coordinates (first list: channel idx, second list: time idx) for the high frequency call
    peaks_idx_LF : tuple
        tuple of lists containing the detected call indexes coordinates (first list: channel idx, second list: time idx) for the low frequency call
    time : numpy.ndarray
        time vector
    dist : numpy.ndarray
        distance vector along the cable
    fs : float
        sampling frequency
    dx : float
        spatial step
    selected_channels : list
        list of selected channels indexes [start, stop, step]
    file_begin_time_utc : int, optional
        time stamp of file, by default 0
    """    

    fig = plt.figure(figsize=(12,10))
    cplot = plt.imshow(abs(sp.hilbert(trace, axis=1)) * 1e9, extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='jet', origin='lower',  aspect='auto', vmin=0, vmax=0.4, alpha=0.35)
    plt.scatter(peaks_idx_HF[1] / fs, (peaks_idx_HF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='red', marker='.', label='HF_note')
    plt.scatter(peaks_idx_LF[1] / fs, (peaks_idx_LF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='green', marker='.', label='LF_note')
    bar = fig.colorbar(cplot, aspect=30, pad=0.015)
    bar.set_label('Strain Envelope [-] (x$10^{-9}$)')
    plt.xlabel('Time [s]')  
    plt.ylabel('Distance [km]')
    plt.legend(loc="upper right")
    # plt.savefig('test.pdf', format='pdf')

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    plt.tight_layout()
    plt.show()

    return


def detection_spectcorr(trace, peaks_idx_HF, peaks_idx_LF, time, dist, spectro_fs, dx, selected_channels, file_begin_time_utc=None):
    """Plot the strain trace matrix [dist x time] with call detection above it

    Parameters
    ----------
    trace : numpy.ndarray
        [channel x time sample] array containing the strain data in the spatio-temporal domain
    peaks_idx_HF : tuple
        tuple of lists containing the detected call indexes coordinates (first list: channel idx, second list: time idx) for the high frequency call
    peaks_idx_LF : tuple
        tuple of lists containing the detected call indexes coordinates (first list: channel idx, second list: time idx) for the low frequency call
    time : numpy.ndarray
        time vector
    dist : numpy.ndarray
        distance vector along the cable
    spectro_fs : float
        sampling frequency of the spectrograms
    dx : float
        spatial step
    selected_channels : list
        list of selected channels indexes [start, stop, step]
    file_begin_time_utc : int, optional
        time stamp of file, by default 0
    """    

    fig = plt.figure(figsize=(12,10))
    cplot = plt.imshow(abs(sp.hilbert(trace, axis=1)) * 1e9, extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='jet', origin='lower',  aspect='auto', vmin=0, vmax=0.4, alpha=0.35)
    plt.scatter(peaks_idx_HF[1] / spectro_fs, (peaks_idx_HF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='red', marker='x', label='HF call')
    plt.scatter(peaks_idx_LF[1] / spectro_fs, (peaks_idx_LF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='green', marker='.', label='LF_note')

    bar = fig.colorbar(cplot, aspect=30, pad=0.015)
    bar.set_label('Strain Envelope [-] (x$10^{-9}$)')
    plt.xlabel('Time [s]')  
    plt.ylabel('Distance [km]')
    plt.legend(loc="upper right")
    # plt.savefig('test.pdf', format='pdf')

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    plt.tight_layout()
    plt.show()

    return


def detection_grad(trace, peaks_idx, time, dist, fs, dx, selected_channels, file_begin_time_utc=None):
    """Plot the strain trace matrix [dist x time] with call detection above it

    Parameters
    ----------
    trace : numpy.ndarray
        [channel x time sample] array containing the strain data in the spatio-temporal domain
    peaks_idx_HF : tuple
        tuple of lists containing the detected call indexes coordinates (first list: channel idx, second list: time idx) for the high frequency call
    peaks_idx_LF : tuple
        tuple of lists containing the detected call indexes coordinates (first list: channel idx, second list: time idx) for the low frequency call
    time : numpy.ndarray
        time vector
    dist : numpy.ndarray
        distance vector along the cable
    fs : float
        sampling frequency
    dx : float
        spatial step
    selected_channels : list
        list of selected channels indexes [start, stop, step]
    file_begin_time_utc : int, optional
        time stamp of file, by default 0
    """    

    fig = plt.figure(figsize=(12,10))
    cplot = plt.imshow(abs(sp.hilbert(trace, axis=1)) * 1e9, extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='jet', origin='lower',  aspect='auto', vmin=0, vmax=0.4, alpha=0.35)
    plt.scatter(peaks_idx[1] / fs, (peaks_idx[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='red', marker='x', label='Fin call')
    bar = fig.colorbar(cplot, aspect=30, pad=0.015)
    bar.set_label('Strain Envelope [-] (x$10^{-9}$)')
    plt.xlabel('Time [s]')  
    plt.ylabel('Distance [km]')
    plt.legend(loc="upper right")
    # plt.savefig('test.pdf', format='pdf')

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    plt.tight_layout()
    plt.show()

    return


def snr_matrix(snr_m, time, dist, vmax, file_begin_time_utc=None, title=None):
    """Matrix plot of the local signal to noise ratio (SNR)

    Parameters
    ----------
    snr_m : numpy.ndarray
        [channel x time sample] array containing the SNR in the spatio-temporal domain
    time : nummpy.ndarray
        time vector
    dist : numpy.ndarray
        distance vector along the cable
    vmax : float
        maximun value of the plot (dB)
    """    
    fig = plt.figure(figsize=(12, 10))
    snrp = plt.imshow(snr_m, extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='turbo', origin='lower',  aspect='auto', vmin=0, vmax=vmax)
    bar = fig.colorbar(snrp, aspect=30, pad=0.015)
    bar.set_label('SNR [dB]')
    bar.ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.0f'))
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    
    if isinstance(file_begin_time_utc, datetime):
        if isinstance(title, str):
            plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S")+'/ '+title, loc='right')
        else:
            plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    plt.tight_layout()
    plt.show()

    return


def plot_cross_correlogramHL(corr_m_HF, corr_m_LF, time, dist, maxv, minv=0, file_begin_time_utc=None):
    """
    Plot the cross-correlogram between HF and LF notes.

    Parameters
    ----------
    corr_m_HF : numpy.ndarray
        The cross-correlation matrix of the HF notes.
    corr_m_LF : numpy.ndarray
        The cross-correlation matrix of the LF notes.
    time : numpy.ndarray
        The time values.
    dist : numpy.ndarray
        The distance values.
    maxv : float
        The maximum value for the colorbar.
    minv : int, optional
        The minimum value for the colorbar. Default is 0.
    file_begin_time_utc : datetime.datetime, optional
        The beginning time of the file in UTC. Default is None.

    Returns
    -------
    None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    im1 = ax1.imshow(abs(sp.hilbert(corr_m_HF, axis=1)), extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='turbo', origin='lower', aspect='auto', vmin=minv, vmax=maxv) 
    ax1.set_xlabel('Time [s]') 
    ax1.set_ylabel('Distance [km]') 
    ax1.set_title('HF note', loc='right')

    im2 = ax2.imshow(abs(sp.hilbert(corr_m_LF, axis=1)), extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='turbo',origin='lower', aspect='auto', vmin=minv, vmax=maxv) 
    ax2.set_xlabel('Time [s]') 
    ax2.set_title('LF note', loc='right')

    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', aspect=50, pad=0.02) 
    cbar.set_label('Cross-correlation envelope []')
    plt.show()

    return


def plot_cross_correlogram(corr_m, time, dist, maxv, minv=0, file_begin_time_utc=None):
    """
    Plot the cross-correlogram between HF and LF notes.

    Parameters
    ----------
    corr_m : numpy.ndarray
        The cross-correlation matrix
    time : numpy.ndarray
        The time values.
    dist : numpy.ndarray
        The distance values.
    maxv : float
        The maximum value for the colorbar.
    minv : int, optional
        The minimum value for the colorbar. Default is 0.
    file_begin_time_utc : datetime.datetime, optional
        The beginning time of the file in UTC. Default is None.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    im = ax.imshow(abs(sp.hilbert(corr_m, axis=1)), extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='turbo', origin='lower', aspect='auto', vmin=minv, vmax=maxv) 
    ax.set_xlabel('Time [s]') 
    ax.set_ylabel('Distance [km]') 
    ax.set_title('Cross-correlogram', loc='right')

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', aspect=50, pad=0.02) 
    cbar.set_label('Cross-correlation envelope []')
    plt.show()

    return


def import_roseus():
    """
    Import the colormap from the colormap/roseus_matplotlib.py file

    Returns
    -------
    ListedColormap
        colormap
    """
    from matplotlib.colors import ListedColormap
    # Roseus colormap data
    # https://github.com/dofuuz/roseus

    roseus_data = [
        [0.004528, 0.004341, 0.004307],
        [0.005625, 0.006156, 0.006010],
        [0.006628, 0.008293, 0.008161],
        [0.007551, 0.010738, 0.010790],
        [0.008382, 0.013482, 0.013941],
        [0.009111, 0.016520, 0.017662],
        [0.009727, 0.019846, 0.022009],
        [0.010223, 0.023452, 0.027035],
        [0.010593, 0.027331, 0.032799],
        [0.010833, 0.031475, 0.039361],
        [0.010941, 0.035875, 0.046415],
        [0.010918, 0.040520, 0.053597],
        [0.010768, 0.045158, 0.060914],
        [0.010492, 0.049708, 0.068367],
        [0.010098, 0.054171, 0.075954],
        [0.009594, 0.058549, 0.083672],
        [0.008989, 0.062840, 0.091521],
        [0.008297, 0.067046, 0.099499],
        [0.007530, 0.071165, 0.107603],
        [0.006704, 0.075196, 0.115830],
        [0.005838, 0.079140, 0.124178],
        [0.004949, 0.082994, 0.132643],
        [0.004062, 0.086758, 0.141223],
        [0.003198, 0.090430, 0.149913],
        [0.002382, 0.094010, 0.158711],
        [0.001643, 0.097494, 0.167612],
        [0.001009, 0.100883, 0.176612],
        [0.000514, 0.104174, 0.185704],
        [0.000187, 0.107366, 0.194886],
        [0.000066, 0.110457, 0.204151],
        [0.000186, 0.113445, 0.213496],
        [0.000587, 0.116329, 0.222914],
        [0.001309, 0.119106, 0.232397],
        [0.002394, 0.121776, 0.241942],
        [0.003886, 0.124336, 0.251542],
        [0.005831, 0.126784, 0.261189],
        [0.008276, 0.129120, 0.270876],
        [0.011268, 0.131342, 0.280598],
        [0.014859, 0.133447, 0.290345],
        [0.019100, 0.135435, 0.300111],
        [0.024043, 0.137305, 0.309888],
        [0.029742, 0.139054, 0.319669],
        [0.036252, 0.140683, 0.329441],
        [0.043507, 0.142189, 0.339203],
        [0.050922, 0.143571, 0.348942],
        [0.058432, 0.144831, 0.358649],
        [0.066041, 0.145965, 0.368319],
        [0.073744, 0.146974, 0.377938],
        [0.081541, 0.147858, 0.387501],
        [0.089431, 0.148616, 0.396998],
        [0.097411, 0.149248, 0.406419],
        [0.105479, 0.149754, 0.415755],
        [0.113634, 0.150134, 0.424998],
        [0.121873, 0.150389, 0.434139],
        [0.130192, 0.150521, 0.443167],
        [0.138591, 0.150528, 0.452075],
        [0.147065, 0.150413, 0.460852],
        [0.155614, 0.150175, 0.469493],
        [0.164232, 0.149818, 0.477985],
        [0.172917, 0.149343, 0.486322],
        [0.181666, 0.148751, 0.494494],
        [0.190476, 0.148046, 0.502493],
        [0.199344, 0.147229, 0.510313],
        [0.208267, 0.146302, 0.517944],
        [0.217242, 0.145267, 0.525380],
        [0.226264, 0.144131, 0.532613],
        [0.235331, 0.142894, 0.539635],
        [0.244440, 0.141559, 0.546442],
        [0.253587, 0.140131, 0.553026],
        [0.262769, 0.138615, 0.559381],
        [0.271981, 0.137016, 0.565500],
        [0.281222, 0.135335, 0.571381],
        [0.290487, 0.133581, 0.577017],
        [0.299774, 0.131757, 0.582404],
        [0.309080, 0.129867, 0.587538],
        [0.318399, 0.127920, 0.592415],
        [0.327730, 0.125921, 0.597032],
        [0.337069, 0.123877, 0.601385],
        [0.346413, 0.121793, 0.605474],
        [0.355758, 0.119678, 0.609295],
        [0.365102, 0.117540, 0.612846],
        [0.374443, 0.115386, 0.616127],
        [0.383774, 0.113226, 0.619138],
        [0.393096, 0.111066, 0.621876],
        [0.402404, 0.108918, 0.624343],
        [0.411694, 0.106794, 0.626540],
        [0.420967, 0.104698, 0.628466],
        [0.430217, 0.102645, 0.630123],
        [0.439442, 0.100647, 0.631513],
        [0.448637, 0.098717, 0.632638],
        [0.457805, 0.096861, 0.633499],
        [0.466940, 0.095095, 0.634100],
        [0.476040, 0.093433, 0.634443],
        [0.485102, 0.091885, 0.634532],
        [0.494125, 0.090466, 0.634370],
        [0.503104, 0.089190, 0.633962],
        [0.512041, 0.088067, 0.633311],
        [0.520931, 0.087108, 0.632420],
        [0.529773, 0.086329, 0.631297],
        [0.538564, 0.085738, 0.629944],
        [0.547302, 0.085346, 0.628367],
        [0.555986, 0.085162, 0.626572],
        [0.564615, 0.085190, 0.624563],
        [0.573187, 0.085439, 0.622345],
        [0.581698, 0.085913, 0.619926],
        [0.590149, 0.086615, 0.617311],
        [0.598538, 0.087543, 0.614503],
        [0.606862, 0.088700, 0.611511],
        [0.615120, 0.090084, 0.608343],
        [0.623312, 0.091690, 0.605001],
        [0.631438, 0.093511, 0.601489],
        [0.639492, 0.095546, 0.597821],
        [0.647476, 0.097787, 0.593999],
        [0.655389, 0.100226, 0.590028],
        [0.663230, 0.102856, 0.585914],
        [0.670995, 0.105669, 0.581667],
        [0.678686, 0.108658, 0.577291],
        [0.686302, 0.111813, 0.572790],
        [0.693840, 0.115129, 0.568175],
        [0.701300, 0.118597, 0.563449],
        [0.708682, 0.122209, 0.558616],
        [0.715984, 0.125959, 0.553687],
        [0.723206, 0.129840, 0.548666],
        [0.730346, 0.133846, 0.543558],
        [0.737406, 0.137970, 0.538366],
        [0.744382, 0.142209, 0.533101],
        [0.751274, 0.146556, 0.527767],
        [0.758082, 0.151008, 0.522369],
        [0.764805, 0.155559, 0.516912],
        [0.771443, 0.160206, 0.511402],
        [0.777995, 0.164946, 0.505845],
        [0.784459, 0.169774, 0.500246],
        [0.790836, 0.174689, 0.494607],
        [0.797125, 0.179688, 0.488935],
        [0.803325, 0.184767, 0.483238],
        [0.809435, 0.189925, 0.477518],
        [0.815455, 0.195160, 0.471781],
        [0.821384, 0.200471, 0.466028],
        [0.827222, 0.205854, 0.460267],
        [0.832968, 0.211308, 0.454505],
        [0.838621, 0.216834, 0.448738],
        [0.844181, 0.222428, 0.442979],
        [0.849647, 0.228090, 0.437230],
        [0.855019, 0.233819, 0.431491],
        [0.860295, 0.239613, 0.425771],
        [0.865475, 0.245471, 0.420074],
        [0.870558, 0.251393, 0.414403],
        [0.875545, 0.257380, 0.408759],
        [0.880433, 0.263427, 0.403152],
        [0.885223, 0.269535, 0.397585],
        [0.889913, 0.275705, 0.392058],
        [0.894503, 0.281934, 0.386578],
        [0.898993, 0.288222, 0.381152],
        [0.903381, 0.294569, 0.375781],
        [0.907667, 0.300974, 0.370469],
        [0.911849, 0.307435, 0.365223],
        [0.915928, 0.313953, 0.360048],
        [0.919902, 0.320527, 0.354948],
        [0.923771, 0.327155, 0.349928],
        [0.927533, 0.333838, 0.344994],
        [0.931188, 0.340576, 0.340149],
        [0.934736, 0.347366, 0.335403],
        [0.938175, 0.354207, 0.330762],
        [0.941504, 0.361101, 0.326229],
        [0.944723, 0.368045, 0.321814],
        [0.947831, 0.375039, 0.317523],
        [0.950826, 0.382083, 0.313364],
        [0.953709, 0.389175, 0.309345],
        [0.956478, 0.396314, 0.305477],
        [0.959133, 0.403499, 0.301766],
        [0.961671, 0.410731, 0.298221],
        [0.964093, 0.418008, 0.294853],
        [0.966399, 0.425327, 0.291676],
        [0.968586, 0.432690, 0.288696],
        [0.970654, 0.440095, 0.285926],
        [0.972603, 0.447540, 0.283380],
        [0.974431, 0.455025, 0.281067],
        [0.976139, 0.462547, 0.279003],
        [0.977725, 0.470107, 0.277198],
        [0.979188, 0.477703, 0.275666],
        [0.980529, 0.485332, 0.274422],
        [0.981747, 0.492995, 0.273476],
        [0.982840, 0.500690, 0.272842],
        [0.983808, 0.508415, 0.272532],
        [0.984653, 0.516168, 0.272560],
        [0.985373, 0.523948, 0.272937],
        [0.985966, 0.531754, 0.273673],
        [0.986436, 0.539582, 0.274779],
        [0.986780, 0.547434, 0.276264],
        [0.986998, 0.555305, 0.278135],
        [0.987091, 0.563195, 0.280401],
        [0.987061, 0.571100, 0.283066],
        [0.986907, 0.579019, 0.286137],
        [0.986629, 0.586950, 0.289615],
        [0.986229, 0.594891, 0.293503],
        [0.985709, 0.602839, 0.297802],
        [0.985069, 0.610792, 0.302512],
        [0.984310, 0.618748, 0.307632],
        [0.983435, 0.626704, 0.313159],
        [0.982445, 0.634657, 0.319089],
        [0.981341, 0.642606, 0.325420],
        [0.980130, 0.650546, 0.332144],
        [0.978812, 0.658475, 0.339257],
        [0.977392, 0.666391, 0.346753],
        [0.975870, 0.674290, 0.354625],
        [0.974252, 0.682170, 0.362865],
        [0.972545, 0.690026, 0.371466],
        [0.970750, 0.697856, 0.380419],
        [0.968873, 0.705658, 0.389718],
        [0.966921, 0.713426, 0.399353],
        [0.964901, 0.721157, 0.409313],
        [0.962815, 0.728851, 0.419594],
        [0.960677, 0.736500, 0.430181],
        [0.958490, 0.744103, 0.441070],
        [0.956263, 0.751656, 0.452248],
        [0.954009, 0.759153, 0.463702],
        [0.951732, 0.766595, 0.475429],
        [0.949445, 0.773974, 0.487414],
        [0.947158, 0.781289, 0.499647],
        [0.944885, 0.788535, 0.512116],
        [0.942634, 0.795709, 0.524811],
        [0.940423, 0.802807, 0.537717],
        [0.938261, 0.809825, 0.550825],
        [0.936163, 0.816760, 0.564121],
        [0.934146, 0.823608, 0.577591],
        [0.932224, 0.830366, 0.591220],
        [0.930412, 0.837031, 0.604997],
        [0.928727, 0.843599, 0.618904],
        [0.927187, 0.850066, 0.632926],
        [0.925809, 0.856432, 0.647047],
        [0.924610, 0.862691, 0.661249],
        [0.923607, 0.868843, 0.675517],
        [0.922820, 0.874884, 0.689832],
        [0.922265, 0.880812, 0.704174],
        [0.921962, 0.886626, 0.718523],
        [0.921930, 0.892323, 0.732859],
        [0.922183, 0.897903, 0.747163],
        [0.922741, 0.903364, 0.761410],
        [0.923620, 0.908706, 0.775580],
        [0.924837, 0.913928, 0.789648],
        [0.926405, 0.919031, 0.803590],
        [0.928340, 0.924015, 0.817381],
        [0.930655, 0.928881, 0.830995],
        [0.933360, 0.933631, 0.844405],
        [0.936466, 0.938267, 0.857583],
        [0.939982, 0.942791, 0.870499],
        [0.943914, 0.947207, 0.883122],
        [0.948267, 0.951519, 0.895421],
        [0.953044, 0.955732, 0.907359],
        [0.958246, 0.959852, 0.918901],
        [0.963869, 0.963887, 0.930004],
        [0.969909, 0.967845, 0.940623],
        [0.976355, 0.971737, 0.950704],
        [0.983195, 0.975580, 0.960181],
        [0.990402, 0.979395, 0.968966],
        [0.997930, 0.983217, 0.976920],]
    return ListedColormap(roseus_data, name='Roseus')


def import_parula():
    """
    Import the colormap parula from matlab

    Returns
    -------
    ListedColormap
        colormap
    """
    from matplotlib.colors import ListedColormap
    # Parula colormap data
    parula_data = cm_data = [
        [0.2422, 0.1504, 0.6603],
        [0.2444, 0.1534, 0.6728],
        [0.2464, 0.1569, 0.6847],
        [0.2484, 0.1607, 0.6961],
        [0.2503, 0.1648, 0.7071],
        [0.2522, 0.1689, 0.7179],
        [0.254, 0.1732, 0.7286],
        [0.2558, 0.1773, 0.7393],
        [0.2576, 0.1814, 0.7501],
        [0.2594, 0.1854, 0.761],
        [0.2611, 0.1893, 0.7719],
        [0.2628, 0.1932, 0.7828],
        [0.2645, 0.1972, 0.7937],
        [0.2661, 0.2011, 0.8043],
        [0.2676, 0.2052, 0.8148],
        [0.2691, 0.2094, 0.8249],
        [0.2704, 0.2138, 0.8346],
        [0.2717, 0.2184, 0.8439],
        [0.2729, 0.2231, 0.8528],
        [0.274, 0.228, 0.8612],
        [0.2749, 0.233, 0.8692],
        [0.2758, 0.2382, 0.8767],
        [0.2766, 0.2435, 0.884],
        [0.2774, 0.2489, 0.8908],
        [0.2781, 0.2543, 0.8973],
        [0.2788, 0.2598, 0.9035],
        [0.2794, 0.2653, 0.9094],
        [0.2798, 0.2708, 0.915],
        [0.2802, 0.2764, 0.9204],
        [0.2806, 0.2819, 0.9255],
        [0.2809, 0.2875, 0.9305],
        [0.2811, 0.293, 0.9352],
        [0.2813, 0.2985, 0.9397],
        [0.2814, 0.304, 0.9441],
        [0.2814, 0.3095, 0.9483],
        [0.2813, 0.315, 0.9524],
        [0.2811, 0.3204, 0.9563],
        [0.2809, 0.3259, 0.96],
        [0.2807, 0.3313, 0.9636],
        [0.2803, 0.3367, 0.967],
        [0.2798, 0.3421, 0.9702],
        [0.2791, 0.3475, 0.9733],
        [0.2784, 0.3529, 0.9763],
        [0.2776, 0.3583, 0.9791],
        [0.2766, 0.3638, 0.9817],
        [0.2754, 0.3693, 0.984],
        [0.2741, 0.3748, 0.9862],
        [0.2726, 0.3804, 0.9881],
        [0.271, 0.386, 0.9898],
        [0.2691, 0.3916, 0.9912],
        [0.267, 0.3973, 0.9924],
        [0.2647, 0.403, 0.9935],
        [0.2621, 0.4088, 0.9946],
        [0.2591, 0.4145, 0.9955],
        [0.2556, 0.4203, 0.9965],
        [0.2517, 0.4261, 0.9974],
        [0.2473, 0.4319, 0.9983],
        [0.2424, 0.4378, 0.9991],
        [0.2369, 0.4437, 0.9996],
        [0.2311, 0.4497, 0.9995],
        [0.225, 0.4559, 0.9985],
        [0.2189, 0.462, 0.9968],
        [0.2128, 0.4682, 0.9948],
        [0.2066, 0.4743, 0.9926],
        [0.2006, 0.4803, 0.9906],
        [0.195, 0.4861, 0.9887],
        [0.1903, 0.4919, 0.9867],
        [0.1869, 0.4975, 0.9844],
        [0.1847, 0.503, 0.9819],
        [0.1831, 0.5084, 0.9793],
        [0.1818, 0.5138, 0.9766],
        [0.1806, 0.5191, 0.9738],
        [0.1795, 0.5244, 0.9709],
        [0.1785, 0.5296, 0.9677],
        [0.1778, 0.5349, 0.9641],
        [0.1773, 0.5401, 0.9602],
        [0.1768, 0.5452, 0.956],
        [0.1764, 0.5504, 0.9516],
        [0.1755, 0.5554, 0.9473],
        [0.174, 0.5605, 0.9432],
        [0.1716, 0.5655, 0.9393],
        [0.1686, 0.5705, 0.9357],
        [0.1649, 0.5755, 0.9323],
        [0.161, 0.5805, 0.9289],
        [0.1573, 0.5854, 0.9254],
        [0.154, 0.5902, 0.9218],
        [0.1513, 0.595, 0.9182],
        [0.1492, 0.5997, 0.9147],
        [0.1475, 0.6043, 0.9113],
        [0.1461, 0.6089, 0.908],
        [0.1446, 0.6135, 0.905],
        [0.1429, 0.618, 0.9022],
        [0.1408, 0.6226, 0.8998],
        [0.1383, 0.6272, 0.8975],
        [0.1354, 0.6317, 0.8953],
        [0.1321, 0.6363, 0.8932],
        [0.1288, 0.6408, 0.891],
        [0.1253, 0.6453, 0.8887],
        [0.1219, 0.6497, 0.8862],
        [0.1185, 0.6541, 0.8834],
        [0.1152, 0.6584, 0.8804],
        [0.1119, 0.6627, 0.877],
        [0.1085, 0.6669, 0.8734],
        [0.1048, 0.671, 0.8695],
        [0.1009, 0.675, 0.8653],
        [0.0964, 0.6789, 0.8609],
        [0.0914, 0.6828, 0.8562],
        [0.0855, 0.6865, 0.8513],
        [0.0789, 0.6902, 0.8462],
        [0.0713, 0.6938, 0.8409],
        [0.0628, 0.6972, 0.8355],
        [0.0535, 0.7006, 0.8299],
        [0.0433, 0.7039, 0.8242],
        [0.0328, 0.7071, 0.8183],
        [0.0234, 0.7103, 0.8124],
        [0.0155, 0.7133, 0.8064],
        [0.0091, 0.7163, 0.8003],
        [0.0046, 0.7192, 0.7941],
        [0.0019, 0.722, 0.7878],
        [0.0009, 0.7248, 0.7815],
        [0.0018, 0.7275, 0.7752],
        [0.0046, 0.7301, 0.7688],
        [0.0094, 0.7327, 0.7623],
        [0.0162, 0.7352, 0.7558],
        [0.0253, 0.7376, 0.7492],
        [0.0369, 0.74, 0.7426],
        [0.0504, 0.7423, 0.7359],
        [0.0638, 0.7446, 0.7292],
        [0.077, 0.7468, 0.7224],
        [0.0899, 0.7489, 0.7156],
        [0.1023, 0.751, 0.7088],
        [0.1141, 0.7531, 0.7019],
        [0.1252, 0.7552, 0.695],
        [0.1354, 0.7572, 0.6881],
        [0.1448, 0.7593, 0.6812],
        [0.1532, 0.7614, 0.6741],
        [0.1609, 0.7635, 0.6671],
        [0.1678, 0.7656, 0.6599],
        [0.1741, 0.7678, 0.6527],
        [0.1799, 0.7699, 0.6454],
        [0.1853, 0.7721, 0.6379],
        [0.1905, 0.7743, 0.6303],
        [0.1954, 0.7765, 0.6225],
        [0.2003, 0.7787, 0.6146],
        [0.2061, 0.7808, 0.6065],
        [0.2118, 0.7828, 0.5983],
        [0.2178, 0.7849, 0.5899],
        [0.2244, 0.7869, 0.5813],
        [0.2318, 0.7887, 0.5725],
        [0.2401, 0.7905, 0.5636],
        [0.2491, 0.7922, 0.5546],
        [0.2589, 0.7937, 0.5454],
        [0.2695, 0.7951, 0.536],
        [0.2809, 0.7964, 0.5266],
        [0.2929, 0.7975, 0.517],
        [0.3052, 0.7985, 0.5074],
        [0.3176, 0.7994, 0.4975],
        [0.3301, 0.8002, 0.4876],
        [0.3424, 0.8009, 0.4774],
        [0.3548, 0.8016, 0.4669],
        [0.3671, 0.8021, 0.4563],
        [0.3795, 0.8026, 0.4454],
        [0.3921, 0.8029, 0.4344],
        [0.405, 0.8031, 0.4233],
        [0.4184, 0.803, 0.4122],
        [0.4322, 0.8028, 0.4013],
        [0.4463, 0.8024, 0.3904],
        [0.4608, 0.8018, 0.3797],
        [0.4753, 0.8011, 0.3691],
        [0.4899, 0.8002, 0.3586],
        [0.5044, 0.7993, 0.348],
        [0.5187, 0.7982, 0.3374],
        [0.5329, 0.797, 0.3267],
        [0.547, 0.7957, 0.3159],
        [0.5609, 0.7943, 0.305],
        [0.5748, 0.7929, 0.2941],
        [0.5886, 0.7913, 0.2833],
        [0.6024, 0.7896, 0.2726],
        [0.6161, 0.7878, 0.2622],
        [0.6297, 0.7859, 0.2521],
        [0.6433, 0.7839, 0.2423],
        [0.6567, 0.7818, 0.2329],
        [0.6701, 0.7796, 0.2239],
        [0.6833, 0.7773, 0.2155],
        [0.6963, 0.775, 0.2075],
        [0.7091, 0.7727, 0.1998],
        [0.7218, 0.7703, 0.1924],
        [0.7344, 0.7679, 0.1852],
        [0.7468, 0.7654, 0.1782],
        [0.759, 0.7629, 0.1717],
        [0.771, 0.7604, 0.1658],
        [0.7829, 0.7579, 0.1608],
        [0.7945, 0.7554, 0.157],
        [0.806, 0.7529, 0.1546],
        [0.8172, 0.7505, 0.1535],
        [0.8281, 0.7481, 0.1536],
        [0.8389, 0.7457, 0.1546],
        [0.8495, 0.7435, 0.1564],
        [0.86, 0.7413, 0.1587],
        [0.8703, 0.7392, 0.1615],
        [0.8804, 0.7372, 0.165],
        [0.8903, 0.7353, 0.1695],
        [0.9, 0.7336, 0.1749],
        [0.9093, 0.7321, 0.1815],
        [0.9184, 0.7308, 0.189],
        [0.9272, 0.7298, 0.1973],
        [0.9357, 0.729, 0.2061],
        [0.944, 0.7285, 0.2151],
        [0.9523, 0.7284, 0.2237],
        [0.9606, 0.7285, 0.2312],
        [0.9689, 0.7292, 0.2373],
        [0.977, 0.7304, 0.2418],
        [0.9842, 0.733, 0.2446],
        [0.99, 0.7365, 0.2429],
        [0.9946, 0.7407, 0.2394],
        [0.9966, 0.7458, 0.2351],
        [0.9971, 0.7513, 0.2309],
        [0.9972, 0.7569, 0.2267],
        [0.9971, 0.7626, 0.2224],
        [0.9969, 0.7683, 0.2181],
        [0.9966, 0.774, 0.2138],
        [0.9962, 0.7798, 0.2095],
        [0.9957, 0.7856, 0.2053],
        [0.9949, 0.7915, 0.2012],
        [0.9938, 0.7974, 0.1974],
        [0.9923, 0.8034, 0.1939],
        [0.9906, 0.8095, 0.1906],
        [0.9885, 0.8156, 0.1875],
        [0.9861, 0.8218, 0.1846],
        [0.9835, 0.828, 0.1817],
        [0.9807, 0.8342, 0.1787],
        [0.9778, 0.8404, 0.1757],
        [0.9748, 0.8467, 0.1726],
        [0.972, 0.8529, 0.1695],
        [0.9694, 0.8591, 0.1665],
        [0.9671, 0.8654, 0.1636],
        [0.9651, 0.8716, 0.1608],
        [0.9634, 0.8778, 0.1582],
        [0.9619, 0.884, 0.1557],
        [0.9608, 0.8902, 0.1532],
        [0.9601, 0.8963, 0.1507],
        [0.9596, 0.9023, 0.148],
        [0.9595, 0.9084, 0.145],
        [0.9597, 0.9143, 0.1418],
        [0.9601, 0.9203, 0.1382],
        [0.9608, 0.9262, 0.1344],
        [0.9618, 0.932, 0.1304],
        [0.9629, 0.9379, 0.1261],
        [0.9642, 0.9437, 0.1216],
        [0.9657, 0.9494, 0.1168],
        [0.9674, 0.9552, 0.1116],
        [0.9692, 0.9609, 0.1061],
        [0.9711, 0.9667, 0.1001],
        [0.973, 0.9724, 0.0938],
        [0.9749, 0.9782, 0.0872],
        [0.9769, 0.9839, 0.0805]]
    return ListedColormap(parula_data, name='Parula')