"""
dask_wrap.py - Dask wrapper functions for DAS data processing

This module provides functions to wrap up the functions of das4whales in a dask way.

Authors: Léa Bouffaut, Quentin Goestchel
Date: 2024
"""

# File that wrap up functions in dsp.py in a dask way
import h5py
import wget
import os
import numpy as np
import dask.array as da
from datetime import datetime

import das4whales as dw


def load_das_data(filename, selected_channels, metadata):
    """
    Load the DAS data corresponding to the input file name as strain according to the selected channels.

    Parameters
    ----------
    filename : str
        The full path to the data to load.
    selected_channels : list
        A list containing the start, stop, and step values for selecting channels.
    metadata : dict
        A dictionary filled with metadata (sampling frequency, channel spacing, scale factor, etc.).

    Returns
    -------
    np.ndarray
        A [channel x sample] numpy array containing the strain data.
    np.ndarray
        The corresponding time axis (s).
    np.ndarray
        The corresponding distance along the FO cable axis (m).
    datetime.datetime
        The beginning time of the file.

    Raises
    ------
    ValueError
        If the file is not found.

    """
    if not os.path.exists(filename):
        raise ValueError('File not found')

    f = h5py.File(filename, 'r') # HDF5 file
    d = f['Acquisition/Raw[0]/RawData']   # Pointer on on-disk array f

    # UTC Time vector for naming
    raw_data_time = f['Acquisition/Raw[0]/RawDataTime']

    # For future save
    file_begin_time_utc = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)

    # Store the following as the dimensions of our data block
    nnx = d.shape[0]
    nns = d.shape[1]

    # Define new time and distance axes
    tx = np.arange(nns) / metadata["fs"]
    dist = (np.arange(nnx)[selected_channels[0]:selected_channels[1]:selected_channels[2]]) * metadata["dx"] 
    return d, tx, dist, file_begin_time_utc


def raw2strain(tr, metadata, selected_channels):
    """Convert a daskarray filled of int32 to a 

    Parameters
    ----------
    tr : dask.array.core.Array
        daskarray built on the HDF5 pointer
    metadata : dict
        dictionary of metadata
    selected_channels : list
        list of selected spatial indexes and spatial step

    Returns
    -------
    dask.array.core.Array
        daskarray filled with scaled float64
    """    
    trace = tr[selected_channels[0]:selected_channels[1]:selected_channels[2], :].astype(np.float64)
    trace -= da.mean(trace, axis=1, keepdims=True) #demeaning using dask mean function
    trace *= metadata["scale_factor"]
    return trace