"""
data_handle.py - data handling functions for DAS data

This module provides various functions to handle DAS data, including loading, downloading, and conditioning. 
It aims at having specific functions for each interrogator type.

Authors: Léa Bouffaut, Quentin Goestchel, Erfan Horeh
Date: 2023-2024
"""

import h5py
import wget
import os
import numpy as np
import dask.array as da
from datetime import datetime
from nptdms import TdmsFile


def hello_world_das_package():
    print("Yepee! You now have access to all the functionalities of the das4whale python package!")


# Definition of the functions for DAS data conditioning
def get_acquisition_parameters(filepath, interrogator='optasense'):
    """
    Retrieve acquisition parameters based on the specified interrogator.

    Parameters
    ----------
    filepath : str
        The file path to the data file.
    interrogator : str, optional
        The interrogator type, one of {'optasense', 'silixa', 'mars', 'alcatel'}.
        Defaults to 'optasense'.

    Returns
    -------
    metadata : dict or None
        Metadata related to the acquisition parameters. Returns None if no matching
        interrogator is found.

    Raises
    ------
    ValueError
        If the interrogator name is not in the predefined list.
    """
    # List the known used interrogators:
    interrogator_list = ['optasense', 'silixa', 'mars', 'alcatel']
    if interrogator in interrogator_list:

        if interrogator == 'optasense':
            metadata = get_metadata_optasense(filepath)

        elif interrogator == 'silixa':
            metadata = get_metadata_silixa(filepath)

        elif interrogator == 'mars':
            metadata = get_metadata_mars(filepath)

        elif interrogator == 'alcatel':
            metadata = get_metadata_alcatel(filepath)

    else:
        raise ValueError('Interrogator name incorrect')

    return metadata


def get_metadata_optasense(filepath):
    """Gets DAS acquisition parameters for the optasense interrogator 

    Parameters
    ----------
    filepath : string
        a string containing the full path to the data to load

    Returns
    -------
    metadata : dict
        dictionary filled with metadata, key's breakdown:\n
        fs: the sampling frequency (Hz)\n
        dx: interval between two virtual sensing points also called channel spacing (m)\n
        nx: the number of spatial samples also called channels\n
        ns: the number of time samples\n
        n: refractive index of the fiber\n
        GL: the gauge length (m)\n
        scale_factor: the value to convert DAS data from strain rate to strain

    """
    # Make sure the file exists
    if os.path.exists(filepath):
        # Ensure the closure of the file after reading
        with h5py.File(filepath, 'r') as fp1:
            fp1 = h5py.File(filepath, 'r')

            fs = fp1['Acquisition']['Raw[0]'].attrs['OutputDataRate'] # sampling rate in Hz
            dx = fp1['Acquisition'].attrs['SpatialSamplingInterval'] # channel spacing in m
            ns = fp1['Acquisition']['Raw[0]']['RawDataTime'].attrs['Count']
            n = fp1['Acquisition']['Custom'].attrs['Fibre Refractive Index'] # refractive index
            GL = fp1['Acquisition'].attrs['GaugeLength'] # gauge length in m
            nx = fp1['Acquisition']['Raw[0]'].attrs['NumberOfLoci'] # number of channels
            scale_factor = (2 * np.pi) / 2 ** 16 * (1550.12 * 1e-9) / (0.78 * 4 * np.pi * n * GL)

        meta_data = {'fs': fs, 'dx': dx, 'ns': ns,'n': n,'GL': GL, 'nx':nx , 'scale_factor': scale_factor}
    else:
        raise FileNotFoundError(f'File {filepath} not found')

    return meta_data


def get_metadata_silixa(filepath):
    """Gets DAS acquisition parameters for the silixa interrogator 

    Parameters
    ----------
    filepath : string
        a string containing the full path to the data to load

    Returns
    -------
    metadata : dict
        dictionary filled with metadata, key's breakdown:\n
        fs: the sampling frequency (Hz)\n
        dx: interval between two virtual sensing points also called channel spacing (m)\n
        nx: the number of spatial samples also called channels\n
        ns: the number of time samples\n
        n: refractive index of the fiber\n
        GL: the gauge length (m)\n
        scale_factor: the value to convert DAS data from strain rate to strain

    """

    # Make sure the file exists
    if os.path.exists(filepath):
        fp = TdmsFile.read(filepath)
        props = fp.properties
        group = fp['Measurement']
        acousticData = np.asarray( [group[channel].data for channel in group] )

        fs = props['SamplingFrequency[Hz]'] # sampling rate in Hz
        dx = props['SpatialResolution[m]'] # channel spacing in m
        ns = acousticData.shape[1]
        n =  props['FibreIndex'] # refractive index
        GL =  props['GaugeLength'] # gauge length in m
        nx = acousticData.shape[0] # number of channels
        scale_factor = (116 * fs * 10**-9) / (GL * 2**13)

        meta_data = {'fs': fs, 'dx': dx, 'ns': ns,'n': n,'GL': GL, 'nx':nx , 'scale_factor': scale_factor}
    else:
        raise FileNotFoundError(f'File {filepath} not found')
    
    return meta_data


def raw2strain(trace, metadata):
    """Transform the amplitude of raw das data from strain-rate to strain according to scale factor


    Parameters
    ----------
    trace : array-like
        a [channel x time sample] nparray containing the raw data in the spatio-temporal domain
    metadata : dict
        dictionary filled with metadata (fs, dx, nx, ns, n, GL, scale_factor)

    Returns
    -------
    trace : array-like
        a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    """    
    # Remove the mean trend from each channel and scale

    trace -= np.mean(trace, axis=1, keepdims=True)
    trace *= metadata["scale_factor"] 
    return trace


def load_das_data(filename, selected_channels, metadata):
    """
    Load the DAS data corresponding to the input file name as strain according to the selected channels.

    Parameters
    ----------
    filename : str
        A string containing the full path to the data to load.
    selected_channels : list
        A list containing the selected channels.
    metadata : dict
        A dictionary filled with metadata (sampling frequency, channel spacing, scale factor...).

    Returns
    -------
    trace : np.ndarray
        A [channel x sample] nparray containing the strain data.
    tx : np.ndarray
        The corresponding time axis (s).
    dist : np.ndarray
        The corresponding distance along the FO cable axis (m).
    file_begin_time_utc : datetime.datetime
        The beginning time of the file, can be printed using file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S").
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File {filename} not found')

    with h5py.File(filename, 'r') as fp:
        # Data matrix
        raw_data = fp['Acquisition/Raw[0]/RawData']

        # Selection the traces corresponding to the desired channels
        # Loaded as float64, float 32 might be sufficient? 
        trace = raw_data[selected_channels[0]:selected_channels[1]:selected_channels[2], :].astype(np.float64)
        trace = raw2strain(trace, metadata)

        # UTC Time vector for naming
        raw_data_time = fp['Acquisition']['Raw[0]']['RawDataTime']

        # For future save
        file_begin_time_utc = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)

        # Store the following as the dimensions of our data block
        nnx = trace.shape[0]
        nns = trace.shape[1]

        # Define new time and distance axes
        tx = np.arange(nns) / metadata["fs"]
        dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata["dx"]        

    return trace, tx, dist, file_begin_time_utc


def dl_file(url):
    """Download the file at the given url

    Parameters
    ----------
    url : string
        url location of the file

    Returns
    -------
    filepath : string
        local path destination of the file
    """    
    filename = url.split('/')[-1]
    filepath = os.path.join('data',filename)
    if os.path.exists(filepath) == True:
        print(f'{filename} already stored locally')
    else:
        # Create the data subfolder if it doesn't exist
        os.makedirs('data', exist_ok=True)
        wget.download(url, out='data', bar=wget.bar_adaptive)
        print(f'Downloaded {filename}')
    return filepath #TODO: add filenames as output to create large daskarrays

