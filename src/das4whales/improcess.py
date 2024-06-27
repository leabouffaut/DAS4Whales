"""
improcess.py - image processing functions for DAS data

This module contains functions for image processing of DAS data.

Authors: Quentin Goestchel
Date: 2023-2024

"""

import numpy as np
import cv2
import scipy.signal as sp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.transform import radon, iradon
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from tqdm import tqdm


def scale_pixels(img):
    """Scale the pixel values of an image.

    This function scales the pixel values of an image to the range [0, 1].

    Parameters
    ----------
    img : numpy.ndarray
        The input image.

    Returns
    -------
    numpy.ndarray
        The scaled image.

    """

    img = (img - img.min())/(img.max() - img.min())
    return img


def trace2image(trace):
    """Convert a DAS trace to an image.

    This function converts a DAS trace matrix to an image.

    Parameters
    ----------
    trace : numpy.ndarray
        The input trace.

    Returns
    -------
    numpy.ndarray
        The image.

    """

    image = np.abs(sp.hilbert(trace, axis=1)) / np.std(trace, axis=1, keepdims=True)
    image = scale_pixels(image) * 255
    return image


def angle_fromspeed(c0, fs, dx, selected_channels):
    """Calculate the angle from the speed of sound.

    This function calculates the angle from the speed of sound.

    Parameters
    ----------
    c0 : float
        The speed of sound.
    fs : float
        The sampling frequency.
    dx : float
        The spatial resolution.
    selected_channels : list
        list of the selected channels number  [start, end, step].

    Returns
    -------
    float
        The angle from the speed of sound.

    """

    ratio = c0 / (fs * dx * selected_channels[2])
    print('Detection speed ratio: ', ratio)

    # angle between sound speed lines and the horizontal
    theta_c0 = np.arctan(ratio) * 180 / np.pi
    print('Angle: ', theta_c0)
    return theta_c0


def gabor_filt_design(theta_c0, plot=False):    
    """Design Gabor filters for lines that are oriented along sound speed.

    This function designs Gabor filters for lines that are oriented along sound speed.

    Parameters
    ----------
    theta_c0 : float
        The angle from the speed of sound.

    Returns
    -------
    numpy.ndarray
        The Gabor filters.

    """

    # Define parameters for the Gabor filter
    ksize = 100  # Kernel size 
    sigma = 4 # Standard deviation of the Gaussian envelope
    theta = np.pi/2 + np.deg2rad(theta_c0) # Orientation angle (in radians)
    lambd = 20 #theta_c0 * np.pi / 180  # Wavelength of the sinusoidal factor
    gamma = 0.15 # Spatial aspect ratio (controls the ellipticity of the filter)

    # Create the Gabor filter
    gabor_filtup = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_64F)
    gabor_filtdown = np.flipud(gabor_filtup)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.subplot(121)
        plt.imshow(gabor_filtup, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        plt.xlabel('Time indices')
        plt.ylabel('Distance indices')
        plt.colorbar(orientation='horizontal')

        plt.subplot(122)
        plt.imshow(gabor_filtdown, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        plt.xlabel('Time indices')
        plt.colorbar(orientation='horizontal')
        plt.tight_layout()
        plt.show()
    return gabor_filtup, gabor_filtdown


def gradient_oriented(image, direction):
    """Calculate the gradient oriented value of an image.

    This function calculates the gradient oriented value of an image based on the given direction.

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    direction : tuple
        A tuple containing the direction values (dft, dfx).

    Returns
    -------
    numpy.ndarray
        The gradient oriented value of the image.

    """
    dft, dfx = direction
    if dfx == 0:
        grad = -(image[:, :-dft] - image[:, dft:])
    elif dft == 0:
        grad = -(image[dfx:, :] - image[:-dfx, :])
    else:
        grad = -(image[dfx:-dfx, :-dft] - 0.5 * image[2 * dfx:, dft:] - 0.5 * image[:-2*dfx, dft:])

    return grad


def detect_diagonal_edges(matrix, threshold):
    # Calculate derivatives along both axes
    # dx , dy = np.gradient(matrix)
    
    # Construct a diagonal filter kernel
    # Sobel adaptation
    # diagonal_filter = np.array([[ 0,  1,  2],
    #                             [-1,  0,  1],
    #                             [-2, -1,  0]])
    
    # Prewitt adaptation
    # diagonal_filter = np.array([[ 0,  1,  1],
    #                             [-1,  0,  1],
    #                             [-1, -1,  0]])
    
    #Second order
    # diagonal_filter = np.array([[-1, -1, 2], 
    #                             [-1, 2, -1],
    #                             [2, -1, -1]])
    
    diagonal_filter = np.array([[ 0,  1,  1,  1,  1],
                                [-1,  0,  1,  1,  1],
                                [-1, -1, 0,   1,  1],
                                [-1, -1, -1,  0,  1],
                                [-1, -1, -1, -1,  0]])
    
    # diagonal_filter = np.array([[ 0,  1,  1,  1,  1,  1,  1],
    #                             [-1,  0,  1,  1,  1,  1,  1],
    #                             [-1, -1,  0,  1,  1,  1,  1],
    #                             [-1, -1, -1,  0,  1,  1,  1],
    #                             [-1, -1, -1, -1,  0,  1,  1],
    #                             [-1, -1, -1, -1, -1,  0,  1],
    #                             [-1, -1, -1, -1, -1, -1,  0]])
    
    # diagonal_filter = np.array([[ 0,  1,  1,  1,  1,  1,  1,  1],
    #                             [-1,  0,  1,  1,  1,  1,  1,  1],
    #                             [-1, -1,  0,  1,  1,  1,  1,  1],
    #                             [-1, -1, -1,  0,  1,  1,  1,  1],
    #                             [-1, -1, -1, -1,  0,  1,  1,  1],
    #                             [-1, -1, -1, -1, -1,  0,  1,  1],
    #                             [-1, -1, -1, -1, -1, -1,  0,  1],
    #                             [-1, -1, -1, -1, -1, -1, -1,  0]])


    diagonal_filterleft = np.fliplr(diagonal_filter)

    # Convolve the gradient with the diagonal filter
    diagonal_gradient = sp.fftconvolve(matrix, diagonal_filter, mode='same') + sp.fftconvolve(matrix, diagonal_filterleft, mode='same')

    # diagonal_gradient /= np.sum(np.abs(diagonal_gradient))

    # Apply threshold to identify diagonal edges
    # diagonal_edges = diagonal_gradient > threshold

    return diagonal_gradient


def diagonal_edge_detection(img, threshold):
    """Detect diagonal edges in an image. 
    Inspired from https://github.com/Ocean-Data-Lab/das-finwhale-vocalization/

    This function detects diagonal edges in an image.

    Parameters

    ----------
    img : numpy.ndarray
        The input image.
    threshold : float
        The threshold value.

    Returns
    -------
    numpy.ndarray
        The diagonal edges in the image.

    """

    img = torch.tensor(img, dtype=torch.float32)

    weight_left = torch.tensor([[2, -1, -1], 
                                [-1, 2, -1],
                                [-1, -1, 2]], dtype=torch.float32)
    
    weight_right = torch.flip(weight_left, [0])
    
    conv_left = F.conv2d(img.unsqueeze(0), weight_left.unsqueeze(0).unsqueeze(0), padding=1)
    conv_right = F.conv2d(img.unsqueeze(0), weight_right.unsqueeze(0).unsqueeze(0), padding=1)
    
    combined = conv_left + conv_right
    
    sigmoid_output = torch.sigmoid(combined)
    
    thresholded_output = (combined > threshold).float()
    
    return combined.squeeze(0)


def detect_long_lines(img):
    # Convert the image to grayscale (if it's not already)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img.copy() #* 255
    gray = np.uint8(gray)
    imglines = img.copy()

    plt.figure
    plt.imshow(gray, cmap='gray', origin='lower')
    plt.show()

    # Apply Gaussian blur to the image
    # blurred = cv2.GaussianBlur(gray, (9, 9), 1)

    # Apply a bilateral filter to the image
    blurred = cv2.bilateralFilter(gray,5,30,30)

    plt.figure()
    plt.imshow(blurred, cmap='gray', origin='lower')
    plt.show()

    # Use Canny edge detection on the blurred image
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=False)

    plt.figure()
    plt.imshow(edges, cmap='gray', origin='lower')
    plt.show()

    # keep only edges with a certain orientation (45 degrees)

    # Use Hough line transform to detect lines
    lines = cv2.HoughLinesP(
            edges, # Input edge image
            10, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=140, # Min number of votes for valid line
            minLineLength=10, # Min allowed length of line
            maxLineGap=100 # Max allowed gap between line for joining them
            )

    # Draw the lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw only the 45 degree lines

        cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return imglines


def bilateral_filter(img, diameter, sigma_color, sigma_space):
    """Apply bilateral filter to an image.

    This function applies bilateral filter to an image.

    Parameters
    ----------
    img : numpy.ndarray
        The input image.
    diameter : int
        Diameter of each pixel neighborhood that is used during filtering.
    sigma_color : float
        Filter sigma in the color space.
    sigma_space : float
        Filter sigma in the coordinate space.

    Returns
    -------
    numpy.ndarray
        The filtered image.
    
    """

    # Apply bilateral filter to the image
    filtered_img = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    return filtered_img


def compute_radon_transform(image, theta=None):
    """Compute the Radon transform of an image.

    This function computes the Radon transform of an image.

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    theta : numpy.ndarray, optional (default=None)   
        The projection angles (in degrees).

    Returns
    -------
    numpy.ndarray
        The Radon transform of the image.

    """
    # Compute the Radon transform
    radon_image = radon(image, theta=theta, circle=False)
    return radon_image


def gaussian_filter(img, size, sigma):
    """Apply Gaussian filter to an image.

    This function applies Gaussian filter to an image.

    Parameters
    ----------
    img : numpy.ndarray
        The input image.
    size : int
        The size of the filter.
    sigma : float
        The standard deviation of the filter.

    Returns
    -------
    numpy.ndarray
        The filtered image.

    """

    filtered_img = cv2.GaussianBlur(img, (size, size), sigma)
    return filtered_img


def binning(image, ft, fx):
    """Apply binning to an image.

    This function applies binning to an image.

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    ft : float
        The binning factor along the time axis.
    fx : float
        The binning factor along the spatial axis.
    
    Returns
    -------
    numpy.ndarray
        The binned image.
    """

    # img = cv2.resize(image, (0, 0), fx=ft, fy=fx, interpolation=cv2.INTER_AREA)


    imagebin = transforms.ToTensor()(image)    
    imagebin = transforms.Resize((int(image.shape[0] * fx) , int(image.shape[1] * ft)))(imagebin)
    img = imagebin.numpy()[0]
    return img


def apply_smooth_mask(array, mask, sigma=1.5):
    """Apply a smooth mask to an array.

    This function applies a smooth mask to an array.

    Parameters
    ----------
    array : numpy.ndarray
        The input array.
    mask : numpy.ndarray
        The mask to apply.
    sigma : float, optional (default=1.5)
        The standard deviation of the Gaussian filter applied to the mask.

    Returns
    -------
    numpy.ndarray
        The masked array.

    """

    # Apply Gaussian blur to the mask to smooth edges
    smoothed_mask = scipy_gaussian_filter(mask.astype(float), sigma=sigma, mode='reflect')
    
    # Normalize smoothed mask to range [0, 1]
    smoothed_mask = (smoothed_mask - np.min(smoothed_mask)) / (np.max(smoothed_mask) - np.min(smoothed_mask))

    # Element-wise multiplication of array with the smoothed mask
    masked_array = array * mask
    
    return masked_array


