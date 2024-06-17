# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:45:57 2024

@author: jingyan
"""

from matplotlib import pyplot as plt
from skimage import data, exposure
import numpy as np
from scipy import stats
import cv2

image_path = "D:\\jingyan\\MA\\final output directory\\grabcut2.png"
image = cv2.imread(image_path)

# Ensure the image is successfully read
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Initialize feature array to store mean, standard deviation, and skewness for each RGB channel
features = np.zeros(shape=(3, 3))

for k in range(image.shape[2]):
    channel_data = image[:, :, k].astype(np.float64)
    
    # Check and handle potential nan values
    if np.isnan(channel_data).any():
        channel_data = np.nan_to_num(channel_data, nan=0.0)
    
    mu = np.mean(channel_data)  # Compute mean
    delta = np.std(channel_data)  # Compute standard deviation
    skew = stats.skew(channel_data.flatten())  # Compute skewness, using flatten to convert the matrix to a 1D array
    
    features[0, k] = mu
    features[1, k] = delta
    features[2, k] = skew

print("color:", features)  # Mean, standard deviation, and skewness for each color channel
