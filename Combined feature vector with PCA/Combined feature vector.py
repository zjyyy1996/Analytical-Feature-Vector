# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:40:20 2024

@author: jingyan
"""

import matplotlib.pyplot as plt
from skimage.filters import roberts
from skimage.feature import local_binary_pattern
import cv2
import numpy as np
from scipy import stats

# Define the image path
image_path = "D:\\jingyan\\MA\\final output directory\\grabcut2.png"

# Function to extract edge detection features
def extract_edge_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image file not found or cannot be read")

    # Apply Roberts edge detection
    edge_roberts = roberts(image)

    # Convert the edge detection result to binary using Otsu's thresholding
    _, binary_edges = cv2.threshold((edge_roberts * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Flatten the contours to create a vector
    edge_vector = np.concatenate([c.flatten() for c in contours]) if contours else np.array([])
    return edge_vector

# Function to extract color features
def extract_color_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Initialize feature array to store mean, standard deviation, and skewness for each RGB channel
    features = np.zeros(shape=(3, 3))
    for k in range(image.shape[2]):
        channel_data = image[:, :, k].astype(np.float64)
        mu = np.mean(channel_data)  # Compute mean
        delta = np.std(channel_data)  # Compute standard deviation
        skew = stats.skew(channel_data.flatten())  # Compute skewness, using flatten to convert the matrix to a 1D array
        features[0, k] = mu
        features[1, k] = delta
        features[2, k] = skew

    color_vector = features.flatten()
    return color_vector

# Function to extract texture features
def extract_texture_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image loading failed")

    # Enhance contrast using histogram equalization
    image = cv2.equalizeHist(image)

    # Define LBP parameters
    radius = 3  # Radius for LBP
    n_points = 8 * radius  # Number of sampling points around the center

    # Compute LBP
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

    # Convert LBP image to a vector
    lbp_vector = lbp_image.flatten()
    return lbp_vector

# Extract all features
edge_vector = extract_edge_features(image_path)
color_vector = extract_color_features(image_path)
lbp_vector = extract_texture_features(image_path)

# Combine all feature vectors into a single feature vector
combined_vector = np.concatenate((edge_vector, color_vector, lbp_vector))

print("Combined feature vector:", combined_vector)
print("Combined feature vector shape:", combined_vector.shape)
