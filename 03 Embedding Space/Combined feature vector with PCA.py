# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:55:45 2024

@author: jingyan
"""

import matplotlib.pyplot as plt
from skimage.filters import roberts
from skimage.feature import local_binary_pattern
import cv2
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to select an image file interactively
def select_image_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    root.destroy()
    return file_path

# Select the image file interactively
image_path = select_image_file()
if not image_path:
    raise FileNotFoundError("No image file selected")

# Function to extract edge detection features
def extract_edge_features(image):
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
def extract_color_features(image):
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
def extract_texture_features(image):
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

# Load the original image
original_image = cv2.imread(image_path)
if original_image is None:
    raise FileNotFoundError(f"Image file not found or cannot be read: {image_path}")

# Generate transformed images (e.g., rotated versions)
transformed_images = [original_image]
for angle in [90, 180, 270]:
    M = cv2.getRotationMatrix2D((original_image.shape[1] / 2, original_image.shape[0] / 2), angle, 1)
    rotated_image = cv2.warpAffine(original_image, M, (original_image.shape[1], original_image.shape[0]))
    transformed_images.append(rotated_image)

# Initialize lists to store feature vectors
edge_vectors = []
color_vectors = []
lbp_vectors = []

# Process each transformed image and extract features
for img in transformed_images:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_vectors.append(extract_edge_features(gray_image))
    color_vectors.append(extract_color_features(img))
    lbp_vectors.append(extract_texture_features(gray_image))

# Combine all feature vectors into a single feature matrix
combined_features = np.array([np.concatenate((edge, color, lbp)) for edge, color, lbp in zip(edge_vectors, color_vectors, lbp_vectors)])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=3)  # Reduce to 3 dimensions for visualization
reduced_features = pca.fit_transform(combined_features)

# Visualize the reduced features
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2])
ax.set_title("Vectors Represented in Embedding Space")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()
