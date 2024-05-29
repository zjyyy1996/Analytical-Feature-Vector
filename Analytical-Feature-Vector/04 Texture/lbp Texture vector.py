# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:59:43 2024

@author: jingyan
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

def detect_texture_defects_with_lbp(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image loading failed")
        return

    # Enhance contrast using histogram equalization
    image = cv2.equalizeHist(image)

    # Define LBP parameters
    radius = 3  # Radius for LBP
    n_points = 8 * radius  # Number of sampling points around the center

    # Compute LBP
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

    # Convert LBP image to a vector
    lbp_vector = lbp_image.flatten()

    # Print LBP vector
    print("LBP vector:", lbp_vector)

    # Visualize the LBP image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lbp_image, cmap='gray')
    plt.title('LBP Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Provide image path
image_path = "D:\\jingyan\\MA\\final output directory\\grabcut2.png"
detect_texture_defects_with_lbp(image_path)

