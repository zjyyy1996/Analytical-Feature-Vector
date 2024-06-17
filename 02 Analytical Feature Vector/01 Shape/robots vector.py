# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:19:54 2024

@author: jingyan
"""

import matplotlib.pyplot as plt
from skimage.filters import roberts
import cv2
import numpy as np

# Read the image and convert to grayscale
image_path = "D:\\jingyan\\MA\\final output directory\\grabcut2.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image file not found or cannot be read")
else:
    # Apply Roberts edge detection
    edge_roberts = roberts(image)

    # Convert the edge detection result to binary using Otsu's thresholding
    _, binary_edges = cv2.threshold((edge_roberts * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Flatten the contours to create a vector
    edge_vector = np.concatenate([c.flatten() for c in contours]) if contours else np.array([])

    if edge_vector.size == 0:
        print("The edge detection vector is empty after processing.")
    else:
        print("Edge detection vector:", edge_vector.tolist())

    # Plot the original image and the edge detection result
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(binary_edges, cmap='gray')
    ax[1].set_title('Roberts Edge Detection')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

