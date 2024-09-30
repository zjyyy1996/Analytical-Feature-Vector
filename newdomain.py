# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:47:32 2024

@author: lenovo
"""

import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, filedialog
from scipy.spatial.distance import euclidean

def get_npy_files(folder):
    """Get all .npy file paths in the folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]

def load_npy_data(file_paths):
    """Load .npy files and merge them into one array."""
    data_list = [np.load(file) for file in file_paths]
    return np.vstack(data_list)

def compute_centroid(vectors):
    """Calculate the centroid of 3D vectors."""
    return np.mean(vectors, axis=0)

def perform_tsne(data, perplexity):
    """Use t-SNE to reduce dimensionality to three dimensions."""
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(data)

def main():
    # Create window
    Tk().withdraw()  # Hide the main window
    all_centroids = []  # Store all centroids

    while True:
        folder_path = filedialog.askdirectory(title="Select Folder")  # Choose folder
        
        if not folder_path:
            break

        npy_files = get_npy_files(folder_path)
        
        if not npy_files:
            continue

        # Load data
        data = load_npy_data(npy_files)
        n_samples = data.shape[0]
        
        # Standardize data
        data = StandardScaler().fit_transform(data)

        # Ensure the number of samples is greater than 1
        if n_samples <= 1:
            continue

        # Use t-SNE to reduce dimensionality to three dimensions
        perplexity = min(30, n_samples - 1)
        three_d_vectors = perform_tsne(data, perplexity)

        # Calculate centroid
        centroid = compute_centroid(three_d_vectors)
        all_centroids.append(centroid)

    # Output all centroids
    print("All centroids' 3D vectors:")
    for idx, centroid in enumerate(all_centroids):
        print(f"Folder {idx + 1}: {centroid}")

    # Select user's vector
    new_vector_file = filedialog.askopenfilename(title="Select New .npy File", filetypes=[("Numpy files", "*.npy")])
    
    if new_vector_file:
        new_vector = np.load(new_vector_file)

        # Standardize new vector
        new_vector = StandardScaler().fit_transform(new_vector.reshape(1, -1))

        # Use t-SNE to reduce the new vector to three dimensions
        combined_data = np.vstack([data, new_vector])
        three_d_combined_vectors = perform_tsne(combined_data, perplexity)

        # Extract the 3D representation of the new vector
        new_vector_3d = three_d_combined_vectors[-1]

        print(f"new vector: {new_vector_3d}")

        # Calculate Euclidean distance to all centroids
        for idx, centroid in enumerate(all_centroids):
            distance = euclidean(new_vector_3d, centroid)
            print(f"Distance to folder {idx + 1}: {distance}")

if __name__ == "__main__":
    main()