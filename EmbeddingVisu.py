# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:09:01 2024

@author: lenovo
"""

import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_npy_files_from_folder(folder_path):
    embeddings = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            embedding = np.load(file_path)
            embeddings.append(embedding)
    return np.array(embeddings)

def select_folders():
    folder_paths = []
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    while True:
        # Open a file dialog to select a folder
        folder_path = filedialog.askdirectory(title="Select a folder containing .npy files")
        if not folder_path:
            break
        folder_paths.append(folder_path)
        # Ask the user if they want to select another folder
        if not tk.messagebox.askyesno("Continue?", "Do you want to select another folder?"):
            break
    return folder_paths

def main():
    # Select multiple folders
    folder_paths = select_folders()

    if not folder_paths:
        print("No folders were selected.")
        return None

    embeddings_list = []
    labels_list = []
    folder_colors = []
    color_map = plt.cm.get_cmap('tab10', len(folder_paths))

    for i, folder_path in enumerate(folder_paths):
        folder_name = os.path.basename(folder_path)
        embeddings = load_npy_files_from_folder(folder_path)
        embeddings_list.append(embeddings)
        labels_list.append([folder_name] * len(embeddings))
        folder_colors.append(color_map(i))

    all_embeddings = np.vstack(embeddings_list)
    all_labels = np.concatenate(labels_list)

    # Compute t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(all_embeddings)

    # Plot the 3D t-SNE results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start_idx = 0
    for i, folder_path in enumerate(folder_paths):
        embeddings = embeddings_list[i]
        num_embeddings = len(embeddings)
        tsne_subset = tsne_results[start_idx:start_idx + num_embeddings]
        ax.scatter(tsne_subset[:, 0], tsne_subset[:, 1], tsne_subset[:, 2], color=folder_colors[i], label=os.path.basename(folder_path))
        start_idx += num_embeddings

    ax.set_title('3D t-SNE of Embeddings')
    ax.legend()

    # Instead of plt.show(), return the figure to be used in the GUI
    return fig

if __name__ == "__main__":
    main()