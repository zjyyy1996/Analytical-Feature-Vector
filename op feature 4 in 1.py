# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:18:04 2024

@author: jingyan
"""

import cv2
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte, color
from tkinter import Tk, filedialog
import os

def ask_directory(prompt):
    Tk().withdraw()
    return filedialog.askdirectory(title=prompt)

def ask_file(prompt):
    Tk().withdraw()
    return filedialog.askopenfilename(title=prompt, filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    return binary

def calculate_shape_parameters(image):
    binary = preprocess_image(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in the image.")
        return None
    c = max(contours, key=cv2.contourArea)
    
    moments = cv2.moments(c)
    hu_moments = cv2.HuMoments(moments).flatten()

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    convexity = float(cv2.contourArea(c)) / hull_area
    
    if len(c) < 5:
        ellipse = cv2.minAreaRect(c)
        (center, (MA, ma), angle) = ellipse
        MA, ma = max(MA, ma), min(MA, ma)
        eccentricity = 0
    else:
        ellipse = cv2.fitEllipse(c)
        (center, (MA, ma), angle) = ellipse
        MA, ma = max(MA, ma), min(MA, ma)
        eccentricity = np.sqrt(1 - (ma / MA) ** 2)
    
    roundness = (4 * cv2.contourArea(c)) / (np.pi * (MA / 2) ** 2)
    circularity = (4 * np.pi * cv2.contourArea(c)) / (cv2.arcLength(c, True) ** 2)

    return np.concatenate(([aspect_ratio, roundness, eccentricity, convexity, circularity], hu_moments))

def calculate_color_features(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    features = np.zeros(shape=(9, ))  # 确保有足够的空间存储特征
    
    for i, img in enumerate([image, image_hsv, image_lab]):
        for k in range(img.shape[2]):
            channel_data = img[:, :, k].astype(np.float64)
            if np.isnan(channel_data).any():
                channel_data = np.nan_to_num(channel_data, nan=0.0)
            mu = np.mean(channel_data)
            delta = np.std(channel_data)
            skew = stats.skew(channel_data.flatten())
            features[i * 3 + k] = mu  # 修正特征存储的索引
            features[i * 3 + k] = delta
            features[i * 3 + k] = skew
            
    return features.flatten()

def calculate_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    
    entropy_value = entropy(img_as_ubyte(gray), disk(5))
    avg_entropy = np.mean(entropy_value)
    
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    
    return np.concatenate(([contrast, homogeneity, energy, correlation, dissimilarity, avg_entropy], lbp_hist))

def calculate_intensity_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)
    return np.array([mean_intensity, std_intensity, min_intensity, max_intensity])

def main():
    img_path = ask_file("Select the input image file")
    if not img_path:
        print("No image file selected.")
        return
    image = cv2.imread(img_path)
    if image is None:
        print(f"Image not found or cannot be read at {img_path}")
        return

    shape_parameters = calculate_shape_parameters(image)
    if shape_parameters is None:
        print("Failed to compute shape parameters.")
        return
    color_features = calculate_color_features(image)
    texture_features = calculate_texture_features(image)
    intensity_features = calculate_intensity_features(image)

    all_features = np.concatenate((shape_parameters, color_features, texture_features, intensity_features))
    all_features = (all_features - all_features.mean()) / all_features.std()  # 标准化处理

    result_text = "\n".join([f"{feature:.4f}" for feature in all_features])

    save_dir = ask_directory("Select the directory to save the results")
    if not save_dir:
        print("No directory selected to save results.")
        return

    img_name = os.path.basename(img_path)
    img_base_name, _ = os.path.splitext(img_name)
    save_txt_path = os.path.join(save_dir, f"{img_base_name}_feature.txt")

    with open(save_txt_path, 'w') as f:
        f.write(result_text)


if __name__ == "__main__":
    main()
