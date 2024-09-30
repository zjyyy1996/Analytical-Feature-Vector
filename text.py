# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 22:41:29 2024

@author: jingyan
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
import os
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte

# Function: Prompt user to select a file
def ask_file(prompt, filetypes):
    Tk().withdraw()
    return filedialog.askopenfilename(title=prompt, filetypes=filetypes)

# Function: Prompt user to select a directory
def ask_directory(prompt):
    Tk().withdraw()
    return filedialog.askdirectory(title=prompt)

# Crop image function
def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in the image.")
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# Image preprocessing: grayscale conversion and binarization
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    return binary

# Calculate shape parameters of the image
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

# Calculate color features
def calculate_color_features(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    features = np.zeros(shape=(9, ))  
    
    for i, img in enumerate([image, image_hsv, image_lab]):
        for k in range(img.shape[2]):
            channel_data = img[:, :, k].astype(np.float64)
            if np.isnan(channel_data).any():
                channel_data = np.nan_to_num(channel_data, nan=0.0)
            mu = np.mean(channel_data)
            delta = np.std(channel_data)
            skew = stats.skew(channel_data.flatten())
            features[i * 3 + k] = mu  
            features[i * 3 + k] = delta
            features[i * 3 + k] = skew
            
    return features.flatten()

# Calculate texture features
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

# Calculate intensity features
def calculate_intensity_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)
    return np.array([mean_intensity, std_intensity, min_intensity, max_intensity])

# Read text file and save as NumPy array
def process_text_file(file_path, save_dir):
    all_values = []
    with open(file_path, 'r') as file:
        for line in file:
            value = line.strip()
            if value:
                try:
                    all_values.append(float(value))
                except ValueError:
                    print(f"Skipping invalid value: {value} in file: {file_path}")
    
    values_array = np.array(all_values)
    common_prefix = get_common_prefix(file_path)
    save_path = os.path.join(save_dir, f"{common_prefix}_values.npy")
    np.save(save_path, values_array)
    #print(f"Text values saved to {save_path}")

# Get common prefix of a file
def get_common_prefix(file_path):
    if not file_path:
        return ""
    return os.path.splitext(os.path.basename(file_path))[0]

# Main function
def main():
    # Process image file
    img_path = ask_file("Select input image file", [("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
    if not img_path:
        print("No image file selected.")
        return
    image = cv2.imread(img_path)
    if image is None:
        print(f"Cannot read image or image does not exist: {img_path}")
        return

    # Prompt user to select a directory to save results
    save_dir = ask_directory("Select directory to save results")
    if not save_dir:
        print("No directory selected to save results.")
        return

    # Crop image
    cropped_image = crop_image(image)
    if cropped_image is None:
        print("Cannot crop the image.")
        return

    # Save cropped image
    img_name = os.path.basename(img_path)
    img_base_name, img_ext = os.path.splitext(img_name)
    save_cropped_image_path = os.path.join(save_dir, f"{img_base_name}_cropped{img_ext}")
    success = cv2.imwrite(save_cropped_image_path, cropped_image)

    # Extract shape, color, texture, and intensity features
    shape_parameters = calculate_shape_parameters(cropped_image)
    color_features = calculate_color_features(cropped_image)
    texture_features = calculate_texture_features(cropped_image)
    intensity_features = calculate_intensity_features(cropped_image)

    # Merge all features and save as text file
    all_features = np.concatenate((shape_parameters, color_features, texture_features, intensity_features))
    result_text = "\n".join([f"{feature:.4f}" for feature in all_features])
    
    # Merge all features and print to console
    all_features = np.concatenate((shape_parameters, color_features, texture_features, intensity_features))
    print("Composite feature vector：")
    print(all_features)  

    # Save feature file in the same directory as the image
    save_txt_path = os.path.join(save_dir, f"{img_base_name}_feature.txt")
    with open(save_txt_path, 'w') as f:
        f.write(result_text)

    # Process the text file and save the NumPy array
    text_file_path = ask_file("Select the text file to process", [("Text files", "*.txt")])
    if text_file_path:
        process_text_file(text_file_path, save_dir)

if __name__ == "__main__":
    main()
