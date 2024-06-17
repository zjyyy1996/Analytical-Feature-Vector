# -*- coding: utf-8 -*-
"""
Created on Wed May 29 01:53:45 2024

@author: jingyan
"""

 
import cv2
import numpy as np
import os
from tkinter import Tk, filedialog

# Function to prompt the user to select a directory
def ask_directory(prompt):
    Tk().withdraw()
    return filedialog.askdirectory(title=prompt)

# Function to prompt the user to select a file
def ask_file(prompt):
    Tk().withdraw()
    return filedialog.askopenfilename(title=prompt)

# Ask user to select the output directory
output_dir = ask_directory("Select the output directory")

# Ensure the directories exist
os.makedirs(output_dir, exist_ok=True)

# Ask user to select the input image file
img_path = ask_file("Select the input image file")
img = cv2.imread(img_path)
imgmask = img.copy()      
drawing = False

# Get the base name of the input image file
img_name = os.path.basename(img_path)
img_base_name, img_ext = os.path.splitext(img_name)

# Save the input image to the output directory
input_image_output_path = os.path.join(output_dir, img_name)
cv2.imwrite(input_image_output_path, img)

# Initialize mask and models for grabCut algorithm
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = [0, 0, 0, 0]  
leftButtonDown = False
leftButtonUp = True

# Mouse callback function for selecting the ROI (Region of Interest)
def on_mouse(event, x, y, flag, param):        
    global rect
    global leftButtonDown
    global leftButtonUp

    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[2] = x
        rect[1] = y
        rect[3] = y
        leftButtonDown = True
        leftButtonUp = False

    if event == cv2.EVENT_MOUSEMOVE:
        if leftButtonDown and not leftButtonUp:
            rect[2] = x
            rect[3] = y        

    if event == cv2.EVENT_LBUTTONUP:
        if leftButtonDown and not leftButtonUp:
            x_min = min(rect[0], rect[2])
            y_min = min(rect[1], rect[3])
            x_max = max(rect[0], rect[2])
            y_max = max(rect[1], rect[3])
            rect[0] = x_min
            rect[1] = y_min
            rect[2] = x_max
            rect[3] = y_max
            leftButtonDown = False      
            leftButtonUp = True

# Create the bounding box window
# Create a window and set the mouse callback function
bounding_window_name = 'bounding_box'
bounding_window_title = 'Draw a bounding box around the defect'
cv2.namedWindow(bounding_window_name) 
cv2.setWindowTitle(bounding_window_name, bounding_window_title)
cv2.setMouseCallback(bounding_window_name, on_mouse)
cv2.imshow(bounding_window_name, img)

# Main loop to handle the user interaction for ROI selection
while cv2.waitKey(2) == -1:
    if leftButtonDown and not leftButtonUp:  
        img_copy = img.copy()
        cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)  
        cv2.namedWindow(bounding_window_name, cv2.WINDOW_NORMAL) 
        cv2.imshow(bounding_window_name, img_copy)

    elif not leftButtonDown and leftButtonUp and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
        # Convert to width and height
        rect[2] = rect[2] - rect[0]
        rect[3] = rect[3] - rect[1]
        rect_copy = tuple(rect.copy())   
        rect = [0, 0, 0, 0]
        cv2.grabCut(img, mask, rect_copy, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img_show = img * mask2[:, :, np.newaxis]      
        cv2.imshow('Initial mask', img_show)                     
        
        # Initialize drawing flags
        drawing_bg = False
        drawing_fg = False

        # Mouse callback function for fine-tuning the mask
        def draw_2(event, x, y, flag, param):
            global img, imgmask, drawing_bg, drawing_fg
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing_bg = True
       
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing_bg:
                    cv2.circle(imgmask, (x, y), 3, (0, 0, 0), -1)
  
            elif event == cv2.EVENT_LBUTTONUP:
                drawing_bg = False
    
            if event == cv2.EVENT_RBUTTONDOWN:
                drawing_fg = True
    
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing_fg:
                    cv2.circle(imgmask, (x, y), 3, (255, 255, 255), -1)
  
            elif event == cv2.EVENT_RBUTTONUP:
                drawing_fg = False
    
            elif event == cv2.EVENT_RBUTTONDBLCLK:
                mask_img_name = f"{img_base_name}_mask{img_ext}"
                file_path = os.path.join(output_dir, mask_img_name)
                cv2.imwrite(file_path, imgmask)  # Save the mask image
                imgmask_BF = cv2.imread(file_path)              
                mask3 = cv2.cvtColor(imgmask_BF, cv2.COLOR_BGR2GRAY)
                mask[mask3 == 0] = 0                                     
                mask[mask3 == 255] = 1  
                cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                mask3 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                img = img * mask3[:, :, np.newaxis]                         
                cv2.imshow('Segmented area', img)
                
                # Save the final image to the specified directory
                final_output_path = os.path.join(output_dir, f"{img_base_name}_segment{img_ext}")
                cv2.imwrite(final_output_path, img)
                print(f"Image saved to {final_output_path}")
                
            cv2.imshow(outline_filling_window_name, imgmask)                
        
        outline_filling_window_name = 'Draw outline and fill defective area'
        cv2.namedWindow( outline_filling_window_name)                       
        cv2.setMouseCallback( outline_filling_window_name, draw_2)              
        cv2.imshow(outline_filling_window_name, imgmask)

cv2.waitKey(0)
cv2.destroyAllWindows()