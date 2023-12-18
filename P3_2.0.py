# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:27:35 2023

@author: 15127
"""

import cv2
import numpy as np

# Load the image
image = cv2.imread('motherboard_image.JPEG')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)##########
cv2.imwrite('thresholded_image.jpg', thresholded)
thresholdedImage = cv2.imread('thresholded_image.jpg')



# Apply GaussianBlur to reduce noise and help contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to find edges in the image
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)##############

# Define the minimum and maximum contour area thresholds
min_area = 500
max_area = 5000

# Create a copy of the original image
contour_image = image.copy()

# Iterate through contours and filter based on area
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        filtered_contours.append(contour)
        # Optionally, draw the contours on the image
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)

# Create an empty mask
mask = np.zeros_like(gray)

# Draw contours on the mask
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)######################

# Save the contour mask as an image
cv2.imwrite('contour_mask.jpg', mask)
maskimage = cv2.imread('contour_mask.jpg')

# Extract PCB using bitwise_and
pcb_extraction = cv2.bitwise_and(thresholdedImage, maskimage)

# Display the original image
cv2.imshow('Original Image', image)

# Display the thresholded image
cv2.imshow('Thresholded Image', thresholded)

# Display the contour mask
cv2.imshow('Contour Mask', mask)



# Save the extracted PCB as an image
cv2.imwrite('extracted_pcb.jpg', pcb_extraction)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()