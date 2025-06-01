import cv2
import numpy as np

def extract_features(contour):
    # Calculate area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate circularity
    circularity = 0
    if perimeter != 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)

    return [area, perimeter, circularity]