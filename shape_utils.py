import cv2
import numpy as np

def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        return "square"
    else:
        return "circle"

def extract_features(image):
    # Basit özellik çıkarma örneği (örneğin, görüntü boyutları veya histogram)
    area = np.sum(image > 0)  # Alan (piksel sayısı)
    perimeter = cv2.arcLength(image, True)  # Çevre
    return [area, perimeter]

