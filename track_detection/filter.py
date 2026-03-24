import cv2
import numpy as np
# remove noise form the image
def median_filter(img, kernal_size):
    return cv2.medianBlur(np.array(img), kernal_size)