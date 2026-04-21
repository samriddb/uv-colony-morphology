import cv2
import numpy as np


def preprocess(img_rgb, plate_mask):
    """clahe on luminance + gaussian blur, masked to plate only"""

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 4)

    # zero out everything outside the plate
    blurred[plate_mask == 0] = 0

    return blurred
