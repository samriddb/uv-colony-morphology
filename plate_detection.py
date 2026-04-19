import cv2
import numpy as np


def detect_plate(img_rgb):
    """find the circular plate boundary + return binary mask and center/radius"""

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=60,
        param2=35,
        minRadius=int(min(h, w) * 0.30),
        maxRadius=int(min(h, w) * 0.55),
    )

    mask = np.zeros((h, w), dtype=np.uint8)

    if circles is not None:
        cx, cy, r = np.round(circles[0][0]).astype(int)

        # shrink radius to cut off the metal rim + any edge artifact
        r_inner = int(r * 0.88)
        cv2.circle(mask, (cx, cy), r_inner, 255, -1)
        info = {"cx": int(cx), "cy": int(cy), "r": int(r_inner)}
    else:
        # fallback - assume whole image is the plate
        mask[:] = 255
        info = {"cx": w // 2, "cy": h // 2, "r": min(h, w) // 2}

    return mask, info
