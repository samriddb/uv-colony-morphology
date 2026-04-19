import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects


def segment(gray_masked, plate_mask, plate_info, min_colony_px=300, edge_margin=0.10):
    """
    otsu threshold -> morph clean -> distance transform -> watershed
    then drops anything whose centroid lands within edge_margin * radius of the plate boundary
    (those are rim artifacts, not real colonies)
    """

    cx, cy, r = plate_info["cx"], plate_info["cy"], plate_info["r"]

    # --- erode the usable area inward so the rim never seeds a colony ---
    rim_px = int(r * edge_margin)
    inner_mask = plate_mask.copy()
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_px * 2 + 1, rim_px * 2 + 1))
    inner_mask = cv2.erode(inner_mask, kernel_erode, iterations=1)

    # --- otsu threshold (computed only on plate pixels to avoid bias) ---
    plate_px = gray_masked[inner_mask == 255]
    thresh_val, _ = cv2.threshold(
        plate_px.reshape(-1, 1).astype(np.uint8),
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    binary = np.zeros_like(gray_masked, dtype=np.uint8)
    binary[(gray_masked > thresh_val) & (inner_mask == 255)] = 255

    # --- morphological clean ---
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=2)

    cleaned = remove_small_objects(closed.astype(bool), max_size=min_colony_px, connectivity=2)
    cleaned_u8 = cleaned.astype(np.uint8) * 255

    # --- distance transform + local maxima as watershed seeds ---
    dist = ndimage.distance_transform_edt(cleaned)
    min_seed_gap = max(8, int(dist.max() * 0.15))

    coords = peak_local_max(dist, min_distance=min_seed_gap, labels=cleaned)
    seed_mask = np.zeros(dist.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True

    markers, _ = ndimage.label(seed_mask)
    labels = watershed(-dist, markers, mask=cleaned)

    return labels, cleaned_u8, dist
