import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops


def measure(labels, img_rgb, plate_info, edge_margin=0.10):
    """
    regionprops on each labelled colony
    filters out anything whose centroid is too close to the plate edge
    returns dataframe of per-colony measurements
    """

    cx, cy, r = plate_info["cx"], plate_info["cy"], plate_info["r"]
    max_dist = r * (1.0 - edge_margin)  # colonies must sit inside this radius

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    props = regionprops(labels, intensity_image=gray)

    rows = []
    for p in props:
        if p.area < 300:
            continue

        yc, xc = p.centroid

        # drop colonies whose centroid is near the plate boundary
        dist_to_center = np.sqrt((xc - cx) ** 2 + (yc - cy) ** 2)
        if dist_to_center > max_dist:
            continue

        rows.append({
            "label":           p.label,
            "cy":              yc,
            "cx":              xc,
            "area_px":         p.area,
            "perimeter":       p.perimeter,
            "equiv_diam":      p.equivalent_diameter_area,
            "eccentricity":    p.eccentricity,
            "orientation_deg": np.degrees(p.orientation),
            "solidity":        p.solidity,
            "major_axis":      p.axis_major_length,
            "minor_axis":      p.axis_minor_length,
            "mean_intensity":  p.intensity_mean,
            "bbox":            p.bbox,
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    return df
