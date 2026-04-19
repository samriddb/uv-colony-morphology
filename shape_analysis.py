import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def shape_analysis(labels, df):
    """
    per colony: ellipse fit, convex hull, compactness, elongation
    returns dataframe of shape metrics
    """

    rows = []

    for _, row in df.iterrows():
        lbl = int(row["label"])
        colony_mask = (labels == lbl).astype(np.uint8) * 255

        contours, _ = cv2.findContours(colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)

        # fit ellipse to get principal axis angle
        ellipse_angle = np.nan
        if len(cnt) >= 5:
            _, (ma, mi), angle = cv2.fitEllipse(cnt)
            ellipse_angle = angle

        # convex hull area
        hull_area = np.nan
        try:
            pts = cnt[:, 0, :].astype(float)
            if len(pts) >= 3:
                hull_area = ConvexHull(pts).volume  # 2d: volume = area
        except Exception:
            pass

        peri = row["perimeter"]
        compactness = (4 * np.pi * row["area_px"]) / (peri ** 2) if peri > 0 else np.nan
        elongation  = row["major_axis"] / row["minor_axis"] if row["minor_axis"] > 0 else np.nan

        rows.append({
            "label":         lbl,
            "ellipse_angle": ellipse_angle,
            "hull_area":     hull_area,
            "hull_ratio":    row["area_px"] / hull_area if hull_area else np.nan,
            "compactness":   compactness,
            "elongation":    elongation,
        })

    return pd.DataFrame(rows)
