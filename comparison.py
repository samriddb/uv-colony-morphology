import numpy as np


def compare(df_pre, df_post, plate_pre, plate_post):
    """high level delta between pre and post plates"""

    def plate_area(p):
        return np.pi * p["r"] ** 2

    cov_pre  = df_pre["area_px"].sum()  / plate_area(plate_pre)
    cov_post = df_post["area_px"].sum() / plate_area(plate_post)

    return {
        "colonies_pre":       len(df_pre),
        "colonies_post":      len(df_post),
        "colony_delta":       len(df_post) - len(df_pre),
        "median_area_pre":    df_pre["area_px"].median()  if len(df_pre)  else 0,
        "median_area_post":   df_post["area_px"].median() if len(df_post) else 0,
        "coverage_pct_pre":   cov_pre  * 100,
        "coverage_pct_post":  cov_post * 100,
        "coverage_delta_pct": (cov_post - cov_pre) * 100,
    }
