"""
colony_analyzer.py
------------------
Complete bacterial colony analysis pipeline in one file
Analyzes bacterial colonies on agar plates with UV stress effects

Usage:
    python colony_analyzer.py --pre images/pre.png --post images/post.png
    python colony_analyzer.py --single images/plate.png
    python colony_analyzer.py --pre images/pre.png --post images/post.png --k 4

Dependencies: opencv-python-headless scikit-image scikit-learn scipy networkx matplotlib numpy pandas python-louvain
"""

import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import KDTree, ConvexHull
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

OUT_DIR = "outputs"

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING SETUP & STYLE
# ══════════════════════════════════════════════════════════════════════════════

# consistent style across all plots
plt.rcParams.update({
    "figure.facecolor":   "#1a1a1a",
    "axes.facecolor":     "#1a1a1a",
    "axes.edgecolor":     "#444444",
    "axes.labelcolor":    "#cccccc",
    "xtick.color":        "#888888",
    "ytick.color":        "#888888",
    "text.color":         "#cccccc",
    "grid.color":         "#2e2e2e",
    "grid.linewidth":     0.5,
    "axes.grid":          True,
    "axes.titlecolor":    "#ffffff",
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.facecolor":   "#2a2a2a",
    "legend.edgecolor":   "#444444",
    "legend.fontsize":    8,
    "figure.titlesize":   13,
    "figure.titleweight": "bold",
})

ACCENT  = "#00d4aa"    # teal
ACCENT2 = "#f5a623"    # amber
RED     = "#e05c5c"
PURPLE  = "#9b6dff"

TAB20 = plt.get_cmap("tab20")

# ══════════════════════════════════════════════════════════════════════════════
# PLATE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(img_rgb, plate_mask):
    """clahe on luminance + gaussian blur, masked to plate only"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

    # zero out everything outside the plate
    blurred[plate_mask == 0] = 0

    return blurred

# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# COLONY METRICS
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# KNN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def knn(df, k=5):
    """kdtree over colony centroids - returns distances, indices, and summary df"""
    if len(df) < 2:
        return None, None, None

    k = min(k, len(df) - 1)
    coords = df[["cy", "cx"]].values

    tree = KDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)  # +1 because self is included

    # drop self (col 0)
    dists = dists[:, 1:]
    idxs  = idxs[:, 1:]

    summary = pd.DataFrame({
        "label":        df["label"].values,
        "nn1_dist":     dists[:, 0],
        "mean_k_dist":  dists.mean(axis=1),
        "std_k_dist":   dists.std(axis=1),
    })

    return dists, idxs, summary

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(df, knn_idxs, knn_dists):
    """
    weighted undirected graph - nodes are colonies, edges are knn connections
    computes degree, clustering coeff, betweenness centrality, communities
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_node(
            int(row["label"]),
            pos=(float(row["cx"]), float(row["cy"])),
            area=float(row["area_px"]),
            diam=float(row["equiv_diam"]),
        )

    labels = df["label"].values
    for i, (nbrs, dists) in enumerate(zip(knn_idxs, knn_dists)):
        u = int(labels[i])
        for j, d in zip(nbrs, dists):
            v = int(labels[j])
            if not G.has_edge(u, v) and d > 0:
                G.add_edge(u, v, weight=float(1.0 / d), dist=float(d))

    degree     = dict(G.degree())
    clustering = nx.clustering(G, weight="weight")

    try:
        between = nx.betweenness_centrality(G, weight="dist")
    except Exception:
        between = {n: 0.0 for n in G.nodes()}

    communities = None
    if HAS_LOUVAIN and len(G.nodes) > 2:
        try:
            communities = community_louvain.best_partition(G, weight="weight")
        except Exception:
            pass

    stats = {
        "nodes":        G.number_of_nodes(),
        "edges":        G.number_of_edges(),
        "avg_degree":   np.mean(list(degree.values())),
        "avg_clust":    np.mean(list(clustering.values())),
        "n_components": len(list(nx.connected_components(G))),
        "density":      nx.density(G),
    }

    return G, degree, clustering, between, communities, stats

# ══════════════════════════════════════════════════════════════════════════════
# SHAPE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _color_labels(labels):
    """map integer label image -> rgb float for display"""
    out = np.zeros((*labels.shape, 3), dtype=np.float32)
    for u in np.unique(labels):
        if u == 0:
            continue
        c = TAB20((u % 20) / 20)[:3]
        out[labels == u] = c
    return out

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  saved → {path}")

def save_seg_clean(img_rgb, labels, out_dir, prefix):
    """colored segmentation overlay - no labels, no axes, just the visual"""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_position([0, 0, 1, 1])   # fill the whole figure

    overlay = np.clip(img_rgb.astype(float) / 255.0, 0.0, 1.0)
    colored = _color_labels(labels)
    mask = labels > 0

    alpha = 0.55
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * colored[mask]

    ax.imshow(np.clip(overlay, 0.0, 1.0))
    ax.axis("off")

    _save(fig, os.path.join(out_dir, f"{prefix}_seg_clean.png"))

def save_seg_labeled(img_rgb, labels, df, plate_info, out_dir, prefix):
    """segmentation with centroid markers and colony id labels"""
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle(f"{prefix.upper()} — colony segmentation", color="white")

    overlay = np.clip(img_rgb.astype(float) / 255.0, 0.0, 1.0)
    colored = _color_labels(labels)
    mask = labels > 0
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * colored[mask]
    ax.imshow(np.clip(overlay, 0.0, 1.0))

    # plate boundary
    circle = plt.Circle(
        (plate_info["cx"], plate_info["cy"]), plate_info["r"],
        fill=False, edgecolor=ACCENT, lw=1.5, linestyle="--", alpha=0.6,
    )
    ax.add_patch(circle)

    for _, row in df.iterrows():
        ax.plot(row["cx"], row["cy"], "+", color="white",
                markersize=7, markeredgewidth=1.2, zorder=5)
        ax.text(
            row["cx"] + 5, row["cy"] - 5, str(int(row["label"])),
            color="white", fontsize=5.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="#000000aa", ec="none"),
            zorder=6,
        )

    ax.set_title(f"{len(df)} colonies detected", color="#aaaaaa", fontsize=9)
    ax.axis("off")

    _save(fig, os.path.join(out_dir, f"{prefix}_seg_labeled.png"))

def save_size_dist(df, out_dir, prefix):
    """colony area histogram"""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(f"{prefix.upper()} — colony size distribution")

    areas = df["area_px"].values
    ax.hist(areas, bins=min(25, len(areas)),
            color=ACCENT, edgecolor="#111111", alpha=0.85, linewidth=0.5)

    med = np.median(areas)
    mn  = np.mean(areas)
    ax.axvline(med, color=ACCENT2, lw=1.8, linestyle="--", label=f"median  {med:,.0f} px²")
    ax.axvline(mn,  color=RED,     lw=1.8, linestyle=":",  label=f"mean    {mn:,.0f} px²")

    ax.set_xlabel("colony area  (px²)")
    ax.set_ylabel("count")
    ax.legend()

    _save(fig, os.path.join(out_dir, f"{prefix}_size_dist.png"))

def save_knn(knn_dists, out_dir, prefix):
    """nearest-neighbour distance histogram"""
    if knn_dists is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{prefix.upper()} — nearest-neighbour distances")

    nn1 = knn_dists[:, 0]
    all_k = knn_dists.flatten()

    axes[0].hist(nn1, bins=min(25, len(nn1)),
                 color=PURPLE, edgecolor="#111111", alpha=0.85, linewidth=0.5)
    axes[0].axvline(np.median(nn1), color=ACCENT2, lw=1.8, linestyle="--",
                    label=f"median  {np.median(nn1):.1f} px")
    axes[0].set_xlabel("distance to nearest neighbour  (px)")
    axes[0].set_ylabel("count")
    axes[0].set_title("NN1 distance")
    axes[0].legend()

    axes[1].hist(all_k, bins=min(30, len(all_k)),
                 color=ACCENT, edgecolor="#111111", alpha=0.85, linewidth=0.5)
    axes[1].set_xlabel("distance  (px)")
    axes[1].set_ylabel("count")
    axes[1].set_title("all k-NN distances")

    _save(fig, os.path.join(out_dir, f"{prefix}_knn.png"))

def save_graph(img_rgb, G, degree, communities, out_dir, prefix):
    """graph overlay on the plate image"""
    if G is None or len(G.nodes) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle(f"{prefix.upper()} — colony graph  (nodes coloured by community/degree)")

    ax.imshow(img_rgb, alpha=0.45)

    pos = nx.get_node_attributes(G, "pos")

    # edges
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color="#00d4aa", alpha=0.25, lw=0.7, zorder=1)

    # nodes
    nodes  = list(pos.keys())
    xs     = [pos[n][0] for n in nodes]
    ys     = [pos[n][1] for n in nodes]
    c_vals = [communities.get(n, 0) for n in nodes] if communities \
             else [degree.get(n, 0) for n in nodes]

    sc = ax.scatter(xs, ys, c=c_vals, cmap="tab10" if communities else "YlOrRd",
                    s=35, zorder=3, edgecolors="white", linewidths=0.4)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01)
    cbar.set_label("community id" if communities else "degree", color="#aaaaaa")
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")

    ax.axis("off")

    _save(fig, os.path.join(out_dir, f"{prefix}_graph.png"))

def save_orientation(df, out_dir, prefix):
    """rose plot of colony growth direction"""
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f"{prefix.upper()} — growth orientation")

    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#1a1a1a")

    angles = df["orientation_deg"].dropna().values % 180
    angles_rad = np.deg2rad(angles)
    # mirror to make it a full rose (orientation is axial)
    all_a = np.concatenate([angles_rad, angles_rad + np.pi])

    bins = np.linspace(0, 2 * np.pi, 37)
    counts, _ = np.histogram(all_a, bins=bins)
    width = bins[1] - bins[0]

    ax.bar(bins[:-1], counts, width=width, bottom=0,
           color=ACCENT, edgecolor="#111111", alpha=0.8, align="edge")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.tick_params(colors="#888888")
    ax.spines["polar"].set_edgecolor("#444444")

    _save(fig, os.path.join(out_dir, f"{prefix}_orientation.png"))

def save_comparison(comp, out_dir):
    """side by side bar charts for pre vs post comparison"""
    labels  = ["pre-UV", "post-UV"]
    colors  = [ACCENT, RED]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("pre vs post UV — plate comparison")

    metrics = [
        ("colony count",     [comp["colonies_pre"],     comp["colonies_post"]]),
        ("median area (px²)",[comp["median_area_pre"],  comp["median_area_post"]]),
        ("plate coverage %", [comp["coverage_pct_pre"], comp["coverage_pct_post"]]),
    ]

    for ax, (title, vals) in zip(axes, metrics):
        bars = ax.bar(labels, vals, color=colors, edgecolor="#111111", width=0.5)
        ax.set_title(title)

        # value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{v:.1f}" if isinstance(v, float) else str(v),
                ha="center", va="bottom", fontsize=9, color="white",
            )

        ax.set_ylim(0, max(vals) * 1.2 + 1)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "comparison.png"))

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def analyse_plate(img_path, tag, k=5):
    """full pipeline for one plate - returns dict of results"""
    print(f"\n[{tag}]  {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"  error: can't read {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"  detecting plate boundary ...")
    plate_mask, plate_info = detect_plate(img_rgb)

    print(f"  preprocessing ...")
    gray = preprocess(img_rgb, plate_mask)

    print(f"  segmenting colonies ...")
    labels, binary, dist = segment(gray, plate_mask, plate_info)

    print(f"  measuring colony props ...")
    df = measure(labels, img_rgb, plate_info)
    print(f"  → {len(df)} colonies after edge filtering")

    print(f"  running knn (k={k}) ...")
    knn_dists, knn_idxs, knn_summary = knn(df, k=k)

    print(f"  building colony graph ...")
    if knn_idxs is not None:
        G, degree, clustering, between, communities, g_stats = \
            build_graph(df, knn_idxs, knn_dists)
    else:
        G, degree, clustering, between, communities, g_stats = \
            None, {}, {}, {}, None, {}

    print(f"  shape analysis ...")
    shape_df = shape_analysis(labels, df)

    # ── save figures ──────────────────────────────────────────────────────────
    prefix = tag.lower().replace(" ", "_").replace("-", "_")

    print(f"  saving figures ...")
    save_seg_clean  (img_rgb, labels, OUT_DIR, prefix)
    save_seg_labeled(img_rgb, labels, df, plate_info, OUT_DIR, prefix)
    save_size_dist  (df, OUT_DIR, prefix)
    save_knn        (knn_dists, OUT_DIR, prefix)
    save_graph      (img_rgb, G, degree, communities, OUT_DIR, prefix)
    save_orientation(df, OUT_DIR, prefix)

    # ── console report ────────────────────────────────────────────────────────
    print_report(tag, df, knn_summary, g_stats, shape_df)

    return {
        "img_rgb":    img_rgb,
        "labels":     labels,
        "df":         df,
        "knn_dists":  knn_dists,
        "knn_idxs":   knn_idxs,
        "G":          G,
        "degree":     degree,
        "communities":communities,
        "shape_df":   shape_df,
        "plate_info": plate_info,
        "g_stats":    g_stats,
    }

def print_report(tag, df, knn_summary, g_stats, shape_df):
    """print a clean summary to stdout"""
    sep = "─" * 55
    print(f"\n{'═'*55}")
    print(f"  {tag}")
    print(f"{'═'*55}")

    print(f"  segmentation")
    print(f"  {sep}")
    print(f"  colonies          : {len(df)}")
    if len(df):
        print(f"  mean area         : {df['area_px'].mean():>10,.1f} px²")
        print(f"  median area       : {df['area_px'].median():>10,.1f} px²")
        print(f"  mean diameter     : {df['equiv_diam'].mean():>10.1f} px")
        print(f"  mean eccentricity : {df['eccentricity'].mean():>10.3f}   (0=circle 1=line)")
        print(f"  mean solidity     : {df['solidity'].mean():>10.3f}   (1=convex)")

    if knn_summary is not None:
        print(f"\n  nearest neighbour")
        print(f"  {sep}")
        print(f"  mean nn1 dist     : {knn_summary['nn1_dist'].mean():>10.1f} px")
        print(f"  median nn1 dist   : {knn_summary['nn1_dist'].median():>10.1f} px")
        print(f"  min nn1 dist      : {knn_summary['nn1_dist'].min():>10.1f} px")
        print(f"  max nn1 dist      : {knn_summary['nn1_dist'].max():>10.1f} px")

    if g_stats:
        print(f"\n  graph")
        print(f"  {sep}")
        for k, v in g_stats.items():
            val = f"{v:.3f}" if isinstance(v, float) else str(v)
            print(f"  {k:<20}: {val}")

    if len(shape_df):
        print(f"\n  shape")
        print(f"  {sep}")
        print(f"  mean compactness  : {shape_df['compactness'].dropna().mean():>10.3f}   (1=circle)")
        print(f"  mean elongation   : {shape_df['elongation'].dropna().mean():>10.3f}")
        valid_a = shape_df["ellipse_angle"].dropna()
        if len(valid_a):
            print(f"  mean angle        : {valid_a.mean():>10.1f}°")
            print(f"  angle std         : {valid_a.std():>10.1f}°")

    print(f"\n  top 10 colonies by area")
    cols = ["label", "area_px", "equiv_diam", "eccentricity", "orientation_deg", "solidity"]
    print(df.nlargest(min(10, len(df)), "area_px")[cols].to_string(index=False))
    print()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="bacterial colony analyser")
    parser.add_argument("--pre",    help="path to pre-UV image")
    parser.add_argument("--post",   help="path to post-UV image")
    parser.add_argument("--single", help="analyse a single image")
    parser.add_argument("--k",      type=int, default=5, help="k for knn (default 5)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.single:
        analyse_plate(args.single, tag="single", k=args.k)

    elif args.pre and args.post:
        pre  = analyse_plate(args.pre,  tag="pre-UV",  k=args.k)
        post = analyse_plate(args.post, tag="post-UV", k=args.k)

        print("\n[comparison]  pre vs post ...")
        comp = compare(pre["df"], post["df"], pre["plate_info"], post["plate_info"])

        print(f"\n{'═'*55}")
        print("  pre / post summary")
        print(f"{'═'*55}")
        for key, val in comp.items():
            print(f"  {key:<28}: {val:.2f}" if isinstance(val, float) else
                  f"  {key:<28}: {val}")

        save_comparison(comp, OUT_DIR)

    else:
        parser.print_help()
        sys.exit(1)

    print(f"\ndone. all figures saved to ./{OUT_DIR}/")

if __name__ == "__main__":
    main()