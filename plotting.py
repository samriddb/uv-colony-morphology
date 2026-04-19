import os
import cv2
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

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


# ──────────────────────────────────────────────────────────────────────────────
# individual figure functions - each saves its own file
# ──────────────────────────────────────────────────────────────────────────────

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
