"""
colony_analysis/main.py
-----------------------
runs the full bacterial colony analysis pipeline on a pre/post image pair
(or a single image with --single)

usage:
    python main.py --pre images/pre.png --post images/post.png
    python main.py --single images/pre.png
    python main.py --pre images/pre.png --post images/post.png --k 4
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

from plate_detection import detect_plate
from preprocessing   import preprocess
from segmentation    import segment
from colony_metrics  import measure
from knn_analysis    import knn
from graph_analysis  import build_graph
from shape_analysis  import shape_analysis
from comparison      import compare
import plotting


OUT_DIR = "outputs"


# ──────────────────────────────────────────────────────────────────────────────

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
    plotting.save_seg_clean  (img_rgb, labels, OUT_DIR, prefix)
    plotting.save_seg_labeled(img_rgb, labels, df, plate_info, OUT_DIR, prefix)
    plotting.save_size_dist  (df, OUT_DIR, prefix)
    plotting.save_knn        (knn_dists, OUT_DIR, prefix)
    plotting.save_graph      (img_rgb, G, degree, communities, OUT_DIR, prefix)
    plotting.save_orientation(df, OUT_DIR, prefix)

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


# ──────────────────────────────────────────────────────────────────────────────

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

        plotting.save_comparison(comp, OUT_DIR)

    else:
        parser.print_help()
        sys.exit(1)

    print(f"\ndone. all figures saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
