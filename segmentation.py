# import cv2
# import numpy as np
# from scipy import ndimage
# from skimage.feature import peak_local_max
# from skimage.segmentation import watershed
# from skimage.morphology import remove_small_objects


# def segment(gray_masked, plate_mask, plate_info, min_colony_px=300, edge_margin=0.10):
#     """
#     otsu threshold -> morph clean -> distance transform -> watershed
#     then drops anything whose centroid lands within edge_margin * radius of the plate boundary
#     (those are rim artifacts, not real colonies)
#     """

#     cx, cy, r = plate_info["cx"], plate_info["cy"], plate_info["r"]

#     # --- erode the usable area inward so the rim never seeds a colony ---
#     rim_px = int(r * edge_margin)
#     inner_mask = plate_mask.copy()
#     kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_px * 2 + 1, rim_px * 2 + 1))
#     inner_mask = cv2.erode(inner_mask, kernel_erode, iterations=1)

#     # --- otsu threshold (computed only on plate pixels to avoid bias) ---
#     plate_px = gray_masked[inner_mask == 255]
#     thresh_val, _ = cv2.threshold(
#         plate_px.reshape(-1, 1).astype(np.uint8),
#         0, 255,
#         cv2.THRESH_BINARY + cv2.THRESH_OTSU,
#     )
    
#     thresh_val = thresh_val + 10

#     binary = np.zeros_like(gray_masked, dtype=np.uint8)
#     binary[(gray_masked > thresh_val) & (inner_mask == 255)] = 255

#     # --- morphological clean ---
#     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=2)
#     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=2)

#     cleaned = remove_small_objects(closed.astype(bool), max_size=min_colony_px, connectivity=2)
#     cleaned_u8 = cleaned.astype(np.uint8) * 255

#     # --- distance transform + local maxima as watershed seeds ---
#     dist = ndimage.distance_transform_edt(cleaned)
#     min_seed_gap = max(8, int(dist.max() * 0.15))

#     coords = peak_local_max(dist, min_distance=min_seed_gap, labels=cleaned)
#     seed_mask = np.zeros(dist.shape, dtype=bool)
#     seed_mask[tuple(coords.T)] = True

#     markers, _ = ndimage.label(seed_mask)
#     labels = watershed(-dist, markers, mask=cleaned)

#     return labels, cleaned_u8, dist



"""
colony_selector.py
------------------
Interactive colony deselection UI using OpenCV.

Drop-in replacement for the segment() function.  After the watershed step,
an OpenCV window opens showing the coloured segmentation overlay.  Click any
colony to toggle it off (greyed out); click again to restore it.  Press
ENTER or Q to confirm the selection and return cleaned labels.

Usage (standalone):
    from colony_selector import segment
    labels, binary, dist = segment(gray_masked, plate_mask, plate_info)
    # … rest of pipeline uses the returned labels normally
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

# ── colour palette (matches the rest of the pipeline's TAB20 style) ──────────
import matplotlib.pyplot as plt
_TAB20 = plt.get_cmap("tab20")

def _colony_colour(label_id: int) -> tuple:
    """Return a BGR uint8 colour for a given integer label."""
    r, g, b, _ = _TAB20((label_id % 20) / 20)
    return (int(b * 255), int(g * 255), int(r * 255))   # OpenCV is BGR


# ═════════════════════════════════════════════════════════════════════════════
# INTERACTIVE SELECTOR
# ═════════════════════════════════════════════════════════════════════════════

def interactive_colony_selector(img_rgb: np.ndarray,
                                 labels: np.ndarray) -> np.ndarray:
    """
    Opens an OpenCV window showing a coloured colony segmentation overlay.

    Controls
    --------
    Left-click   Toggle the clicked colony on / off.
    R            Reset – re-enable all colonies.
    ENTER / Q    Confirm selection and close the window.

    Parameters
    ----------
    img_rgb : H×W×3 uint8 RGB image of the plate.
    labels  : H×W int32 watershed label image (0 = background).

    Returns
    -------
    filtered_labels : H×W int32 array – deselected colonies are set to 0.
    """

    unique_labels = [u for u in np.unique(labels) if u != 0]
    active: set = set(unique_labels)           # all on by default

    # Pre-compute per-colony centroids for label overlays
    centroids: dict = {}
    for u in unique_labels:
        ys, xs = np.where(labels == u)
        centroids[u] = (int(xs.mean()), int(ys.mean()))

    # ── rendering helper ─────────────────────────────────────────────────────

    def render_frame() -> np.ndarray:
        base = img_rgb.astype(np.float32) / 255.0          # H×W×3 float RGB

        # coloured overlay
        for u in unique_labels:
            mask = labels == u
            if u in active:
                # vibrant colony colour
                c = np.array(_TAB20((u % 20) / 20)[:3], dtype=np.float32)
                base[mask] = 0.45 * base[mask] + 0.55 * c
            else:
                # greyed-out / struck-through
                grey = np.full(3, 0.45, dtype=np.float32)
                base[mask] = 0.65 * base[mask] + 0.35 * grey

        frame = np.clip(base * 255, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # draw centroid markers + ID labels
        for u in unique_labels:
            cx, cy = centroids[u]
            if u in active:
                col  = (255, 255, 255)
                thick = 1
            else:
                col  = (90, 90, 90)
                thick = 1
                # draw a red X through the centroid
                cv2.line(frame, (cx - 8, cy - 8), (cx + 8, cy + 8),
                         (50, 50, 200), 2)
                cv2.line(frame, (cx + 8, cy - 8), (cx - 8, cy + 8),
                         (50, 50, 200), 2)

            cv2.drawMarker(frame, (cx, cy), col,
                           cv2.MARKER_CROSS, 12, thick)

            # label ID background box
            text  = str(u)
            scale = 0.42
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                            scale, 1)
            tx, ty = cx + 7, cy - 6
            cv2.rectangle(frame,
                          (tx - 2, ty - th - 2), (tx + tw + 2, ty + bl),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1,
                        cv2.LINE_AA)

        # ── instruction bar at the bottom ────────────────────────────────────
        h, w = frame.shape[:2]
        bar_h = 36
        bar   = np.zeros((bar_h, w, 3), dtype=np.uint8)

        n_on  = len(active)
        n_off = len(unique_labels) - n_on
        info  = (f"  Colonies: {n_on} selected  |  {n_off} deselected   "
                 f"|  Left-click to toggle  |  R = reset  |  ENTER / Q = confirm")
        cv2.putText(bar, info, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
                    cv2.LINE_AA)

        return np.vstack([frame, bar])

    # ── mouse callback ────────────────────────────────────────────────────────

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # guard against clicks on the instruction bar
        if y >= labels.shape[0]:
            return
        lbl = int(labels[y, x])
        if lbl == 0:
            return
        if lbl in active:
            active.discard(lbl)
        else:
            active.add(lbl)
        cv2.imshow(_WIN, render_frame())

    # ── window setup ─────────────────────────────────────────────────────────

    _WIN = "Colony Selector  –  click to deselect  |  ENTER / Q to confirm"
    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

    # fit to screen: cap at 960 px on the long axis
    h, w = img_rgb.shape[:2]
    scale = min(960 / max(h, w), 1.0)
    cv2.resizeWindow(_WIN, int(w * scale), int((h + 36) * scale))

    cv2.setMouseCallback(_WIN, on_mouse)

    # ── main event loop ───────────────────────────────────────────────────────

    # drain stale events before the loop
    for _ in range(5):
        cv2.waitKey(1)
        
    while True:
        cv2.imshow(_WIN, render_frame())
        key = cv2.waitKey(30) & 0xFF
        if key in (13, ord('q'), ord('Q')):
            break
        elif key in (ord('r'), ord('R')):
            active = set(unique_labels)

    cv2.destroyAllWindows()
    cv2.waitKey(1)   # flush so next call starts clean

    # ── build filtered label image ────────────────────────────────────────────
    filtered = labels.copy()
    for u in unique_labels:
        if u not in active:
            filtered[filtered == u] = 0

    removed = sorted(set(unique_labels) - active)
    kept    = sorted(active)
    print(f"  [selector] kept    {len(kept)}  colonies: {kept}")
    if removed:
        print(f"  [selector] removed {len(removed)} colonies: {removed}")

    return filtered


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENT  (drop-in replacement)
# ═════════════════════════════════════════════════════════════════════════════

def segment(gray_masked, plate_mask, plate_info,
            img_rgb=None,
            min_colony_px=300, edge_margin=0.10,
            interactive=True):
    """
    Otsu threshold → morphological clean → distance transform → watershed.

    If ``interactive=True`` (default) and ``img_rgb`` is provided, an
    OpenCV window opens after the watershed so the user can click to
    deselect false-positive colonies before the function returns.

    Parameters
    ----------
    gray_masked   : preprocessed, plate-masked grayscale image.
    plate_mask    : binary uint8 mask of the plate interior.
    plate_info    : dict with keys cx, cy, r.
    img_rgb       : original colour image (H×W×3 uint8 RGB).
                    Required when interactive=True; ignored otherwise.
    min_colony_px : minimum colony area in pixels (removes noise blobs).
    edge_margin   : fraction of the plate radius to exclude at the rim.
    interactive   : if True open the deselection UI before returning.

    Returns
    -------
    labels     : H×W int32 – final, user-curated watershed label image.
    cleaned_u8 : binary mask after morphological cleaning (uint8 0/255).
    dist       : Euclidean distance transform of the cleaned mask.
    """

    cx, cy, r = plate_info["cx"], plate_info["cy"], plate_info["r"]

    # ── erode rim so the metal edge never seeds a colony ─────────────────────
    rim_px = int(r * edge_margin)
    inner_mask = plate_mask.copy()
    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (rim_px * 2 + 1, rim_px * 2 + 1))
    inner_mask = cv2.erode(inner_mask, kernel_erode, iterations=1)

    # ── Otsu threshold (computed only over plate pixels) ─────────────────────
    plate_px = gray_masked[inner_mask == 255]
    thresh_val, _ = cv2.threshold(
        plate_px.reshape(-1, 1).astype(np.uint8),
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    thresh_val = thresh_val 
    # + 10          # small positive bias keeps agar clean

    binary = np.zeros_like(gray_masked, dtype=np.uint8)
    binary[(gray_masked > thresh_val) & (inner_mask == 255)] = 255

    # ── morphological clean ───────────────────────────────────────────────────
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=2)

    cleaned    = remove_small_objects(closed.astype(bool),
                                      max_size=min_colony_px, connectivity=2)
    cleaned_u8 = cleaned.astype(np.uint8) * 255

    # ── distance transform + watershed ───────────────────────────────────────
    dist         = ndimage.distance_transform_edt(cleaned)
    min_seed_gap = max(8, int(dist.max() * 0.15))

    coords    = peak_local_max(dist, min_distance=min_seed_gap, labels=cleaned)
    seed_mask = np.zeros(dist.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True

    markers, _ = ndimage.label(seed_mask)
    labels      = watershed(-dist, markers, mask=cleaned)

    # ── interactive curation ──────────────────────────────────────────────────
    
    if interactive:
        if img_rgb is None:
            print("  [selector] Warning: img_rgb not supplied – "
                  "skipping interactive curation.")
        else:
            print("  [selector] Opening interactive colony selector …")
            print("             Left-click to toggle | R = reset | "
                  "ENTER / Q = confirm")
            labels = interactive_colony_selector(img_rgb, labels)

    return labels, cleaned_u8, dist