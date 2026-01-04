import cv2
import numpy as np

def overlay_image_alpha(background, overlay, x, y, overlay_size=None):
    """
    Overlay `overlay` onto `background` at (x, y) with alpha channel handling.
    overlay: RGBA image (uint8)
    background: BGR image (uint8)
    overlay_size: (w, h) or None to use overlay's size
    """
    bg = background
    ol = overlay

    if overlay_size is not None:
        ol = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

    h, w = ol.shape[:2]
    rows, cols = bg.shape[:2]

    # boundaries check
    if x >= cols or y >= rows or x + w <= 0 or y + h <= 0:
        return bg

    # clip overlay region to background
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, cols)
    y2 = min(y + h, rows)

    ol_x1 = x1 - x
    ol_y1 = y1 - y
    ol_x2 = ol_x1 + (x2 - x1)
    ol_y2 = ol_y1 + (y2 - y1)

    # Extract alpha and color channels
    alpha = ol[ol_y1:ol_y2, ol_x1:ol_x2, 3] / 255.0
    alpha = alpha[..., np.newaxis]
    overlay_rgb = ol[ol_y1:ol_y2, ol_x1:ol_x2, :3]

    bg_region = bg[y1:y2, x1:x2]

    # Blend
    blended = (alpha * overlay_rgb + (1 - alpha) * bg_region).astype(np.uint8)
    bg[y1:y2, x1:x2] = blended

    return bg
