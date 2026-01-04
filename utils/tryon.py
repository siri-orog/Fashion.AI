import cv2
import numpy as np

# Use Haar cascade included with OpenCV
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_largest_face(bgr_image, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
    """
    Returns (x,y,w,h) of the largest detected face in pixel coords or None.
    Input image expected in BGR (OpenCV) format.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    if len(faces) == 0:
        return None
    # choose largest face by area
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    return faces[0]

def extract_face_region(bgr_image, face_bbox, margin=0.35):
    """
    Extract face region with a margin (fraction of width/height).
    Returns RGBA face patch (with alpha mask) and its original bbox coords.
    """
    x, y, w, h = face_bbox
    mw = int(w * margin)
    mh = int(h * margin)
    x1 = max(0, x - mw)
    y1 = max(0, y - mh)
    x2 = min(bgr_image.shape[1], x + w + mw)
    y2 = min(bgr_image.shape[0], y + h + mh)

    patch = bgr_image[y1:y2, x1:x2].copy()
    # create alpha mask from skin-like segmentation by converting to YCrCb and thresholding Cr channel
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    Cr = ycrcb[:, :, 1]
    # adaptive threshold to separate skin-ish area (works reasonably well for frontal faces)
    _, mask = cv2.threshold(Cr, 140, 255, cv2.THRESH_BINARY)
    # refine mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    # create 4-channel RGBA patch
    b, g, r = cv2.split(patch)
    alpha = (mask / 255.0).astype(np.float32)
    alpha3 = np.stack([alpha, alpha, alpha], axis=2)
    rgba = np.dstack([b, g, r, (alpha*255).astype(np.uint8)])
    return rgba, (x1, y1, x2, y2)

def blend_face_onto_model(model_bgr, face_rgba, target_bbox, align_center=True):
    """
    Blend face_rgba onto model_bgr at target_bbox (x,y,w,h).
    face_rgba is RGBA patch. target_bbox is (x,y,w,h) in model coordinates.
    Returns blended BGR image.
    """
    out = model_bgr.copy()
    tx, ty, tw, th = target_bbox
    # Resize face to fit target bbox while keeping aspect ratio
    fh, fw = face_rgba.shape[:2]
    scale_w = tw / fw
    scale_h = th / fh
    scale = min(scale_w, scale_h)
    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))
    face_resized = cv2.resize(face_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # compute paste position (centered)
    if align_center:
        px = tx + (tw - new_w)//2
        py = ty + (th - new_h)//2
    else:
        px = tx
        py = ty

    # clip to image bounds
    px1 = max(0, px); py1 = max(0, py)
    px2 = min(out.shape[1], px + new_w); py2 = min(out.shape[0], py + new_h)
    sx1 = px1 - px; sy1 = py1 - py
    sx2 = sx1 + (px2 - px1); sy2 = sy1 + (py2 - py1)

    if px1 >= px2 or py1 >= py2:
        return out  # nothing to paste

    overlay_region = face_resized[sy1:sy2, sx1:sx2]
    alpha = overlay_region[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay_region[:, :, :3].astype(np.float32)
    bg_region = out[py1:py2, px1:px2].astype(np.float32)

    blended = alpha * overlay_rgb + (1 - alpha) * bg_region
    out[py1:py2, px1:px2] = blended.astype(np.uint8)
    return out

def try_on_user_face(user_bgr, model_bgr):
    """
    High-level helper:
    - detect face in user image and model image,
    - extract user face region with alpha,
    - blend onto model face bbox,
    - return resulting BGR image or None if detection fails.
    """
    user_face_bbox = detect_largest_face(user_bgr)
    if user_face_bbox is None:
        return None, "No face found in user image."

    model_face_bbox = detect_largest_face(model_bgr)
    if model_face_bbox is None:
        return None, "No face found in model image to replace."

    face_rgba, _coords = extract_face_region(user_bgr, user_face_bbox)
    # use model face bbox for target placement
    mx, my, mw, mh = model_face_bbox
    result = blend_face_onto_model(model_bgr, face_rgba, (mx, my, mw, mh))
    return result, None
