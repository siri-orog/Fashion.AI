import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_color(image, k=3):
    """Extract dominant color using KMeans."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    dominant_color = colors[np.argmax(counts)]
    return tuple(map(int, dominant_color))

def extract_palette(image, n_colors=5):
    """Extract color palette using KMeans (replacement for ColorThief)."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(img)
    palette = [tuple(map(int, color)) for color in kmeans.cluster_centers_]
    return palette

