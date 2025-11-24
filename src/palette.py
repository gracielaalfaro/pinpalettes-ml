import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def extract_palette(img: Image.Image, k=6, resize=256, random_state=42):
    """
    Extract dominant colors via k-means.
    Returns: (colors_rgb, hex_colors, proportions)
    """
    img_small = img.resize((resize, resize))
    arr = np.asarray(img_small) / 255.0
    pixels = arr.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels)
    proportions = counts / counts.sum()

    order = np.argsort(proportions)[::-1]
    centers = centers[order]
    proportions = proportions[order]

    colors_rgb = (centers * 255).astype(int)
    hex_colors = [rgb_to_hex(c) for c in colors_rgb]
    return colors_rgb, hex_colors, proportions

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def palette_stats(img: Image.Image):
    """
    Simple vibe stats: brightness, saturation, contrast, warmth.
    """
    arr = np.asarray(img).astype(float) / 255.0
    r, g, b = arr[...,0], arr[...,1], arr[...,2]

    brightness = arr.mean()
    contrast = arr.std()

    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    saturation = np.mean((maxc - minc) / (maxc + 1e-8))
    warmth = np.mean(r - b)

    return {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "warmth": float(warmth),
    }

