import os
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def list_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f.lower())[1] in IMG_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size)
    return img
