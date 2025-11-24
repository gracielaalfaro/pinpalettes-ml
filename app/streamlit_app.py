import os
import sys
import time
import numpy as np
import streamlit as st
from PIL import Image

# --- Force repo root onto path (absolute, based on THIS file) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------------------------------------------

from src.utils import list_images, load_image
from src.palette import extract_palette, palette_stats
from src.embed import load_clip, embed_images
from src.recommend import CLIPRecommender, rerank_by_palette
from src.board import make_moodboard
import os
import sys
import time
import numpy as np
import streamlit as st
from PIL import Image

# Make repo root importable
sys.path.insert(0, os.path.abspath(".."))
st.set_page_config(page_title="PinPalettes-ML", layout="wide")

st.title("🌸 pinpalettes-ml")
st.caption("Upload an inspiration image → get palette, vibe stats, similar images, and a moodboard.")

SAMPLE_DIR = "../data/sample"
PROC_DIR = "../data/processed"
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs("../results/moodboards", exist_ok=True)

EMB_PATH = os.path.join(PROC_DIR, "embeddings.npy")
PATHS_PATH = os.path.join(PROC_DIR, "image_paths.npy")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Settings")
k_colors = st.sidebar.slider("Palette colors (k)", 3, 10, 6)
top_k = st.sidebar.slider("Recommendations (top-k)", 3, 20, 8)
use_palette_rerank = st.sidebar.checkbox("Palette-aware re-ranking", value=True)
cols_board = st.sidebar.slider("Moodboard columns", 2, 5, 3)
tile_size = st.sidebar.slider("Moodboard tile size", 128, 384, 256)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Sample Dataset")
st.sidebar.write(f"Folder: `{SAMPLE_DIR}`")
sample_imgs = list_images(SAMPLE_DIR)
st.sidebar.write(f"Images found: **{len(sample_imgs)}**")

rebuild_emb = st.sidebar.button("Rebuild embeddings")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource
def get_clip():
    model, preprocess, device = load_clip("ViT-B-32", "openai")
    return model, preprocess, device

def load_or_build_embeddings(image_paths, force_rebuild=False):
    if (not force_rebuild) and os.path.exists(EMB_PATH) and os.path.exists(PATHS_PATH):
        cached_paths = np.load(PATHS_PATH, allow_pickle=True).tolist()
        if cached_paths == image_paths:
            emb = np.load(EMB_PATH)
            return emb, cached_paths

    model, preprocess, device = get_clip()
    with st.spinner("Embedding sample images with CLIP... (first time may take a bit)"):
        emb = embed_images(image_paths, model, preprocess, device)
    np.save(EMB_PATH, emb)
    np.save(PATHS_PATH, np.array(image_paths, dtype=object))
    return emb, image_paths

# ---------------------------
# Main: upload query image
# ---------------------------
uploaded = st.file_uploader("Upload an inspiration image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

query_img = Image.open(uploaded).convert("RGB")

# Show query
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("🖼️ Query Image")
    st.image(query_img, use_container_width=True)

# Palette + stats
rgb, hexs, props = extract_palette(query_img, k=k_colors)
stats = palette_stats(query_img)

with c2:
    st.subheader("🎨 Palette")
    palette_bar = np.zeros((50, 400, 3), dtype=np.uint8)
    x = 0
    for color, p in zip(rgb, props):
        w = int(400 * p)
        palette_bar[:, x:x+w, :] = color
        x += w
    st.image(palette_bar, use_container_width=True)
    st.write("**HEX colors:**", ", ".join(hexs))

    st.subheader("💫 Vibe Metrics")
    st.json(stats)

st.markdown("---")

# ---------------------------
# Build embeddings + recommender
# ---------------------------
if len(sample_imgs) < 3:
    st.warning("Add at least 3 images to data/sample/ for recommendations.")
    st.stop()

emb, image_paths = load_or_build_embeddings(sample_imgs, force_rebuild=rebuild_emb)
rec = CLIPRecommender(embeddings=emb, image_paths=image_paths)

# Embed query (single image)
model, preprocess, device = get_clip()
q_emb = embed_images([uploaded], model, preprocess, device)[0]

# Retrieve neighbors
neighbors, scores = rec.query(q_emb, top_k=min(top_k, len(image_paths)))

if use_palette_rerank:
    neighbors = rerank_by_palette(query_img, neighbors, top_k=min(top_k, len(neighbors)))

# ---------------------------
# Display recommendations
# ---------------------------
st.subheader("🔍 Similar Images (CLIP)")
cols = st.columns(len(neighbors))
for col, p in zip(cols, neighbors):
    with col:
        st.image(Image.open(p).convert("RGB"), use_container_width=True)
        st.caption(os.path.basename(p))

st.markdown("---")

# ---------------------------
# Moodboard
# ---------------------------
st.subheader("🧩 Moodboard")
out_path = f"../results/moodboards/moodboard_{int(time.time())}.png"
make_moodboard(neighbors, out_path, tile_size=tile_size, cols=cols_board)

board_img = Image.open(out_path)
st.image(board_img, use_container_width=True)
st.caption(f"Saved to `{out_path}`")

