# 🌸 **PinPalettes ML: Aesthetic Palette Extraction & Visual Recommender**
*Pinterest-inspired ML pipeline for extracting color palettes, aesthetic metrics, and visually similar recommendations from images.*

---

## 🌟 Overview

**pinpalettes-ml** is a visual machine learning project that transforms any image into:

1. 🎨 **A dominant color palette** (HEX/RGB)  
2. 💫 **Aesthetic “vibe metrics”** (warmth, brightness, saturation, contrast)  
3. 🔍 **Visually similar recommendations** using CLIP embeddings  
4. 🖼️ **Auto-generated Pinterest-style moodboards**  

This repo combines creative visual aesthetics with real ML techniques like:
- k-means clustering  
- CLIP vision-language embeddings  
- nearest-neighbor search  
- image processing & collage generation  

The result is a full, reproducible, production-style ML pipeline for aesthetic image analysis and recommendation.

---

## 🚀 Features

### 🎨 **1. Aesthetic Palette Extraction**
- Extract dominant colors using k-means  
- Output HEX + RGB values  
- Compute mood stats:
  - brightness  
  - contrast  
  - saturation  
  - warmth  

### 💡 **2. CLIP Image Embeddings**
- Encode images into semantic embedding space  
- Supports batch embedding + caching  
- Works with any aesthetic dataset  

### 🔎 **3. Visual Recommender System**
- CLIP-based nearest neighbors  
- Optional palette-aware re-ranking  
- Returns top-K visually similar images  

### 🖼️ **4. Moodboard Generator**
- Automatically creates Pinterest-style collages  
- Saves to `results/moodboards/`  
- Perfect for visual storytelling  

---

## 📘 Usage
- Run Palette Extraction
- Generate CLIP Embeddings
- Visual Recommender Demo
- Generate Moodboards

---

## 🔒 Data Policy
To keep the repo lightweight, ethical, and compliant:
- Large raw datasets belong in data/raw/ and are gitignored
- Only sample images are included
- All embeddings can be regenerated locally

  
---
## 📜 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with attribution.

---
## 💗 Credits
Inspired by:
- Pinterest visual culture
- Aesthetic color theory

  
<br> Created by Graciela Alfaro <3

