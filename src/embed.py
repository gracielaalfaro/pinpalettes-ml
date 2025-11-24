import torch
import open_clip
import numpy as np
from tqdm import tqdm
from PIL import Image

def load_clip(model_name="ViT-B-32", pretrained="openai", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device)
    model.eval()
    return model, preprocess, device

@torch.no_grad()
def embed_images(image_paths, model, preprocess, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        imgs = torch.stack(imgs).to(device)
        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
    return np.vstack(embeddings)
