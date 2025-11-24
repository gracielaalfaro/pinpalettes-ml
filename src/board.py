from PIL import Image
import math

def make_moodboard(image_paths, out_path, tile_size=256, cols=4):
    imgs = [Image.open(p).convert("RGB").resize((tile_size, tile_size)) for p in image_paths]
    rows = math.ceil(len(imgs) / cols)

    board = Image.new("RGB", (cols*tile_size, rows*tile_size), (255,255,255))
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        board.paste(img, (c*tile_size, r*tile_size))

    board.save(out_path)
    return out_path
