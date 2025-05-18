# label_crops_with_clip.py
import os, glob, shutil, torch, clip
from PIL import Image
from tqdm import tqdm

CROP_DIR = "data/dice_crops"  # input crops from previous step
OUT_ROOT = "data/auto_labels"  # will create subdirs 1/ 2/ 3/ 4/ 5/ worm/
BATCHSIZE = 256  # CPU batch; tweak to fit RAM
MODEL_NAME = "ViT-B/32"  # fastest CPU CLIP
DEVICE = "cpu"

classes = ["1", "2", "3", "4", "5", "worm"]
prompts = [f"a photo of a die face showing {c} blue dots" for c in classes[:-1]] + [
    "a photo of a die face showing a red worm"]

# 1. Load CLIP
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
text_tokens = clip.tokenize(prompts).to(DEVICE)
with torch.no_grad():
    text_feats = model.encode_text(text_tokens)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)

# 2. Create output folders
for c in classes:
    os.makedirs(os.path.join(OUT_ROOT, c), exist_ok=True)

# 3. Batch over PNG/JPG crops
crop_paths = glob.glob(os.path.join(CROP_DIR, "*.*"))
for i in tqdm(range(0, len(crop_paths), BATCHSIZE)):
    batch_paths = crop_paths[i: i + BATCHSIZE]
    imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(DEVICE)

    with torch.no_grad():
        img_feats = model.encode_image(imgs)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        sims = img_feats @ text_feats.T  # [B, 6]
        preds = sims.argmax(dim=1).cpu().tolist()

    # move/copy each crop into predicted class dir
    for src, cls_idx in zip(batch_paths, preds):
        shutil.copy(src, os.path.join(OUT_ROOT, classes[cls_idx], os.path.basename(src)))
