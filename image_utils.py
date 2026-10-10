from pathlib import Path
from PIL import Image
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
def list_images(folder):
    return sorted(p for p in Path(folder).rglob('*') if p.suffix.lower() in EXTS)
def load_rgb(path): return Image.open(path).convert('RGB')
