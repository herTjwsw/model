"""Make 64x64 RGB training images from the downloaded thumbnail originals."""
import json
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
count = 0
for item in manifest:
    src = ROOT / item["file"]
    try:
        im = Image.open(src).convert("RGB")
        # Square center crop preserves a coherent central subject at 64px.
        im = ImageOps.fit(im, (64, 64), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        im.save(OUT / f"{item['id']}.jpg", quality=94)
        count += 1
    except (OSError, KeyError):
        pass
print(f"Prepared {count} RGB images (64x64) in {OUT}")
