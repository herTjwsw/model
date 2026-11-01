"""Download public thumbnail images for the official Game Theorists videos."""
import concurrent.futures, json, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from PIL import Image, ImageFile
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = False
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def fetch(video):
    vid = video["id"]
    for size in ("maxresdefault", "sddefault", "hqdefault"):
        url = f"https://i.ytimg.com/vi/{vid}/{size}.jpg"
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; thumbnail-dataset/1.0)"})
            with urlopen(req, timeout=20) as r:
                data = r.read()
            im = Image.open(BytesIO(data)).convert("RGB")
            w, h = im.size
            # Reject YouTube's generic unavailable-thumbnail placeholder (120x90).
            if w < 300 or h < 160 or w / h < 1.25:
                continue
            dest = RAW / f"{vid}.jpg"
            im.save(dest, "JPEG", quality=92, optimize=True)
            return {**video, "file": str(dest.relative_to(ROOT)), "width": w, "height": h, "source": size}
        except (HTTPError, URLError, OSError, ValueError):
            continue
    return None


def main():
    videos = json.loads((ROOT / "videos.json").read_text())
    ok, failures = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(fetch, videos), 1):
            if result: ok.append(result)
            else: failures += 1
            if i % 40 == 0 or i == len(videos):
                print(f"{i}/{len(videos)} checked; {len(ok)} usable thumbnails; {failures} unavailable", flush=True)
    (ROOT / "data" / "manifest.json").write_text(json.dumps(ok, indent=2))
    print(f"Saved {len(ok)} thumbnail images in {RAW}")

if __name__ == "__main__": main()
