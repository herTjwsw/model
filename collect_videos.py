"""Collect videos from MatPat's Game Theory YouTube channel via YouTube search.

Only results attributed to the official `The Game Theorists` channel are kept.
This gathers public video IDs and titles; the next step downloads their public
thumbnail images, not the videos themselves.
"""
import json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUERIES = [
    "The Game Theorists MatPat Game Theory",
    "The Game Theorists FNAF theory",
    "The Game Theorists Minecraft theory",
    "The Game Theorists Pokemon theory",
    "The Game Theorists Mario theory",
    "The Game Theorists indie horror theory",
    "The Game Theorists game theory lore",
    "The Game Theorists MatPat theory",
    "The Game Theorists newest videos",
    "The Game Theorists classic theories",
]
N_PER_QUERY = 50


def main():
    seen = {}
    for query in QUERIES:
        cmd = ["yt-dlp", "--flat-playlist", "--dump-json", "--no-warnings", f"ytsearch{N_PER_QUERY}:{query}"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            print(f"timeout: {query}"); continue
        n = 0
        for line in p.stdout.splitlines():
            try: d = json.loads(line)
            except json.JSONDecodeError: continue
            channel = d.get("channel") or d.get("uploader") or ""
            # The MatPat-associated main channel has this exact public channel name.
            if channel.strip().casefold() != "the game theorists":
                continue
            vid = d.get("id")
            if not vid or vid in seen: continue
            seen[vid] = {"id": vid, "title": (d.get("title") or "")[:200], "channel": channel,
                         "uploader_id": d.get("uploader_id") or d.get("channel_id") or "", "query": query}
            n += 1
        print(f"{query}: +{n} (total {len(seen)})", flush=True)
    videos = list(seen.values())
    (HERE / "videos.json").write_text(json.dumps(videos, indent=2))
    print(f"Saved {len(videos)} official Game Theorists videos to videos.json")

if __name__ == "__main__": main()
