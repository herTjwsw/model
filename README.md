# Theory Thumbnail Lab — MatPat / The Game Theorists DCGAN

A runnable, small **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on public video-thumbnail images from MatPat’s **The Game Theorists** YouTube channel. This project intentionally focuses on that channel—not other “theory” creators.

## What’s included

- `videos.json`: 272 unique videos returned by YouTube search and filtered to the exact channel name **The Game Theorists**.
- `data/raw/`: downloaded public thumbnail previews; `data/manifest.json` records video ID, title, dimensions, and which YouTube thumbnail size was used.
- `data/processed/`: 64×64 RGB training images.
- `model.py`, `train.py`: CPU-friendly PyTorch DCGAN and training loop.
- `checkpoints/latest.pt`: trained generator/discriminator checkpoint (epoch 500).
- `samples/epoch_0500.jpg`: a 4×4 batch produced by the saved generator.
- `app.py` and `index.html`: self-contained local gallery and live seed/interpolation demo with **Game Theory** and **My Character** modes. It uses only same-origin/relative URLs and inline CSS/JS.
- `data/avatar_source/`: the six screenshots you supplied; `checkpoints/avatar_latest.pt` is a separate character-mode DCGAN trained for 400 epochs, with sample grid at `samples/avatar_epoch_0400.jpg`.
- `train_avatar.py`: script for retraining the small-data character model.

## Run the web demo

From this folder, with Python 3, PyTorch, Pillow, and NumPy installed:

```bash
python3 app.py
```

Open `http://localhost:8000`. The app also works behind a proxied preview URL. Click **Generate** for a fresh pair of latent seeds; move the slider to interpolate between them. The source gallery displays real dataset thumbnails, while the generated output is labeled as synthetic.

## Rebuild the dataset

`yt-dlp` is needed only for collection. Searches are used because YouTube’s channel-page extraction can be unreliable in some environments. Search results are retained only when the uploader/channel name exactly matches `The Game Theorists`.

```bash
python3 -m pip install yt-dlp pillow numpy
python3 collect_videos.py
python3 download_thumbs.py
python3 prepare.py
```

Only public thumbnail image previews are downloaded; videos are not downloaded. YouTube may change search results, so a future collection may not yield the same 272 IDs.

## Train or continue training

This project was trained on CPU for **500 epochs** with batch size 16 on 272 64×64 images. On a GPU-free machine, the compact model is intentionally small.

```bash
# Start a new model; --epochs is the number of epochs to run
python3 train.py --epochs 100 --batch-size 16

# Continue from checkpoints/latest.pt for 100 more epochs
python3 train.py --epochs 100 --batch-size 16 --resume
```

Training saves `checkpoints/latest.pt`, periodic epoch checkpoints, and sample sheets under `samples/`. The web app loads the latest checkpoint on startup, so restart it to pick up a newer model.

## Character mode

The app's **My Character** toggle selects a separate DCGAN trained from your six uploaded Roblox-avatar screenshots. It ran for 400 epochs with light image augmentation. To retrain after replacing or adding reference screenshots in `data/avatar_source/`:

```bash
python3 train_avatar.py --epochs 400 --batch-size 8
```

With only six source pictures, the avatar model is especially experimental and its generated samples are abstract/artifact-prone; it does not reliably recreate the character. More clean, varied screenshots would help substantially.

## Important limitations and attribution

This is an experimental **64×64** GAN trained from scratch on a small CPU model and a modest dataset. It learns some broad color, composition, and text-like texture patterns, but outputs are abstract and artifact-prone—not polished, reliable, or suitable as finished thumbnails. Increasing resolution, model capacity, and training resources would be needed for higher-quality results. A GAN learns visual patterns; it does not retrieve or reproduce a specific source thumbnail intentionally.

The collected thumbnails belong to their respective rights holders and are included here for a local machine-learning demonstration. **The Game Theorists / MatPat are not affiliated with this project.** Respect YouTube’s terms, creator rights, and applicable copyright rules before sharing or reusing any source or generated image.
