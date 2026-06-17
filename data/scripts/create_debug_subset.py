"""
Create a 500-image debug subset from the VisDrone training set.

Samples 500 random images (seed 42), copies images and YOLO+ annotations
to data/subsets/debug_500/ for fast pipeline validation.

Usage:
    conda activate visdrone
    python data/scripts/create_debug_subset.py
"""

import sys, shutil, random
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.env import get_paths

N_SAMPLES = 500
SEED = 42


def main():
    paths = get_paths()
    data_dir = paths["data"]                     # .../VisDrone2019-DET/
    subset_dir = paths["subsets"] / "debug_500"  # .../subsets/debug_500/

    img_src = data_dir / "images" / "train"
    ann_src = data_dir / "annotations" / "train"
    img_dst = subset_dir / "images"
    ann_dst = subset_dir / "annotations"

    # Collect all train images
    all_images = sorted(img_src.glob("*.jpg"))
    print(f"Train images available: {len(all_images)}")

    # Sample
    rng = random.Random(SEED)
    sampled = rng.sample(all_images, min(N_SAMPLES, len(all_images)))
    print(f"Sampled: {len(sampled)} images (seed={SEED})")

    # Create output dirs
    img_dst.mkdir(parents=True, exist_ok=True)
    ann_dst.mkdir(parents=True, exist_ok=True)

    # Copy
    for img_path in sampled:
        # Image
        shutil.copy2(img_path, img_dst / img_path.name)
        # Corresponding annotation (.txt same stem)
        ann_path = ann_src / img_path.with_suffix(".txt").name
        if ann_path.exists():
            shutil.copy2(ann_path, ann_dst / ann_path.name)

    n_imgs = len(list(img_dst.glob("*")))
    n_anns = len(list(ann_dst.glob("*.txt")))
    print(f"[OK] Debug subset created:")
    print(f"      images:      {img_dst} ({n_imgs} files)")
    print(f"      annotations: {ann_dst} ({n_anns} files)")


if __name__ == "__main__":
    main()
