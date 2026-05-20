"""
Dataset Preparation Script
===========================
Downloads and sets up the PlantVillage dataset (54,306 images, 38 classes).
Also supports custom datasets from your own leaf images.

Run this BEFORE plant_disease_model.py
"""

import os
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────
#  AUTO-DOWNLOAD  (Kaggle API or manual)
# ─────────────────────────────────────────────
KAGGLE_DATASET  = "emmarex/plantdisease"      # 54K images, 38 classes
DATA_DIR        = "New Plant Diseases Dataset(Augmented)/train"
RAW_DIR         = "data/raw"


def check_kaggle():
    try:
        import kaggle
        return True
    except ImportError:
        return False


def download_plantvillage():
    """Download PlantVillage via Kaggle API."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if not check_kaggle():
        print("[Setup] Installing kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

    # Requires ~/.kaggle/kaggle.json with your API key
    # Get it at: https://www.kaggle.com/settings (API section)
    print(f"[Dataset] Downloading '{KAGGLE_DATASET}' from Kaggle...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip")

    # Locate PlantVillage folder
    possible = list(Path(RAW_DIR).rglob("PlantVillage"))
    if possible:
        src = str(possible[0])
        shutil.move(src, DATA_DIR)
        print(f"[Dataset] Moved dataset to '{DATA_DIR}'")
    else:
        print(f"[Dataset] Dataset downloaded to '{RAW_DIR}'. "
              f"Please move the class folders to '{DATA_DIR}/'.")


# ─────────────────────────────────────────────
#  CUSTOM DATASET BUILDER
# ─────────────────────────────────────────────
def create_custom_dataset(source_dir: str, dest_dir: str = DATA_DIR,
                          min_images: int = 50):
    """
    Organise your own leaf photos into the required folder structure.

    Expected input  (any structure):
        source_dir/
            img001.jpg
            img002.jpg  ...

    Required output:
        dest_dir/
            ClassName1/
                img001.jpg ...
            ClassName2/
                ...

    Args:
        source_dir : directory containing your raw images
        dest_dir   : destination (dataset root)
        min_images : minimum images per class (warns if below)
    """
    src = Path(source_dir)
    dst = Path(dest_dir)

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    classes = {}

    for f in src.rglob("*"):
        if f.suffix.lower() in exts:
            class_name = f.parent.name   # use parent folder as class label
            classes.setdefault(class_name, []).append(f)

    print(f"[Custom Dataset] Found {len(classes)} classes in '{source_dir}'")

    for cls, files in classes.items():
        cls_dir = dst / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, cls_dir / f.name)
        if len(files) < min_images:
            print(f"  ⚠  Class '{cls}' has only {len(files)} images "
                  f"(recommend ≥{min_images})")
        else:
            print(f"  ✓  {cls}: {len(files)} images")

    print(f"\n[Custom Dataset] Dataset ready at '{dest_dir}'")


# ─────────────────────────────────────────────
#  DATASET STATISTICS
# ─────────────────────────────────────────────
def dataset_stats(data_dir: str = DATA_DIR):
    """Print per-class image counts and flag imbalance."""
    from collections import Counter
    root = Path(data_dir)
    if not root.exists():
        print(f"[Stats] '{data_dir}' not found. Run download first.")
        return

    exts   = {".jpg", ".jpeg", ".png"}
    counts = Counter()
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in exts)
            counts[cls_dir.name] = n

    total = sum(counts.values())
    print(f"\n{'-'*55}")
    print(f"  Dataset Statistics  |  {data_dir}")
    print(f"{'-'*55}")
    print(f"  Classes : {len(counts)}")
    print(f"  Total   : {total:,} images")
    print(f"  Min     : {min(counts.values())} ({min(counts, key=counts.get)})")
    print(f"  Max     : {max(counts.values())} ({max(counts, key=counts.get)})")
    print(f"  Mean    : {total // len(counts)} images/class\n")

    for cls, n in sorted(counts.items(), key=lambda x: x[1]):
        bar  = "#" * (n // 1000)
        flag = "  ← low" if n < 200 else ""
        print(f"  {cls:<45} {n:5d}  {bar}{flag}")


# ─────────────────────────────────────────────
#  38 PlantVillage Classes (reference)
# ─────────────────────────────────────────────
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___healthy", "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plant Disease Dataset Setup")
    parser.add_argument("--download",    action="store_true",
                        help="Download PlantVillage from Kaggle")
    parser.add_argument("--custom",      type=str, default=None,
                        help="Path to custom leaf image directory")
    parser.add_argument("--stats",       action="store_true",
                        help="Print dataset statistics")
    args = parser.parse_args()

    if args.download:
        download_plantvillage()
    if args.custom:
        create_custom_dataset(args.custom)
    if args.stats or not any([args.download, args.custom]):
        dataset_stats()
