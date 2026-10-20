"""
Merge 3 ASL image datasets into one folder ("Mega Dataset") with one
subfolder per letter A–Y (excluding J and Z, which require motion).

Datasets used:
    1. American Sign Language (ASL) Dataset    -> prefix "asl-ds"      (107,719 images)
    2. ASL-HG American Sign Language Hand Gesture Image Dataset
                                                -> prefix "asl-hg"      (24,000 images)
    3. Sign Language MNIST ORIGINAL             -> prefix "mnist-orig"  (1,680 images)
    Total: 133,399 images

- Files are COPIED, so the original datasets stay untouched.
- Each file is renamed "<source>_<original name>" to avoid name collisions.
- Digits, J and Z classes are skipped.
- Empty (0-byte) files are skipped.
- Safe to re-run: files that already exist in the output are skipped.

Datasets dropped after the first version (image differences):
    American Sign Language, ASL Fingerspelling Images (RGB & Depth),
    Sign Language Gesture Images.

Known broken image (from a dropped dataset, for reference):
    Dataset:     American Sign Language
    Source file: American Sign Language/Train/V/V882.jpg  (0 bytes, empty)
    Merged copy: Mega Dataset/V/asl_train_V882.jpg  (deleted)
"""

import os
import re
import shutil
import time

start_time = time.time()

ROOT = os.path.expanduser("data")
OUT = os.path.join(ROOT, "dataset" ,"D01_megadataset")

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # A–Y without J and Z
IMG_EXT = (".jpg", ".jpeg", ".png")

counts = {}

os.makedirs(os.path.join(ROOT, "dataset"), exist_ok=True)

def put(src, letter, prefix):
    """Copy one image into OUT/<LETTER>/<prefix>_<filename>."""
    letter = letter.upper()
    if letter not in LETTERS:
        return
    # Skip empty/corrupt files
    if os.path.getsize(src) == 0:
        return
    name = f"{prefix}_{os.path.basename(src).replace(' ', '_')}"
    dst = os.path.join(OUT, letter, name)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    counts[prefix] = counts.get(prefix, 0) + 1


def copy_letter_dirs(base, prefix):
    """Copy images from a folder whose subfolders are single letters (A, b, ...)."""
    for d in sorted(os.listdir(base)):
        path = os.path.join(base, d)
        if os.path.isdir(path) and len(d) == 1:
            for f in os.listdir(path):
                if f.lower().endswith(IMG_EXT):
                    put(os.path.join(path, f), d, prefix+"_")


def main():
    for letter in LETTERS:
        os.makedirs(os.path.join(OUT, letter), exist_ok=True)

    # 1. American Sign Language (ASL) Dataset (lowercase letter folders)
    raw_dataset_label = "A03_kaggle_asl"
    print(f"Adding {raw_dataset_label}...")
    copy_letter_dirs(
        base=os.path.join(ROOT, "raw_data", raw_dataset_label, "American"),
        prefix=raw_dataset_label
    )
    print("Done!")
    print()

    # 2. ASL-HG
    raw_dataset_label = "A04_mandalay_aslhg"
    print(f"Adding {raw_dataset_label}...")
    copy_letter_dirs(
        base=os.path.join(ROOT, "raw_data", raw_dataset_label, "ASL_HG_36000", "asl_dataset"),
        prefix=raw_dataset_label,
    )
    print("Done!")
    print()

    # 3. Sign Language MNIST ORIGINAL: user_N/A0.jpg, B3.jpg, ... (letter = first char)
    raw_dataset_label = "B01_original_aslmnist"
    print(f"Adding {raw_dataset_label}...")
    mnist = os.path.join(ROOT, "raw_data", raw_dataset_label, "Dataset")
    for user in sorted(os.listdir(mnist)):
        path = os.path.join(mnist, user)
        if os.path.isdir(path):
            for f in os.listdir(path):
                m = re.match(r"([A-Za-z])\d+", f)
                if m and f.lower().endswith(IMG_EXT):
                    put(os.path.join(path, f), m.group(1), f"{raw_dataset_label}__{user}_")
    print("Done!")
    print()

    # 4. Sign Language MNIST ORIGINAL: user_N/A0.jpg, B3.jpg, ... (letter = first char)
    raw_dataset_label = "B02_aslyset"
    print(f"Adding {raw_dataset_label}...")
    asly = os.path.join(ROOT, "raw_data", raw_dataset_label, "ASLYset", "images")
    for user in sorted(os.listdir(asly)):
        path = os.path.join(asly, user)
        if os.path.isdir(path):
            copy_letter_dirs(
                base=path,
                prefix=f"{raw_dataset_label}__{user}_"
            )

    print("Done!")
    print()

    print("Images per source:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"Total: {sum(counts.values())}")

    end_time = time.time()
    print(f"Processing datasets took {end_time-start_time} seconds")


if __name__ == "__main__":
    main()
