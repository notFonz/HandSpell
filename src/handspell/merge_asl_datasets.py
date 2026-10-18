"""
Merge 6 ASL image datasets into one folder ("Mega Dataset") with one
subfolder per letter A–Y (excluding J and Z, which require motion).

- Files are COPIED, so the original datasets stay untouched.
- Each file is renamed "<source>_<original name>" to avoid name collisions.
- Digits, J, Z, "Nothing", "Space" and "_" classes are skipped.
- Fingerspelling dataset: only RGB "color_" images (no depth maps).
- Sign Language Gesture Images: only raw images (no pre-processed copies).
- Empty (0-byte) files are skipped.
- Safe to re-run

Known broken image (found and removed after the first merge):
    Dataset:     American Sign Language
    Source file: American Sign Language/Train/V/V882.jpg  (0 bytes, empty)
    Merged copy: Mega Dataset/V/asl_train_V882.jpg  (deleted)
The original source file is still there; the empty-file check in put()
keeps it out of the merged dataset.: files that already exist in the output are skipped.
"""

import os
import re
import shutil

ROOT = os.path.expanduser("~/code/data")
OUT = os.path.join(ROOT, "Mega Dataset")

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # A–Y without J and Z
IMG_EXT = (".jpg", ".jpeg", ".png")

counts = {}


def put(src, letter, prefix):
    """Copy one image into OUT/<LETTER>/<prefix>_<filename>."""
    letter = letter.upper()
    if letter not in LETTERS:
        return
    # Skip empty/corrupt files, e.g. American Sign Language/Train/V/V882.jpg
    if os.path.getsize(src) == 0:
        return
    name = f"{prefix}_{os.path.basename(src).replace(' ', '_')}"
    dst = os.path.join(OUT, letter, name)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    counts[prefix] = counts.get(prefix, 0) + 1


def copy_letter_dirs(base, prefix, file_filter=lambda f: True):
    """Copy images from a folder whose subfolders are single letters (A, b, ...)."""
    for d in sorted(os.listdir(base)):
        path = os.path.join(base, d)
        if os.path.isdir(path) and len(d) == 1:
            for f in os.listdir(path):
                if f.lower().endswith(IMG_EXT) and file_filter(f):
                    put(os.path.join(path, f), d, prefix)


def main():
    for letter in LETTERS:
        os.makedirs(os.path.join(OUT, letter), exist_ok=True)

    # 1. American Sign Language (Train/ and Test/ subfolders)
    for split in ("Train", "Test"):
        copy_letter_dirs(
            os.path.join(ROOT, "American Sign Language", split),
            f"asl_{split.lower()}",
        )

    # 2. American Sign Language (ASL) Dataset (lowercase letter folders)
    copy_letter_dirs(os.path.join(ROOT, "American Sign Language (ASL) Dataset"), "asl-ds")

    # 3. ASL Fingerspelling Images (RGB & Depth): signers A–E, color images only
    fs = os.path.join(ROOT, "ASL Fingerspelling Images (RGB & Depth)")
    for user in sorted(os.listdir(fs)):
        if os.path.isdir(os.path.join(fs, user)):
            copy_letter_dirs(
                os.path.join(fs, user),
                f"fingerspell_{user}",
                lambda f: f.startswith("color_"),
            )

    # 4. ASL-HG
    copy_letter_dirs(
        os.path.join(ROOT, "ASL-HG American Sign Language Hand Gesture Image D",
                     "ASL_HG_36000", "asl_dataset"),
        "asl-hg",
    )

    # 5. Sign Language Gesture Images (raw only)
    copy_letter_dirs(
        os.path.join(ROOT, "Sign Language Gesture Images", "Gesture Image Data"),
        "gesture",
    )

    # 6. Sign Language MNIST ORIGINAL: user_N/A0.jpg, B3.jpg, ... (letter = first char)
    mnist = os.path.join(ROOT, "Sign Language MNIST ORIGINAL", "Dataset")
    for user in sorted(os.listdir(mnist)):
        path = os.path.join(mnist, user)
        if os.path.isdir(path):
            for f in os.listdir(path):
                m = re.match(r"([A-Za-z])\d+", f)
                if m and f.lower().endswith(IMG_EXT):
                    put(os.path.join(path, f), m.group(1), f"mnist-orig_{user}")

    print("Images per source:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"Total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
