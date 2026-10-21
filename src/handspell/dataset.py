from enum import Enum
import os
from shutil import copyfile
import re

class LabelMode(Enum):
    FROM_SUBFOLDER = 1
    FROM_FILE_FIRST_LETTER = 2

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # A–Y without J and Z
IMG_EXT = (".jpg", ".jpeg", ".png")


def add_image_to_dataset(image_path_src, letter, prefix, image_path_out):
    """Copy one image into OUT/<LETTER>/<prefix>_<filename>."""
    letter = letter.upper()
    if letter not in LETTERS:
        return
    # Skip empty/corrupt files
    if os.path.getsize(image_path_src) == 0:
        return
    # Replace spaces with underscores
    image_name_clean = os.path.basename(image_path_src).replace(' ', '_')
    name = f"{prefix}__{letter}__{image_name_clean}"
    letter_folder_out = os.path.join(image_path_out, letter)
    dst = os.path.join(letter_folder_out, name)
    if not os.path.exists(dst):
        os.makedirs(letter_folder_out, exist_ok=True)
        copyfile(image_path_src, dst)


def crawl_letter_dirs(root_path, prefix, out_path):
    """Copy images from a folder whose subfolders are single letters (A, b, ...)."""

    counts = {}
    for d in sorted(os.listdir(root_path)):
        path = os.path.join(root_path, d)
        if os.path.isdir(path) and len(d) == 1:
            letter = d
            letter_folder = path
            print(f"Letter Folder: {letter_folder}")
            for f in os.listdir(path):
                if f.lower().endswith(IMG_EXT):
                    image_path = os.path.join(letter_folder, f)
                    add_image_to_dataset(image_path, letter, prefix,
                                         out_path)
                    counts[prefix] = counts.get(prefix, 0) + 1

    return counts

def crawl_flat_folder(in_path, prefix, out_path):
    # Letter is first character of the file name
    counts = {}
    for f in os.listdir(in_path):
        m = re.match(r"([A-Za-z])\d+", f)
        if m and f.lower().endswith(IMG_EXT):
            letter = m.group(1)
            image_path = os.path.join(in_path, f)
            add_image_to_dataset(image_path, letter, prefix, out_path)
            counts[prefix] = counts.get(prefix, 0) + 1
    return counts


def create_or_add_to_dataset(original_data_label:str, dataset_label: str,
                   raw_data_folder_path:str,
                   datasets_folder="data/dataset",
                   user_folders=False,
                   label_mode=LabelMode.FROM_SUBFOLDER):

    out_folder = os.path.join(datasets_folder, dataset_label)
    print(f"Dataset folder: {out_folder}")
    os.makedirs(out_folder, exist_ok=True)

    crawl_function = crawl_letter_dirs if (
        label_mode ==LabelMode.FROM_SUBFOLDER) else crawl_flat_folder

    data_prefix = f"{original_data_label}"
    counts = {}
    if user_folders:
        for user in sorted(os.listdir(raw_data_folder_path)):
            path = os.path.join(raw_data_folder_path, user)
            if os.path.isdir(path):
                user_folder = path
                print(f"User Folder: {user_folder}")
                prefix = f'{data_prefix}__{user}'
                user_counts = crawl_function(user_folder, prefix, out_folder)
                counts = {**counts, **user_counts}
    else:
        counts = crawl_function(raw_data_folder_path, prefix, out_folder)

    return counts


if __name__ == "__main__":

    dataset_label = "D01_megadataset"

    # 1.
    raw_dataset_label = "A03_kaggle_asl"
    raw_dataset_root_folder = os.path.join("data", "raw_data", raw_dataset_label, "American")
    print(raw_dataset_label)
    create_or_add_to_dataset(
        dataset_label=dataset_label,
        original_data_label=raw_dataset_label,
        raw_data_folder_path=raw_dataset_root_folder)

    # 2.
    raw_dataset_label = "A04_mandalay_aslhg"
    raw_dataset_root_folder = os.path.join("data", "raw_data", raw_dataset_label, "ASL_HG_36000", "asl_dataset")
    print(raw_dataset_label)
    create_or_add_to_dataset(
        dataset_label=dataset_label,
        original_data_label=raw_dataset_label,
        raw_data_folder_path=raw_dataset_root_folder)

    # 3.
    raw_dataset_label = "B01_original_aslmnist"
    raw_dataset_root_folder = os.path.join("data", "raw_data", raw_dataset_label, "Dataset")
    print(raw_dataset_label)
    c = create_or_add_to_dataset(
            dataset_label=dataset_label,
            original_data_label=raw_dataset_label,
            raw_data_folder_path=raw_dataset_root_folder,
            user_folders=True,
            label_mode=LabelMode.FROM_FILE_FIRST_LETTER)
    print(c)

    # 4.
    raw_dataset_label = "B02_aslyset"
    raw_dataset_root_folder = os.path.join("data", "raw_data", raw_dataset_label, "ASLYset", "images")
    print(raw_dataset_label)
    c = create_or_add_to_dataset(
                dataset_label=dataset_label,
                original_data_label=raw_dataset_label,
                raw_data_folder_path=raw_dataset_root_folder,
                user_folders=True)
    print(c)
    print()
