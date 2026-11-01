import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image, ImageOps
import numpy as np
import os

from handspell.utils import INDEX_TO_LETTER


def open_image_from_path(image_path:str, exif_transpose=True) -> Image:
    img = Image.open(image_path)


    # EXIF information indicates if image was rotated or mirrored
    if exif_transpose:
        h0, w0 = img.size
        img = ImageOps.exif_transpose(img)
        h, w = img.size
        if h0!=h or w0!=w:
            print(f"EXIF magic! ({h0}, {w0}) to ({h}, {w})")

    # Most reliable way to get whether an image is grayscaled or not,
    #  without getting into pillow's image modes
    img_array = np.array(img)

    h, w = img_array.shape[:2]

    if len(img_array.shape) < 3:
        c = 1
    else:
        _, _, c = img_array.shape


    # img = Image.convert("RGB")
    return img, img_array, {"height": h, "width": w, "channels": c}


def plot_image_in_ax(image_arr: np.array, ax,
                     channels=3,
                     min_value=0,
                     max_value=255):
    cmap = None
    if channels == 1:
        cmap = "gray"

    ax.imshow(image_arr,
            cmap=cmap,
            vmin=min_value,
            vmax=max_value)
    ax.set_axis_off()
    return ax


def plot_image(image_arr: np.array,
               min_value=0,
               max_value=255,
               scale=1,
               dpi=100) -> tuple[Figure, Axes]:
    """Barebones plotting of one image, no padding. """

    if len(image_arr.shape) > 2:
            h, w, c = image_arr.shape
    else:
        h, w = image_arr.shape
        c = 1

    # each pixel becomes a scale×scale block
    new_h = scale * h / dpi
    new_w = scale * w / dpi

    f, ax = plt.subplots(figsize=(new_w, new_h), dpi=dpi)
    f.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plot_image_in_ax(image_arr=image_arr,
                     ax=ax,
                     channels=c,
                     min_value=min_value,
                     max_value=max_value
                     )

    return f, ax




def plot_example_images(dataset_infos_df, path_prefix="",
                        n_examples=5, figsize=(28, 10),
                        debug=False):

    n_letters = len(INDEX_TO_LETTER.keys())

    # First half
    half_i = int(n_letters/2)
    f1, axs1 = plt.subplots(n_examples,half_i, figsize=figsize)
    f1.subplots_adjust(left=0.15, right=0.85, top=0.8, bottom=0, wspace=0.05, hspace=0.01)
    f2, axs2 = plt.subplots(n_examples,half_i, figsize=figsize)
    f2.subplots_adjust(left=0.15, right=0.85, top=0.8, bottom=0, wspace=0.05, hspace=0.01)

    for i, letter in INDEX_TO_LETTER.items():
        letter_bool = dataset_infos_df["label"]==letter
        n_images_letter = letter_bool.sum()

        if n_images_letter > 0:
            random_rows = dataset_infos_df.loc[letter_bool]\
                          .sample(min(n_examples, n_images_letter), replace=False)

        for j in range(n_examples):
            if i < n_letters/2:
                ax = axs1[j, i]
            else:
                ax = axs2[j, i-half_i]

            if j == 0:
                ax.set_title(letter)

            if (n_images_letter > (j+1)):
                random_row = random_rows.iloc[j]
                if debug:
                    print(i, j, random_row.index, random_row[["label", "image_name"]].values)

                _, img_arr, _ = open_image_from_path(
                    image_path=os.path.join(path_prefix, random_row["image_path"]),
                )
                c = random_row["channels"]
                plot_image_in_ax(image_arr=img_arr,
                                ax=ax,
                                channels=c)
            ax.set_axis_off()

    return f1, f2
