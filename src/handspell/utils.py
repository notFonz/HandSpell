import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image, ImageOps
import numpy as np

LETTER_TO_INDEX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25
}

INDEX_TO_LETTER = {v:k for k, v in LETTER_TO_INDEX.items()}


def open_image_from_path(image_path:str, exif_transpose=True) -> Image:
    img = Image.open(image_path)
    channels = img.layers


    # EXIF information indicates if image was rotated or mirrored
    if exif_transpose:
        h0, w0 = img.size
        img = ImageOps.exif_transpose(img)
        h, w = img.size
        if h0!=h or w0!=w:
            print(f"EXIF magic! ({h0}, {w0}) to ({h1}, {w1})")

    h, w = img.size

    # img = Image.convert("RGB")
    return img, {"heigh": h, "width": w, "channels": channels}


def plot_image(image_arr: np.array,
               min_value=0,
               max_value=255,
               scale=1,
               dpi=100) -> Figure:
    if len(image_arr.shape) > 2:
        h, w, c = image_arr.shape
    else:
        h, w = image_arr.shape
        c = 1

    # each pixel becomes a scale×scale block
    new_h = scale * h / dpi
    new_w = scale * w / dpi
    cmap = None
    if c == 1:
        cmap = "gray"

    f, ax = plt.subplots(figsize=(new_w, new_h), dpi=dpi)
    f.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.imshow(image_arr,
              cmap=cmap,
              vmin=min_value,
              vmax=max_value)
    ax.set_axis_off()

    return f, ax
