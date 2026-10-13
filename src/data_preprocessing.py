"""Detect and preprocess hand images for machine learning.

Uses a YOLO-based model to detect hands, extract square image crops,
convert them to grayscale, and normalize them to a resolution of
28 × 28 pixels.

Processed images are saved in a 'preprocessed_images' subdirectory
within the selected input directory.
"""

from matplotlib.patches import Rectangle
from skimage.color import rgb2gray
from tkinter import filedialog
from ultralytics import YOLO
from os import listdir
from PIL import Image
from pathlib import Path
from rich.progress import Progress
import time
import os
import cv2
import tkinter as tk
import numpy as np
import subprocess

# Check if model already exists
file = Path("models/yolo26m-seg-hand.pt")

if file.exists():
    print('Model already downloaded')
else:
    # Download model for hand detection
    subprocess.run([
        "curl",
        "-L",
        "-o",
        "models/yolo26m-seg-hand.pt",
        "https://huggingface.co/NightingaleCen/YOLO26m-seg-hand/resolve/main/yolo26m-seg-hand.pt"
    ], check=True)

model = YOLO("models/yolo26m-seg-hand.pt")

root = tk.Tk()
root.withdraw()

# Opens window and ask to select input folder
folder_path = filedialog.askdirectory()


# Detect foler path and create new folder for processed images
parent_path = os.path.dirname(folder_path)

project_path = Path(__file__).resolve().parents[2]

folder_name = Path(folder_path).name

new_folder_path = project_path/ "data" / "preprocessed"/ folder_name

new_folder_path.mkdir(parents=True, exist_ok=True)

pixel_size = 28

# Track Progress
with Progress() as p:

    for images in os.listdir(folder_path):

        t = p.add_task("Processing...", total=100)

        if (images.endswith(".png") or images.endswith(".jpg")\
            or images.endswith(".jpeg")):

            image_path = os.path.join(folder_path, images)

            # Prediction of hand segmentaion
            hand_prediction = model.predict(image_path, save=True)

            # Output if hand could not detected
            if len(hand_prediction[0].boxes) == 0:
                print(f"No hand detected in {images}; skipping image.")
                continue

            x1, y1, x2, y2 = [float(coordinate) for coordinate in hand_prediction[0].boxes.xyxy[0]]

            #Defining attributes of the predicted hand rectangle
            origin_xy = hand_prediction[0].boxes.xyxy[0][:2]
            height = y2-y1
            width = x2-x1
            center_xy = np.array([(x1+x2)/2. , (y1+y2)/2.])

            # Square transformation
            diag = (height**2 + width**2)**0.5
            orig_square_xy = center_xy - diag/2

            # Defining new X and Y Coordinates
            x_left = orig_square_xy[0]
            y_up = orig_square_xy[1]
            x_right = orig_square_xy[0] + diag
            y_low = orig_square_xy[1] + diag

            # Crop images
            img = Image.open(image_path, mode = 'r')
            cropped_image = img.crop([x_left, y_up, x_right, y_low])

            # Set image color to grey
            grey_image = rgb2gray(cropped_image)

            # Resize Image
            resized_img = cv2.resize(grey_image,(28, 28), interpolation=cv2.INTER_LANCZOS4)

            #Convert and write image file
            grey_image_uint8 = (resized_img *255).astype(np.uint8)
            cv2.imwrite(os.path.join(new_folder_path, images), grey_image_uint8)

            #Track progress
            while not p.finished:
                p.update(t, advance=1)
                time.sleep(0.05)
