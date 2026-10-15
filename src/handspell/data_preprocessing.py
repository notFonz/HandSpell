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


MODEL_PATH = os.environ["HAND_DETECTION_MODEL_PATH"]


def load_model(model_path=None):
    # Check if model already exists
    if model_path is None:
        model_path = MODEL_PATH

    if os.path.exists(model_path):
        print('Model already downloaded')
    else:
        # Download model for hand detection
        subprocess.run([
        "curl",
        "-L",
        "-o",
        model_path,
        "https://huggingface.co/NightingaleCen/YOLO26m-seg-hand/resolve/main/yolo26m-seg-hand.pt"
        ], check=True)

    model = YOLO(model_path)

    return model


def image_preprocessing(image_path, output_size=64, model=None):

    if model is None:
        model = load_model()

    else:
    # Track Progress
        with Progress() as p:

            t = p.add_task("Processing...", total=100)

            if (image_path.endswith(".png") or image_path.endswith(".jpg")\
                or image_path.endswith(".jpeg")):

                # Prediction of hand segmentaion
                hand_prediction = model.predict(image_path, save=True, save_crop = True)

                # Output if hand could not detected
                if len(hand_prediction[0].boxes) == 0:
                    print(f"No hand detected in {image_path}; skipping image.")

                    return
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

                #Resize Image
                resized_img = cv2.resize(grey_image,(output_size, output_size), interpolation=cv2.INTER_LANCZOS4)

                #Track progress
                while not p.finished:
                    p.update(t, advance=1)
                    time.sleep(0.05)

    return resized_img


def folder_processing(output_size = 64):

    # Load model once for the whole iteration
    model = load_model()

    root = tk.Tk()
    root.withdraw()

    project_path = Path(__file__).resolve().parents[2]

    #Opens window and ask to select input folder
    folder_path = filedialog.askdirectory()

    print(os.listdir(folder_path))
    folder_path = Path(folder_path)

    #Iterate through all the subfolders
    for subfolder in folder_path.iterdir():

        #Checks if subfolder has the type directory, if not continue with the next file
        if not subfolder.is_dir():
            continue

        #Determine the name of the current folder
        folder_name = Path(subfolder).name

        #Define the folder path for the respective processed-folder and create that folder
        new_folder_path = project_path/ "data" / "preprocessed"/ folder_name
        new_folder_path.mkdir(parents=True, exist_ok=True)

        #Iterate through all files inside  the subfolder
        for image_path in subfolder.iterdir():

            #Checks if file is a file and not a folder
            if not image_path.is_file():
                continue

            #Checks if file is a image
            if image_path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
                continue

            images = Path(image_path).name
            print(f"Processing: {images} in folder: {subfolder.name}")

            #Call image_preprocessing for every image
            resized_img = image_preprocessing(str(image_path), output_size, model)

            #Convert and write image file
            grey_image_uint8 = (resized_img *255).astype(np.uint8)
            cv2.imwrite(os.path.join(new_folder_path, images), grey_image_uint8)
