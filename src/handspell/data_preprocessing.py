"""Detect and preprocess hand images for machine learning.

Uses a YOLO-based model to detect hands, extract square image crops,
convert them to grayscale, and normalize them to a resolution of
<<<<<<< HEAD
28  28 pixels.
=======
square of chosen size pixels.
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0

Processed images are saved in a 'preprocessed_images' subdirectory
within the selected preprocessing directory.
"""

from skimage.color import rgb2gray
from ultralytics import YOLO
from os import listdir
from PIL import Image
from pathlib import Path
<<<<<<< HEAD
from rich.progress import Progress
from tkinter import filedialog
import tkinter as tk
=======
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0
import cv2
import numpy as np
import subprocess
import os
import pandas as pd

MODEL_PATH = os.environ["HAND_DETECTION_MODEL_PATH"]

def load_model(model_path=None, download_if_needed=True):

<<<<<<< HEAD
def load_model(model_path=None):

=======
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0
    #Check if model already exists
    if model_path is None:
        model_path = MODEL_PATH

<<<<<<< HEAD
=======
    print("Hand detection: ", model_path, os.path.exists(model_path))
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0

    if os.path.exists(model_path):
        print('Model already downloaded')

<<<<<<< HEAD
    else:

=======
    elif download_if_needed:
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0
        # Download model for hand detection
        subprocess.run([
        "curl",
        "-L",
        "-o",
        model_path,
        "https://huggingface.co/NightingaleCen/YOLO26m-seg-hand/resolve/main/yolo26m-seg-hand.pt"
        ], check=True)
    else:
        print('Hand detection model is missing: ', model_path)
        return

    model = YOLO(model_path)

    return model


<<<<<<< HEAD
def image_preprocessing(image_path, output_size=50, model=None):
=======
def image_preprocessing(image_path,
                        output_size=64,
                        model=None,
                        save_image=True):
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0

    if model is None:
        model = load_model()

    if (image_path.endswith(".png") or image_path.endswith(".jpg")\
        or image_path.endswith(".jpeg")):

        # Prediction of hand segmentaion
<<<<<<< HEAD
        hand_prediction = model.predict(image_path, save=True, save_crop = True)
=======
        hand_prediction = model.predict(image_path,
                                        save=save_image,
                                        save_crop=save_image,
                                       verbose=False)
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0


        # Output if hand could not detected
        if len(hand_prediction[0].boxes) == 0:
            print(f"No hand detected in {image_path}; skipping image.")
            return
        x1, y1, x2, y2 = [float(coordinate) for coordinate in hand_prediction[0].boxes.xyxy[0]]

        #Defining attributes of the predicted hand rectangle

        height = y2-y1
        width = x2-x1
        center_xy = np.array([(x1+x2)/2. , (y1+y2)/2.])

        # Square transformation
<<<<<<< HEAD

        diag = np.max([height, width])
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
=======
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
        resized_img = cv2.resize(grey_image,(output_size, output_size),
                                 interpolation=cv2.INTER_LANCZOS4)
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0

    return resized_img

def preprocessing_dataset_interactive(model, output_size = 50):

<<<<<<< HEAD
    project_path = Path(__file__).resolve().parents[2]
=======
def preprocessing_dataset(folder_path:str,
                          project_path:str,
                          model=None,
                          output_size: int=64):

    if model is None:
        # Load model once for the whole iteration
        model = load_model()
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0

    project_path = Path(project_path)

<<<<<<< HEAD
    #Opens window and ask to select input folder
    folder_path = filedialog.askdirectory()

    print(os.listdir(folder_path))
    folder_path = Path(folder_path)

    return preprocessing_dataset(folder_path, project_path, model, output_size)
=======
    folder_path = Path(folder_path)

    rows = []

    #Iterate through all the subfolders
    for subfolder in folder_path.iterdir():

        #Checks if subfolder has the type directory, if not continue with the next file
        if not subfolder.is_dir():
            continue

        #Determine the name of the current folder
        folder_name = Path(subfolder).name

        #Define the folder path for the respective processed-folder and create that folder
        new_folder_path = project_path / "data" / "preprocessed" / folder_name
        new_folder_path.mkdir(parents=True, exist_ok=True)

        print(f"Processing Folder: {subfolder.name}")

        #Iterate through all files inside  the subfolder
        for image_path in subfolder.iterdir():
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0

            #Checks if file is a file and not a folder
            if not image_path.is_file():
                continue

<<<<<<< HEAD
def preprocessing_dataset(folder_path, project_path, model, preprocessing_bool=True, output_size = 50):

    if model is None:
            model = load_model()
    # Load model once for the whole iteration

    folder_path = Path(folder_path)

    df = pd.DataFrame()

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
            print(image_path)
            #Checks if file is a file and not a folder
            if not image_path.is_file():
                continue

            #Checks if file is a image
            if image_path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
                continue

            images = Path(image_path).name
            print(f"Processing: {images} in folder: {subfolder.name}")

            #Call image_preprocessing for every image

            #if preprocessing_bool:
            resized_img = image_preprocessing(str(image_path), output_size, model)
           # else:
            #    resized_img = img



            ###############


            #img = Image.open(image_path, mode = 'r')
            #print(img)
            #grey_image = rgb2gray(img)

            ###############
=======
            #Checks if file is a image
            if image_path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
                continue

            images = Path(image_path).name

            #Call image_preprocessing for every image
            resized_img = image_preprocessing(str(image_path), output_size, model, save_image=False)

            if resized_img is None:

                continue

            # Extract letter from image name
            label = subfolder.name
            output_path = os.path.join(new_folder_path, os.path.join(label, images))
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0

            #Convert and write image file
            grey_image_uint8 = (resized_img *255).astype(np.uint8)
            cv2.imwrite(output_path, grey_image_uint8)

            # Extract letter from image name
<<<<<<< HEAD
            label = os.path.splitext(images)
            label = list(label[0])
=======
            label = subfolder.name
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0

            result=[]

            pixel_values = grey_image_uint8.flatten()

<<<<<<< HEAD
            result = {f'pixel{i+1}': pixel for i, pixel in enumerate(pixel_values)}

            new_row = pd.DataFrame([dict(result)], index=[label[0]])
            df = pd.concat([df, new_row])

    df.index.name = 'label'
    df.to_csv(project_path/ "data" / "preprocessed" / 'processed_training_data.csv')
    print(f'Saved csv file in {project_path/ "data" / "preprocessed"}')
=======
            #result = {f'pixel{i+1}': pixel for i, pixel in enumerate(pixel_values)}


            result = {'image_path': output_path}

            new_row = dict(label=label.upper(), **result)
            rows += [new_row]

    df = pd.DataFrame(rows).set_index("label")
    df.to_csv(project_path/ "data" / "preprocessed" / 'processed_training_data.csv')
    print(f'Saved csv file in {project_path/ "data" / "preprocessed"}')
    return df
>>>>>>> 0221f61edf4edd51f3059a36689cfc815d036fb0
