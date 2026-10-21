import matplotlib.pyplot as plt
import numpy as np
from random import random
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import save_img
import os


from handspell.utils import INDEX_TO_LETTER


def get_image_size_and_channels(image_path):
    img = load_img(image_path)
    width, height = img._size
    channels = img.layers
    return {"width": width,
            "height": height,
            "channels": channels
            }

def print_dataset_infos(infos_df):
    n_images = infos_df.shape[0]
    try:
        n_people = infos_df["person"].nunique()
    except:
        n_people = None
    min_width = infos_df["width"].min()
    max_width = infos_df["width"].max()
    min_height = infos_df["height"].min()
    max_height = infos_df["height"].max()
    min_channels = infos_df["channels"].min()
    max_channels = infos_df["channels"].min()

    print(f"Dataset: {infos_df["dataset"].values[0]}")
    print(f"- {n_images} images")
    if n_people is not None:
        print(f"- {n_people} different people")
    if (min_width == max_width) and (min_height == max_height):
        print(f"- images {min_width}x{min_height} pixels")
    else:
        print(f"- images ranging from {min_width} to {max_width} in pixel width")
        print(f"- images from {min_height} to {max_height} in pixel height")
    if (min_channels == max_channels):
        if min_channels > 1:
            print(f"- All color images ({min_channels} channels)")
        else:
            print(f"- All greyscale images ({min_channels} channel)")
    else:
        print(f"- Images ranging from {min_channels} to {max_channels} channels")


# Plot example images for a given dataset
def get_image_from_info_row(row):
    img = load_img(row["image_path"])
    return img

def plot_example_images(dataset_df, get_image_from_row,
                        n_examples=5, figsize=(24, 10)):
    n_letters = len(INDEX_TO_LETTER.keys())

    # First half
    half_i = int(n_letters/2)
    f1, axs1 = plt.subplots(n_examples,half_i, figsize=figsize)
    f2, axs2 = plt.subplots(n_examples,half_i, figsize=figsize)

    for i, letter in INDEX_TO_LETTER.items():
        letter_bool = dataset_df["letter"]==letter
        print(letter, letter_bool.sum())

        for j in range(n_examples):
            if i < n_letters/2:
                ax = axs1[j, i]
            else:
                ax = axs2[j, i-half_i]
            if letter_bool.sum() > 0:
                random_row = dataset_df.loc[letter_bool].sample().iloc[0]
                img = get_image_from_row(random_row)
                ax.imshow(img, cmap="gray")
            ax.set_title(letter)
            ax.set_axis_off()

    return f1, f2


########################################
#                                      #
#      A01 DATASET FUNCTIONS           #
#                                      #
########################################

# A01 dataset functions

A01_LABEL = "A01_kaggle_aslmnist"
A01_IMAGE_PATH = "data/raw_data/{dataset_label}/{train_or_test}/{letter}/"
A01_IMAGE_NAME = "{letter}_{number}.jpg"

def get_image_from_row_a01(row, plot=True):
    letter = row["label"]
    row = row.values[:]
    img_2828 = np.array(row[1:], dtype=int).reshape(28, 28, 1)
    if plot:
        plt.imshow(img_2828, cmap="gray")
        plt.title(letter)
    return img_2828

# Save samples as images in folders per letter
def save_as_images_a01(dataset_label=A01_LABEL):
    train_df = pd.read_csv(f"../data/raw_data/{dataset_label}/sign_mnist_train.csv")
    train_samples = []
    for i, r in train_df.iterrows():
        letter = INDEX_TO_LETTER[r["label"]]
        img_2828 = get_image_from_row_a01(r, plot=False)

        letter_folder = "../" + A01_IMAGE_PATH.format(
            dataset_label=dataset_label,
            train_or_test="train",
            letter=letter)

        os.makedirs(letter_folder, exist_ok=True)
        image_path = letter_folder+A01_IMAGE_NAME.format(letter=letter, number=i)
        save_img(image_path, img_2828, quality=100)

        train_samples += [{"letter": letter, "image_path": image_path}]

    test_df = pd.read_csv(f"../data/raw_data/{dataset_label}/sign_mnist_test.csv")
    test_samples = []
    for i, r in test_df.iterrows():
        letter = INDEX_TO_LETTER[r["label"]]
        img_2828 = get_image_from_row_a01(r, plot=False)

        letter_folder = "../" + A01_IMAGE_PATH.format(
            dataset_label=dataset_label,
            train_or_test="test",
            letter=letter)

        os.makedirs(letter_folder,exist_ok=True)
        image_path = letter_folder + "/" + \
            A01_IMAGE_NAME.format(letter=letter, number=i)
        save_img(image_path, img_2828, quality=100)

        test_samples += [{"letter": letter, "image_path": image_path}]

    return train_samples, test_samples


# Get | dataset | letter | width | height | channels | image_path | train  or test
def get_a01_dataset_infos(dataset_label=A01_LABEL, save=True):
    dataset_path = "../data/raw_data/" + dataset_label

    train_samples, test_samples = save_as_images_a01()

    train_infos_df = pd.DataFrame(train_samples)
    train_infos_df["test"] = False


    test_infos_df = pd.DataFrame(test_samples)
    test_infos_df["test"] = True

    dataset_infos_df = pd.concat([train_infos_df, test_infos_df])
    dataset_infos_df["dataset"] = A01_LABEL
    dataset_infos_df["width"] = 28
    dataset_infos_df["height"] = 28
    dataset_infos_df["channels"] = 1
    dataset_infos_df["hand_cropped"] = True
    dataset_infos_df["preprocessed"] = True

    if save == True:
        dataset_infos_df.to_csv(f"{dataset_path}/dataset_infos.csv", sep="\t")

    return dataset_infos_df, dataset_path


########################################
#                                      #
#      B01 DATASET FUNCTIONS           #
#                                      #
########################################

B01_LABEL = "B01_original_aslmnist"

# Get | dataset | person | label | image_name | width | height | channels | image_path |
def get_b01_dataset_infos(dataset_label=B01_LABEL, save=True):
    dataset_path = "../data/raw_data/" + dataset_label
    rows = []
    exceptions = []
    for user_folder in os.listdir(dataset_path):
        try:
            folder_files = os.listdir(f"{dataset_path}/{user_folder}")
        except:
            exceptions += [f"{dataset_path}/{user_folder}"]
            continue
        for f in folder_files:
            try:
                img_props = get_image_size_and_channels(
                    dataset_path+"/"+user_folder+"/"+f
                )
                letter = f[0]
                rows += [dict(dataset=dataset_label,
                                person=user_folder,
                                letter=letter,
                                image_name=f,
                                **img_props,
                                image_path=f"{dataset_path}/{user_folder}/{f}",
                                hand_cropped=False,
                                preprocessed=False
                        )]
            except:
                exceptions += [f"{dataset_path}/{user_folder}/{f}"]
    dataset_description_df = pd.DataFrame(rows)

    if save:
        dataset_description_df.to_csv(dataset_path+"/dataset_infos.csv", sep="\t")

    return dataset_description_df, dataset_path
