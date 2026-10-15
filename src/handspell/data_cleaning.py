import numpy as np
import pandas as pd

import os


def get_labels_images(path: str):
    """Function to transform each path containing folders with images to DataFrames, also added if statement so depth images dont get added"""

    # Assign empty lists to variables, labels (a to x) and images
    labels = []
    images = []

    # For each label in the path, folders are named a,b,c,...etc so they have the same names as our labels
    for label in os.listdir(path):

        # Skip everything that is not a directory, otherwise crash, s/o description.md :D
        if not os.path.isdir(os.path.join(path, label)):

            pass

        else:

            # For each image in each label folder
            for image in os.listdir(path + '/' + label):

                # Ignore depth images
                if image.startswith('depth'):

                    pass

                # Append labels and images to the empty variables, also transform label into uppercase
                # Note: need path of label on image
                else:
                    labels.append(label.upper()),
                    images.append(path + '/' + label + '/' + image)

    return pd.DataFrame({'labels':labels,'images':images})



def sort_values_df(list_of_dfs: list):
    """Function to sort values by labels on each DataFrame of a list of DataFrames"""

    for df in list_of_dfs:

        df.sort_values('labels', inplace=True)
        df.reset_index(drop=True, inplace=True)

    return


def drop_unwanted_df(df: pd.DataFrame):
    """Function to drop rows that contain unwanted values, J, Z, _ and numbers in our case"""

    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    df_cleaned = df[~(df.labels.isin(['J', 'Z', '_'] + numbers))].reset_index(drop=True)

    return df_cleaned
