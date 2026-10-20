from tkinter import filedialog
import tkinter as tk
import os
from pathlib import Path

from handspell.data_preprocessing import preprocessing_dataset


def preprocessing_dataset_interactive(output_size = 64):

    project_path = Path(__file__).resolve().parents[2]

    root = tk.Tk()
    root.withdraw()

    #Opens window and ask to select input folder
    folder_path = filedialog.askdirectory()

    print(os.listdir(folder_path))

    return preprocessing_dataset(folder_path, project_path, output_size)
