import os
from pathlib import Path

# Assign root and images parent folder paths to variables

ROOT_PATH = Path(__file__).resolve().parents[2]

IMAGES_DIR = ROOT_PATH / 'data' / 'mapping' / 'ASL Letter Reference'


def text_to_image_dict(path):
    """Function to create a dictionary with letters as keys and image paths as values"""

    # Assign an empty dict to our mapping dictionary and assign the alphabet (No J and Z) to a list
    mappingdict = {}

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
               'X', 'Y']

    # For each file in our path (images labelled with each corresponding letter)
    for label in os.listdir(path):

        # Remove the .jpg suffix so we can use our label as a key value
        letter = label.replace('.jpg', '')

        # Join the path of the parent folder with the image(label)
        img_path = os.path.join(path, label)

        # If check to map letters to their corresponding image path
        if letter in letters:

            mappingdict[letter] = img_path

        # If the letter is not on our original list, print this cool error message :D
        else:

            print(f'{label} is an invalid letter ! Pumba does not approve -_-')

    return mappingdict


def text_to_image_path(letter: str, mappingdict: dict):
    """Function to get assigned value (path) of the letter in the mapping dictionary"""

    return mappingdict.get(letter.upper())

if __name__ == "__main__":

    result = text_to_image_dict(IMAGES_DIR)

    print(len(result))
    print(text_to_image_path('K', result))
    print(text_to_image_path('J', result))
    print(text_to_image_path('k', result))
    print(text_to_image_path('j', result))

    #print(Path(__file__))
    #print(Path(__file__).resolve())

    #print(Path('.'))
    #print(Path('.').resolve())

    #print(Path(__file__).resolve().parent)
    #print(Path(__file__).resolve().parent.parent)
    #print(Path(__file__).resolve().parent.parent.parent)

    #print(Path(__file__).resolve().parents[2])

    #root_path = Path(__file__).resolve().parents[2]

    #print(f'{root_path}/data/mapping/ASL Letter Reference')

    #test_path = root_path / 'data' / 'mapping' / 'ASL Letter Reference'

    #print(test_path)
