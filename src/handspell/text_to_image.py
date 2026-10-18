import os


def text_to_image_dict(path):

    # Assign an empty dict to our mapping dictionnary and assign the alphabet to a list
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

    return mappingdict.get(letter.upper())

if __name__ == "__main__":

    result = text_to_image_dict("/Users/fonz/code/notFonz/HandSpell/data/mapping/ASL Letter Reference")

    print(len(result))
    print(text_to_image_path('K', result))
    print(text_to_image_path('J', result))
    print(text_to_image_path('k', result))
    print(text_to_image_path('j', result))
