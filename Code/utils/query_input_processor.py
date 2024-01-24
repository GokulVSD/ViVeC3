from .dataset_utils import initialize_dataset, get_image_with_label
from os import path
import PIL.Image
from json import load
from numpy import array

INPUT_PATH = path.join(path.dirname(path.dirname( __file__ )), 'input')


def get_query_image(image_input):
    """
    Handle user input for query image, either IMAGE_ID or an image file.
    Returns PIL image or raises an exception.
    """
    if image_input.isnumeric():

        IMG_ID = int(image_input)
        dataset = initialize_dataset()

        if IMG_ID < 0 or IMG_ID >= len(dataset):
            raise Exception("Image ID out of range for Caltech101.")

        return get_image_with_label(dataset, IMG_ID)[0]

    else:

        return PIL.Image.open(path.join(INPUT_PATH, image_input))


def check_label_is_valid(label):
    """
    Checks to see that the provided label is present in Caltech101.
    """
    print("Input label: ", label)

    dataset = initialize_dataset()
    valid_labels = dataset.categories
    if label.isnumeric():
        try:
            label = valid_labels[int(label)]
        except Exception as e:
            raise Exception("Input Label ID is out of bounds.")
    elif label not in valid_labels:
        raise Exception("Label does not exist in Caltech101.")
    else:
        pass

    print("Output label:", label, "\n\n")
    return label


def get_vectors_from_file(file_name):
    """
    Return numpy vectors
    """
    return array(load(open(path.join(INPUT_PATH, file_name))))


def get_feedback_system(ch):
    ch = int(ch)
    if ch != 1 and ch != 2:
        raise Exception("Invalid feedback system")
    return ch


def relevance_feedback_input(ch, valid_image_ids):
    """
    Return None if should quit,
    else returns False if the input was invalid,
    else returns list of (image_id, feedback)
    """
    valid_image_ids = set(valid_image_ids)
    if ch.upper() == 'Q':
        return None
    try:
        feedbacks = []
        tokens = ch.split(',')
        for token in tokens:
            trimmed = token.strip().split(' ')
            image_id = int(trimmed[0].strip())
            feedback = trimmed[1].strip()
            if feedback not in ("R+", "R", "I", "I-"):
                return False
            if image_id not in valid_image_ids:
                return False
            feedbacks.append((image_id, feedback))
    except:
        return False
    return feedbacks


def align_print(val, max_val_len):
    return val + "".join([" "] * (max_val_len - len(val)))

