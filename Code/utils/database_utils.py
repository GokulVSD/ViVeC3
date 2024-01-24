from os import path
from torch import load, save
import blosc
import pickle
from pathlib import Path

DATABASE_PATH = path.join(path.dirname(path.dirname( __file__ )), 'database')


def store(obj, filename):
    """
    Stores in /Code/database/ as a pickled .pt file. Tracked by git LFS.
    """
    print("\n", "Saving: ", filename, "\n")

    save(obj, path.join(DATABASE_PATH, filename))


def retrieve(filename):
    """
    Retrieves .pt file in /Code/database/ as object.
    """
    return load(path.join(DATABASE_PATH, filename))


def exists(filename):
    """
    Check if file exists.
    """
    file = Path(path.join(DATABASE_PATH, filename))
    return file.is_file()


def compressed_store(obj, filename):
    print("\n", "Saving: ", filename, "\n")
    pickled_obj = pickle.dumps(obj)
    compressed_obj = blosc.compress(pickled_obj)

    with open(path.join(DATABASE_PATH, filename), 'wb') as f:
        f.write(compressed_obj)

def compressed_retrieve(filename):
    with open(path.join(DATABASE_PATH, filename), 'rb') as f:
        compressed_data = f.read()

    decompressed_binary = blosc.decompress(compressed_data)
    decompressed_data = pickle.loads(decompressed_binary)
    return decompressed_data