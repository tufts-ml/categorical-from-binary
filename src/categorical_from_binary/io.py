import json
import os


def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


def read_json(filename):
    with open(filename) as f:
        return json.load(f)


def ensure_dir(directory):
    """
    Description:
    Makes sure directory exists before saving to it.

    Parameters:
            directory: An string naming the directory on the local machine where we will save stuff.

    """
    # alternative: os.makedirs(directory, exist_ok=True)
    if not os.path.isdir(directory):
        os.makedirs(directory)
