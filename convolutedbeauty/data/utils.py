import os


def get_images(dir_path, ends_with=".jpg"):
    """
    Get all jpgs in a directory

    :param dir_path: (str) the path to the directory
    :return: (list) a list of file paths in the directory
    """

    return [os.path.join(dir_path, file) for file in os.listdir(dir_path) if ends_with and file.endswith(ends_with)]
