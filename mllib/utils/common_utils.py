import os

def is_file_in_dir(d, filename):
    return os.path.exists(os.path.join(d, filename))