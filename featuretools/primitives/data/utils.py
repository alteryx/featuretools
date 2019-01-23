import os.path


def get_primitive_data_path(filename):
    PWD = os.path.dirname(__file__)
    path = os.path.join(PWD, filename)
    return path
