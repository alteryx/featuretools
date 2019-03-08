import os


def initialize():
    this_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(this_directory, "initialize_plugin_called.txt")
    with open(file_path, 'w') as file:
        file.write("The function was called.")
    return
