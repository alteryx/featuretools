import os


def initialize():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(this_directory, "initialize_plugin_called.txt")
    file = open("initialize_plugin_called.txt", "w")
    file.write("The function was called.")
    file.close()
    return
