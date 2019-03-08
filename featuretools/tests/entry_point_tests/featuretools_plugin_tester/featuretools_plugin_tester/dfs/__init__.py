import os
import json


def dfs(**kwargs):
    this_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(this_directory, "dfs_plugin_tester.json")
    # with open(file_path, 'w') as file:
    #     json.dump(kwargs, file)
    return
