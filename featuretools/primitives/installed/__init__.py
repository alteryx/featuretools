# flake8: noqa
from ..install import load_primitive_from_file, list_primitive_files, get_installation_dir


# iterate over files in installed, import class that are right subclass
installed_dir = get_installation_dir()
files = list_primitive_files(installed_dir)
for filepath in files:
    primitive_name, primitive_obj = load_primitive_from_file(filepath)
    globals()[primitive_name] = primitive_obj
