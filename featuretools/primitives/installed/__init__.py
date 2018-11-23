from featuretools.primitive_utils import load_primitives_from_file, list_primitive_files, get_installation_dir


# iterate over files in installed, import class that are right subclass
installed_dir = get_installation_dir()
files = list_primitive_files(installed_dir)
for filepath in files:
    primitives = load_primitives_from_file(filepath)
    for p in primitives:
        globals()[p] = primitives[p]
