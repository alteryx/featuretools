import os
from featuretools.primitives.install import (load_primitive_from_file,
                                             list_primitive_files)

path = os.path.dirname(__file__)
files = list_primitive_files(os.path.join(path))
primitives = dict([load_primitive_from_file(filepath) for filepath in files])
