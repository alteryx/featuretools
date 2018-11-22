from featuretools.primitive_utils.aggregation_primitive_base import PrimitiveBase

import importlib.util
import os

# iterate over files in installed, import class that are right subclass
installed_dir = os.path.dirname(os.path.realpath(__file__))  # path to the installed dir
files = os.listdir(installed_dir)
for path in files:
    if path[:2] == "__" or path[0] == "." or path[-3:] != ".py":
        continue
    path = os.path.join(installed_dir, path)
    spec = importlib.util.spec_from_file_location("module.name", path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    for primitive_name in dir(foo):
        primitive_class = getattr(foo, primitive_name)
        try:
            if issubclass(primitive_class, PrimitiveBase):
                globals()[primitive_name] = primitive_class
        except:
            continue

    del foo
