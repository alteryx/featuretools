from featuretools.utils.gen_utils import import_or_raise

SKLEARN_ERR_MSG = (
    "The featuretools_sklearn_transformer library is required to use DFSTransformer.\n"
    "Install via pip:\n"
    "    pip install featuretools_sklearn_transformer\n"
)
DFSTransformer = import_or_raise("featuretools_sklearn_transformer", SKLEARN_ERR_MSG).DFSTransformer
