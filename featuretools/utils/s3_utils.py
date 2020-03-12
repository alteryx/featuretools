import json
import shutil

from featuretools.utils.gen_utils import import_or_raise


def use_smartopen_es(file_path, path, transport_params=None, read=True):
    open = import_or_raise("smart_open", SMART_OPEN_ERR_MSG).open
    if read:
        with open(path, "rb", transport_params=transport_params) as fin:
            with open(file_path, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
    else:
        with open(file_path, 'rb') as fin:
            with open(path, 'wb', transport_params=transport_params) as fout:
                shutil.copyfileobj(fin, fout)


def use_smartopen_features(path, features_dict=None, transport_params=None, read=True):
    open = import_or_raise("smart_open", SMART_OPEN_ERR_MSG).open
    if read:
        with open(path, 'r', encoding='utf-8', transport_params=transport_params) as f:
            features_dict = json.load(f)
            return features_dict
    else:
        with open(path, "w", transport_params=transport_params) as f:
            json.dump(features_dict, f)


def get_transport_params(profile_name):
    boto3 = import_or_raise("boto3", BOTO3_ERR_MSG)
    UNSIGNED = import_or_raise("botocore", BOTOCORE_ERR_MSG).UNSIGNED
    Config = import_or_raise("botocore.config", BOTOCORE_ERR_MSG).Config

    if isinstance(profile_name, str):
        transport_params = {'session': boto3.Session(profile_name=profile_name)}
    elif profile_name is False or boto3.Session().get_credentials() is None:
        transport_params = {
            'resource_kwargs': {'config': Config(signature_version=UNSIGNED)}
        }
    else:
        transport_params = None
    return transport_params


BOTO3_ERR_MSG = (
    "The boto3 library is required to read and write from URLs and S3.\n"
    "Install via pip:\n"
    "    pip install boto3\n"
    "Install via conda:\n"
    "    conda install boto3"
)
BOTOCORE_ERR_MSG = (
    "The botocore library is required to read and write from URLs and S3.\n"
    "Install via pip:\n"
    "    pip install botocore\n"
    "Install via conda:\n"
    "    conda install botocore"
)
SMART_OPEN_ERR_MSG = (
    "The smart_open library is required to read and write from URLs and S3.\n"
    "Install via pip:\n"
    "    pip install smart-open\n"
    "Install via conda:\n"
    "    conda install smart_open"
)
