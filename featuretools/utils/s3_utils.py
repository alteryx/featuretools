import json
import shutil

from featuretools.utils.gen_utils import import_or_raise


def use_smartopen_es(file_path, path, transport_params=None, read=True):
    open = import_or_raise("smart_open", "The smart_open library is required to upload or download EntitySets from S3 or URLs").open
    if read:
        with open(path, "rb", transport_params=transport_params) as fin:
            with open(file_path, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
    else:
        with open(file_path, 'rb') as fin:
            with open(path, 'wb', transport_params=transport_params) as fout:
                shutil.copyfileobj(fin, fout)


def use_s3fs_es(file_path, path, read=True):
    s3fs = import_or_raise("s3fs", "The s3fs library is required to upload or download EntitySets from S3")
    s3 = s3fs.S3FileSystem(anon=True)
    if read:
        s3.get(path, file_path)
    else:
        s3.put(file_path, path)


def use_smartopen_features(path, features_dict=None, transport_params=None, read=True):
    open = import_or_raise("smart_open", SMART_OPEN_ERR_MSG).open
    if read:
        with open(path, 'r', encoding='utf-8', transport_params=transport_params) as f:
            features_dict = json.load(f)
            return features_dict
    else:
        with open(path, "w", transport_params=transport_params) as f:
            json.dump(features_dict, f)


def use_s3fs_features(file_path, features_dict=None, read=True):
    s3fs = import_or_raise("s3fs", S3FS_ERR_MSG)
    s3 = s3fs.S3FileSystem(anon=True)
    if read:
        with s3.open(file_path, "r", encoding='utf-8') as f:
            features_dict = json.load(f)
            return features_dict
    else:
        with s3.open(file_path, "w") as f:
            features = json.dumps(features_dict, ensure_ascii=False)
            f.write(features)


BOTO3_ERR_MSG = (
    "The boto3 library is required to read and write from URLs and S3.\n"
    "Install via pip:\n"
    "    pip install boto3\n"
    "Install via conda:\n"
    "    conda install boto3"
)
SMART_OPEN_ERR_MSG = (
    "The smart_open library is required to read and write from URLs and S3.\n"
    "Install via pip:\n"
    "    pip install smart-open\n"
    "Install via conda:\n"
    "    conda install smart_open"
)
S3FS_ERR_MSG = (
    "The s3fs library is required to read and write from S3.\n"
    "Install via pip:\n"
    "    pip install s3fs\n"
    "Install via conda:\n"
    "    conda install s3fs"
)
