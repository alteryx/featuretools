def smart_open_check():
    try:
        from smart_open import open
    except ImportError:
        raise ImportError("Please install the smart_open library")


def s3fs_check():
    try:
        import s3fs
    except ImportError:
        raise ImportError("Please install the s3fs library")


def use_smartopen_es(file_path, path, transport_params=None, read=True):
    smart_open_check()
    if read:
        with open(path, "rb", transport_params=transport_params) as fin:
            with open(file_path, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
    else:
        with open(file_path, 'rb') as fin:
            with open(path, 'wb', transport_params=transport_params) as fout:
                shutil.copyfileobj(fin, fout)


def use_s3fs_es(file_path, path, read=True):
    s3fs_check()
    s3 = s3fs.S3FileSystem(anon=True)
    if read:
        s3.get(path, file_path)
    else:
        s3.put(file_path, path)


def use_smartopen_features(path, features_dict=None, transport_params=None, read=True):
    smart_open_check()
    if read:
        with open(path, 'r', encoding='utf-8', transport_params=transport_params) as f:
            features_dict = json.load(f)
            return features_dict
    else:
        with open(path, "w", transport_params=transport_params) as f:
            json.dump(features_dict, f)


def use_s3fs_features(file_path, features_dict=None, read=True):
    s3fs_check()
    s3 = s3fs.S3FileSystem(anon=True)
    if read:
        with s3.open(file_path, "r", encoding='utf-8') as f:
            features_dict = json.load(f)
            return features_dict
    else:
        with s3.open(file_path, "w") as f:
            features = json.dumps(features_dict, ensure_ascii=False)
            f.write(features)
