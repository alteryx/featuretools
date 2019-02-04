from featuretools import config


def test_get_default_config_does_not_change():
    old_config = config.get_all()

    key = "primitive_data_folder"
    value = "This is an example string"
    config.set({key: value})
    config.set_to_default()

    assert config.get(key) != value

    config.set(old_config)


def test_set_and_get_config():

    key = "primitive_data_folder"
    old_value = config.get(key)
    value = "This is an example string"

    config.set({key: value})
    assert config.get(key) == value

    config.set({key: old_value})


def test_get_all():
    assert config.get_all() == config._data
