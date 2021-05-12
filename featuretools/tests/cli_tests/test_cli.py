import subprocess


def test_info():
    subprocess.check_output(['featuretools', 'info'])


def test_list_primitives():
    subprocess.check_output(['featuretools', 'list-primitives'])
