#!/bin/bash

# Clone featuretools repo
git clone https://github.com/Featuretools/featuretools.git /home/circleci/featuretools
# Checkout specified commit
cd /home/circleci/featuretools
git checkout "${1}"
# Remove build artifacts
rm -rf featuretools/.eggs/ rm -rf featuretools/dist/ rm -rf featuretools/build/
# Create distributions
python setup.py sdist bdist_wheel
# Install twine, module used to upload to pypi
pip install --user twine
# Upload to pypi or testpypi
# To upload to testpypi, run ./upload.sh testpypi instead of ./upload.sh
python -m twine upload dist/* -r "${2:-pypi}"
