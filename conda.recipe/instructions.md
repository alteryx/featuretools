# To upload prebuilt Mac OS packages (assuming you are running Max OS)

Install anaconda/miniconda

Install the anaconda-client and conda-build packages
`conda install anaconda-client conda-build`

Turn automatic uploads off
`conda config --set anaconda_upload no`

Build the featuretools conda package (run this from the root featuretools directory)
`conda build .`

Login in to anaconda (this will prompt you for your anaconda username & password)
`anaconda login`

Upload the build for each Python version to Anaconda
`for build_v in $(conda build . --output); do anaconda upload --user featuretools $build_v; done`

# To upload prebuilt Linux packages via Docker

This will prompt you for your anaconda password
`cd conda.recipe`
`python build_and_upload_conda_package.py --username anaconda_username`
