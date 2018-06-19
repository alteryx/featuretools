FROM continuumio/miniconda3

RUN apt-get update
RUN yes | conda install anaconda-client conda-build
RUN conda config --set anaconda_upload no
