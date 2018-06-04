#!/bin/bash
cd /featuretools
conda build .
yes | anaconda login --username=$1 --password=$2
for build_v in $(conda build . --output); do anaconda upload --user featuretools $build_v; done
