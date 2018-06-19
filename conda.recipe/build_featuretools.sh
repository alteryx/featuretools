#!/bin/bash
cd /featuretools
yes | anaconda login --username=$1 --password=$2
conda build .
for build_v in $(conda build . --output); do anaconda upload --user featuretools $build_v; done
