#!/bin/bash

# create conda env
read -rp "Enter environment name (recommended: srl-pas-probing): " env_name
read -rp "Enter python version (recommended: 3.8): " python_version
conda create -yn "$env_name" python="$python_version"
eval "$(conda shell.bash hook)"
conda activate "$env_name"

# install torch
read -rp "Enter cuda version (e.g. '10.2' or 'none' to avoid installing cuda support): " cuda_version
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch=1.10 cpuonly -c pytorch
else
    conda install -y pytorch=1.10 cudatoolkit=$cuda_version -c pytorch -c nvidia
fi

# install python requirements
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+${cuda_version}.html
