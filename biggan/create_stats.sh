#!/bin/bash
# This script creates a venv with pytorch to run the extract_stats.py script.
# It removes the venv afterwards.
current_dir=$(pwd)
cd $2
git clone "https://github.com/huggingface/pytorch-pretrained-BigGAN"
mv pytorch-pretrained-BigGAN torch_biggan
# python3 -m venv torch_venv --without-pip --system-site-packages
# source torch_venv/bin/activate
pip3 install -r torch_biggan/requirements.txt
pip3 install numpy scipy
python3 extract_stats.py  --size $1 --dir $2
# deactivate
rm -rf torch_biggan
# rm -rf torch_venv
cd $current_dir
