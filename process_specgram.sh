#!/bin/bash

model_in=$1
wav_in=$2
out_dir=$3

tools_dir=/export/a15/mmaciej2/enhancement_separation/rsh_net/tools/

mkdir -p $out_dir
cp $wav_in $out_dir
python $tools_dir/generate_specgram.py $wav_in $out_dir
python3 generate_mask.py $model_in $out_dir/mag_specgram.npy $out_dir
python $tools_dir/specgram_and_masks_to_wav.py $out_dir/specgram.npy $out_dir/mask1.npy $out_dir/mask2.npy $out_dir
python plot_masks.py $out_dir/*.npy
