#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M mmaciej2@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=b1[123456789]*|c*|b2*,ram_free=8G,mem_free=8G,h_rt=72:00:00
#$ -r no
set -e
device=`free-gpu`

mkdir -p intermediate_models plots

python3 train_qsub.py $device
