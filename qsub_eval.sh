#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M mmaciej2@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=b1[123456789]*|c*|b2*,ram_free=8G,mem_free=8G,h_rt=72:00:00
#$ -r no
set -e

if [ "$#" -ne 3 ]; then
  echo "wrong number of arguments"
  echo "usage:"
  echo "$0 <model> <filelist> <out_name>"
  exit
fi

model=$1
filelist=$2
dirout=$3

device=`free-gpu`

outdir=/export/${HOSTNAME}/mmaciej2/$dirout
mkdir -p $outdir/s1 $outdir/s2

echo "Working on machine $HOSTNAME"

python3 eval_qsub.py $device $model $filelist $outdir
