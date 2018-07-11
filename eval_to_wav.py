import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/mmaciej2/tools/python/greg_sell/"))
import audio_tools as at

# Usage: eval_to_wav.py <mix_spec_dir> <mask_dir_in> <wav_dir_out>

fftdim = 512
step = 128
fs = 8000

wlen = 1.0*fftdim/fs
overlap = int(fftdim/step)

def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == os.errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

for file in os.listdir(sys.argv[1]):
  Y = np.load(os.path.join(sys.argv[1], file))
  for source in ['s1', 's2']:
    mkdir_p(os.path.join(sys.argv[3], source))
    mask_name = os.path.join(sys.argv[2], source, file)
    wav_name = os.path.join(sys.argv[3], source, file.replace('.npy','.wav'))
    mask = np.load(mask_name)
    S = np.multiply(Y, mask)
    s = at.inverse_specgram(S,  winlen=wlen, overlap_add=overlap)
    at.writewav(s, fs, wav_name)
