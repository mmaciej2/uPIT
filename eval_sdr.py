import sys
import os
import numpy as np
#from scip.io.wavfile import read
import mir_eval

# Usage: eval_sdr.py <estimated_sources> <actual_sources>

sdrs = []

for file in os.listdir(os.path.join(sys.argv[2], 's1')):
  s1, fs = mir_eval.io.load_wav(os.path.join(sys.argv[1], 's1', file))
  s2, fs = mir_eval.io.load_wav(os.path.join(sys.argv[1], 's2', file))
  source_length = len(s1)
  g1, fs = mir_eval.io.load_wav(os.path.join(sys.argv[2], 's1', file))
  g2, fs = mir_eval.io.load_wav(os.path.join(sys.argv[2], 's2', file))

  gs = (np.c_[g1[0:source_length], g2[0:source_length]]).T
  ss = (np.c_[s1, s2]).T

  sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(gs, ss)

  for value in sdr:
    sdrs.append(value)

sdrs = np.array(sdrs)
outF = open(os.path.join(sys.argv[1], 'SDR_results.txt'), 'w')
outF.write("Mean:\t"+str(np.mean(sdrs))+'\n')
outF.write("Std:\t"+str(np.std(sdrs))+'\n')
outF.write("Max:\t"+str(np.amax(sdrs))+'\n')
outF.write("Min:\t"+str(np.amin(sdrs))+'\n')
outF.close()
