import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for i in range(1,len(sys.argv)):
  mask = np.absolute(np.load(sys.argv[i]))
  plt.imshow(np.flipud(np.amax(mask)-mask), cmap='gray')
  plt.savefig(sys.argv[i].replace(".npy",".png"))
  plt.gcf().clear()
