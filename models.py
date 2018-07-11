#/usr/bin/env python3

import torch
import sys
import os
import numpy as np
import collections
import re

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# finds filename for mask corresponding to a mix spec
def mix_name_to_source_mag_spec_name(filename, index):
  split = os.path.basename(filename).split('-')
  if index == 1:
    return filename.replace("mix","s1")
  else:
    return filename.replace("mix","s2")

# Define collating function (that constructs packed sequences from a batch)
def sequence_collate(batch):
  elem_type = type(batch[0])
  if isinstance(batch[0], collections.Mapping):
    return {key: sequence_collate([d[key] for d in batch]) for key in batch[0]}
  elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
      and elem_type.__name__ != 'string_':
    elem = batch[0]
    if elem_type.__name__ == 'ndarray':
      # array of string classes and object
      if re.search('[SaUO]', elem.dtype.str) is not None:
        raise TypeError(error_msg.format(elem.dtype))

      sorted_batch = sorted(batch, key=lambda x: len(x), reverse=True)
      return pack_sequence([(torch.from_numpy(b)).float() for b in sorted_batch])
    if elem.shape == ():  # scalars
      py_type = float if elem.dtype.name.startswith('float') else int
      return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
  else:
    return torch.utils.data.dataloader.default_collate(batch)

# Define dataset
class WSJ0_2mixDataset(Dataset):

  def __init__(self, filelist):
    self.list = [line.rstrip('\n') for line in open(filelist)]

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_mag_spec = np.load(self.list[idx]).transpose()
    source_mag_spec_1 = np.load(mix_name_to_source_mag_spec_name(self.list[idx], 1)).transpose()
    source_mag_spec_2 = np.load(mix_name_to_source_mag_spec_name(self.list[idx], 2)).transpose()

    permute_1 = np.concatenate((source_mag_spec_1, source_mag_spec_2), axis=1)
    permute_2 = np.concatenate((source_mag_spec_2, source_mag_spec_1), axis=1)

    sample = {'mix': mix_mag_spec, 'permute1': permute_1, 'permute2': permute_2}
    return sample

# define nnet
class EnhBLSTM(nn.Module):
  def __init__(self, batch_size, gpuid):
    super(EnhBLSTM, self).__init__()

    self.batch_size = batch_size
    self.gpuid = gpuid

    self.blstm = nn.LSTM(257, 600, num_layers=2, bidirectional=True)

    self.lin = nn.Linear(600*2, 514)

    self.bn2 = nn.BatchNorm1d(600*2)

    self.hidden = self.init_hidden()

  def init_hidden(self):
    if self.gpuid > -1:
      return (torch.randn(2*2, self.batch_size, 600).cuda(),
              torch.randn(2*2, self.batch_size, 600).cuda())
    else:
      return (torch.randn(2*2, self.batch_size, 600),
              torch.randn(2*2, self.batch_size, 600))

  def forward(self, x):
    x, self.hidden = self.blstm(x, self.hidden)
    x, lens = pad_packed_sequence(x, batch_first=True)
    x = self.bn2(x.permute(0,2,1).contiguous()).permute(0,2,1)
    x = self.lin(x)
    x = F.sigmoid(x)
    return x

