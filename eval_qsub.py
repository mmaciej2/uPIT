#!/usr/bin/env python3

import torch
import sys
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models as m

gpuid = int(sys.argv[1])

#batch = 100
batch = 1

filelist = sys.argv[3]

if gpuid > -1:
  print("Using GPU", gpuid)
  torch.cuda.set_device(gpuid)
  tmp = torch.ByteTensor([0])
  tmp.cuda()


class WSJ0_2mixTestset(Dataset):

  def __init__(self, filelist):
    self.list = [line.rstrip('\n') for line in open(filelist)]

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mix_mag_spec = np.load(self.list[idx]).transpose()

    sample = {'mix': mix_mag_spec, 'name': os.path.basename(self.list[idx])}
    return sample


dataset = WSJ0_2mixTestset(filelist)
dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=m.sequence_collate)

model = m.EnhBLSTM(batch, gpuid)
if gpuid > -1:
  model.cuda()
model.load_state_dict(torch.load(sys.argv[2], map_location=lambda storage, loc: storage.cuda()))


with torch.no_grad():
  for i_batch, sample_batch in enumerate(dataloader):
    if gpuid > -1:
      mix = sample_batch['mix'].cuda()
    name = sample_batch['name']

    model.zero_grad()
    model.hidden = model.init_hidden()
    mask_out = model(mix)

    for i in range(len(name)):
      mask = mask_out[i].cpu().numpy()
      mask = mask.transpose()
      np.save(os.path.join(sys.argv[4], 's1', name[i]), mask[0:257])
      np.save(os.path.join(sys.argv[4], 's2', name[i]), mask[257:514])
