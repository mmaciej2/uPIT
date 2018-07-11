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

import models

class singleDataset(Dataset):

  def __init__(self, filename):
    self.list = [filename]

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):
    mag_spec = np.load(self.list[idx]).transpose()
    return mag_spec

dataset = singleDataset(sys.argv[2])
dataloader = DataLoader(dataset, batch_size=1, collate_fn=models.sequence_collate)

model = models.EnhBLSTM(1, -1)
model.load_state_dict(torch.load(sys.argv[1], map_location=lambda storage, loc: storage))

with torch.no_grad():
  for i_batch, sample_batch in enumerate(dataloader):
    model.zero_grad()
    model.hidden = model.init_hidden()
    mask_out = model(sample_batch)
    mask = mask_out.numpy()
    mask = np.squeeze(mask, axis=0)
    mask = mask.transpose()
    np.save(sys.argv[3]+'/mask1.npy', mask[0:257])
    np.save(sys.argv[3]+'/mask2.npy', mask[257:514])
