#!/usr/bin/env python3

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

import models as m

import quick_plot as qp

if len(sys.argv) == 2:
  gpuid = int(sys.argv[1])
else:
  gpuid = -1


batch = 100

filelist = "filelist_tr.txt"
cv_filelist = "filelist_cv.txt"


if gpuid > -1:
  print("Using GPU", gpuid)
  torch.cuda.set_device(gpuid)
  tmp = torch.ByteTensor([0])
  tmp.cuda()
else:
  print("Using CPU")


print("loading datset")
dataset = m.WSJ0_2mixDataset(filelist)
dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=m.sequence_collate, num_workers=1)

print("initializing model")
model = m.EnhBLSTM(batch, gpuid)
if gpuid > -1:
  model.cuda()
loss_function = nn.MSELoss(reduce=False)
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("using lr=0.001")

torch.save(model.state_dict(), 'intermediate_models/model-init')

print("training")
for epoch in range(200):
  epoch_loss = 0.0
  for i_batch, sample_batch in enumerate(dataloader):
    if gpuid > -1:
      mix = sample_batch['mix'].cuda()
      permute1 = sample_batch['permute1'].cuda()
      permute2 = sample_batch['permute2'].cuda()
    else:
      mix = sample_batch['mix']
      permute1 = sample_batch['permute1']
      permute2 = sample_batch['permute2']

    model.zero_grad()
    model.hidden = model.init_hidden()

    mask_out = model(mix)

    mixes, lens = pad_packed_sequence(mix, batch_first=True)
    permutations1, lens = pad_packed_sequence(permute1, batch_first=True)
    permutations2, lens = pad_packed_sequence(permute2, batch_first=True)
    if gpuid > -1:
      lengths = lens.float().cuda()
    else:
      lengths = lens.float()
    double_mix = torch.cat((mixes, mixes), dim=2)
    masked = mask_out * double_mix
    loss1 = torch.sum(loss_function(masked, permutations1).view(batch, -1), dim=1)
    loss2 = torch.sum(loss_function(masked, permutations2).view(batch, -1), dim=1)
    loss = torch.mean(torch.min(loss1, loss2)/lengths/514/batch)
    epoch_loss += loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()

  if epoch % 5 == 0:
    cv_loss = 0.0
    with torch.no_grad():
      cv_dataset = m.WSJ0_2mixDataset(cv_filelist)
      cv_dataloader = DataLoader(cv_dataset, batch_size=batch, collate_fn=m.sequence_collate)
      for i_batch_cv, sample_batch_cv in enumerate(cv_dataloader):
        if gpuid > -1:
          mix = sample_batch['mix'].cuda()
          permute1 = sample_batch['permute1'].cuda()
          permute2 = sample_batch['permute2'].cuda()
        else:
          mix = sample_batch['mix']
          permute1 = sample_batch['permute1']
          permute2 = sample_batch['permute2']

        model.zero_grad()
        model.hidden = model.init_hidden()

        mask_out = model(mix)

        mixes, lens = pad_packed_sequence(mix, batch_first=True)
        permutations1, lens = pad_packed_sequence(permute1, batch_first=True)
        permutations2, lens = pad_packed_sequence(permute2, batch_first=True)
        if gpuid > -1:
          lengths = lens.float().cuda()
        else:
          lengths = lens.float()
        double_mix = torch.cat((mixes, mixes), dim=2)
        masked = mask_out * double_mix
        loss1 = torch.sum(loss_function(masked, permutations1).view(batch, -1), dim=1)
        loss2 = torch.sum(loss_function(masked, permutations2).view(batch, -1), dim=1)
        loss = torch.mean(torch.min(loss1, loss2)/lengths/514/batch)
        cv_loss += loss
        if i_batch_cv == 0:
          qp.plot(mixes[0].detach().cpu().numpy(), 'plots/epoch'+str(epoch)+'_mix.png')
          qp.plot(masked[0].detach().cpu().numpy(), 'plots/epoch'+str(epoch)+'_masked_mix.png')
          if loss1[0] < loss2[0]:
            qp.plot(permutations1[0].detach().cpu().numpy(), 'plots/epoch'+str(epoch)+'_chosen_permutation.png')
          else:
            qp.plot(permutations2[0].detach().cpu().numpy(), 'plots/epoch'+str(epoch)+'_chosen_permutation.png')
    print("For epoch: "+str(epoch)+" cv set loss is: "+str(cv_loss/50.0))

  print("For epoch: "+str(epoch)+" loss is: "+str(epoch_loss/200.0))
  if epoch % 5 == 0:
    print("Saving model for epoch "+str(epoch))
    torch.save(model.state_dict(), 'intermediate_models/model-'+str(epoch))
  sys.stdout.flush()

torch.save(model.state_dict(), 'model')

