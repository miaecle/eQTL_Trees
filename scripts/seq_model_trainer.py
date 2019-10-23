#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:48:03 2018

@author: zqwu
"""


import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class BatchPairDataset(Dataset):
  def __init__(self, 
               batch_inds, 
               data):
    self.batch_inds = batch_inds
    self.X, self.inter, self.y, self.dat = data
    self.length = len(self.batch_inds)
  
  def __len__(self):
    return self.length

  def __getitem__(self, index):
    if index >= self.length:
      raise IndexError
    inds = self.batch_inds[index]
    y = np.array([self.y[i] for i in inds]) # batch_size
    other_inputs = np.array([self.X[i] for i in inds]) # batch_size * feat_dim
    seq_inputs = [self.inter[i] for i in inds]

    batch_length = max([seq_input.shape[0] for seq_input in seq_inputs])
    for i, seq_input in enumerate(seq_inputs):
      seq_inputs[i] = np.pad(seq_input, ((0, batch_length - seq_input.shape[0]), (0, 0)), 'constant')
    seq_inputs = np.stack(seq_inputs, 1)
    
    return Variable(t.from_numpy(seq_inputs).float()), \
           Variable(t.from_numpy(other_inputs).float()), \
           Variable(t.from_numpy(y).long())
    
class Trainer(object):
  def __init__(self, 
               net, 
               opt, 
               criterion):
    self.net = net
    self.seq_input_dim = self.net.seq_input_dim
    self.other_input_dim = self.net.other_input_dim
    self.opt = opt
    self.criterion = criterion
    if self.opt.gpu:
      self.net = self.net.cuda()
  
  def assemble_batch(self, 
                     data,
                     batch_size=None,
                     sort=True,
                     padding=False):
    if batch_size is None:
      batch_size = self.opt.batch_size
    (X, inter, y, dat) = data
    if sort:
      # Sort by length
      lengths = [inter_feat.shape[0] for inter_feat in inter]
      order = np.argsort(lengths)
    else:
      order = np.arange(len(dat))

    # Assemble samples with similar lengths to a batch
    batch_inds = []
    for i in range(int(np.ceil(len(dat)/float(batch_size)))):
      inds = order[i * batch_size:min((i+1) * batch_size, len(dat))]
      if padding and len(inds) < batch_size:
        inds.extend([inds[0]] * (batch_size - len(inds)))
      batch_inds.append(inds)
    return batch_inds
 
  def train(self, train_data, n_epochs=None, **kwargs):
    self.net.train()
    self.run_model(train_data, train=True, n_epochs=n_epochs, **kwargs)
    return
  
  def display_loss(self, train_data, **kwargs):
    self.run_model(train_data, train=False, n_epochs=1, **kwargs)
    return
    
  def run_model(self, data, train=False, n_epochs=None, **kwargs):
    if train:
      optimizer = Adam(self.net.parameters(),
                       lr=self.opt.lr,
                       betas=(.9, .999))
      self.net.zero_grad()
      epochs = self.opt.max_epoch
    else:
      epochs = 1
    if n_epochs is not None:
      epochs = n_epochs

    (X, inter, y, dat) = data
    n_points = len(dat)
    batch_inds = self.assemble_batch(data)
    
    dataset = BatchPairDataset(batch_inds, data)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for epoch in range(epochs):      
      loss = 0
      print ('start epoch {epoch}'.format(epoch=epoch))
      for batch in data_loader:
        for i, item in enumerate(batch):
          if self.opt.gpu:
            batch[i] = item[0].cuda()
          else:
            batch[i] = item[0]
        seq_input, other_input, label = batch
        output = self.net(seq_input, other_input, label.shape[0])
        error = self.criterion(output, label)
        loss += error
        error.backward()
        if train:
          optimizer.step()
          self.net.zero_grad()
      print ('epoch {epoch} loss: {loss}'.format(epoch=epoch, loss=loss.item()/n_points))
  
  
  def predict(self, test_data):
    (X, inter, y, dat) = test_data
    test_batches = self.assemble_batch(test_data, batch_size=1, sort=False)
    dataset = BatchPairDataset(test_batches, test_data)
    preds = []
    for batch in dataset:
      sample = []
      for i, item in enumerate(batch):
        if self.opt.gpu:
          sample.append(item.cuda())
        else:
          sample.append(item)
          
      seq_input, other_input, label = sample
      preds.append(self.net.predict(seq_input, other_input, 1))
    return preds
  
  def save(self, path):
    t.save(self.net.state_dict(), path)
  
  def load(self, path):
    s_dict = t.load(path, map_location=lambda storage, loc: storage)
    self.net.load_state_dict(s_dict)
 
  def set_seed(self, seed):
    t.manual_seed(seed)
    if self.opt.gpu:
      t.cuda.manual_seed_all(seed)

class SimpleTrainer(Trainer):
  def __init__(self, 
               net, 
               opt, 
               criterion):
    self.net = net
    self.opt = opt
    self.criterion = criterion
    if self.opt.gpu:
      self.net = self.net.cuda()
  
  def assemble_batch(self, 
                     data,
                     batch_size=None,
                     padding=False):
    if batch_size is None:
      batch_size = self.opt.batch_size
    (X, y) = data
    
    n_points = len(X)
    order = np.arange(n_points)

    # Assemble samples with similar lengths to a batch
    batch_Xs = []
    batch_ys = []
    for i in range(int(np.ceil(n_points/float(batch_size)))):
      inds = order[i * batch_size:min((i+1) * batch_size, n_points)]
      if padding and len(inds) < batch_size:
        inds.extend([inds[0]] * (batch_size - len(inds)))
      batch_Xs.append(t.from_numpy(X[inds]).float())
      batch_ys.append(t.from_numpy(y[inds]).long())
    return batch_Xs, batch_ys
    
  def run_model(self, data, train=False, n_epochs=None, **kwargs):
    if train:
      optimizer = Adam(self.net.parameters(),
                       lr=self.opt.lr,
                       betas=(.9, .999))
      self.net.zero_grad()
      epochs = self.opt.max_epoch
    else:
      epochs = 1
    if n_epochs is not None:
      epochs = n_epochs

    n_points = len(data[0])
    batch_Xs, batch_ys = self.assemble_batch(data)
    
    for epoch in range(epochs):      
      loss = 0
      print ('start epoch {epoch}'.format(epoch=epoch))
      for X, y in zip(batch_Xs, batch_ys):
        if self.opt.gpu:
          X = X.cuda()
          y = y.cuda()
        output = self.net(X)
        error = self.criterion(output, y)
        loss += error
        error.backward()
        if train:
          optimizer.step()
          self.net.zero_grad()
      print ('epoch {epoch} loss: {loss}'.format(epoch=epoch, loss=loss.item()/n_points))
  
  def predict(self, test_data):
    batch_Xs, batch_ys = self.assemble_batch(test_data, padding=False)
    preds = []
    for X, y in zip(batch_Xs, batch_ys):
      if self.opt.gpu:
        X = X.cuda()
        y = y.cuda()
      preds.append(self.net.predict(X))
    return preds