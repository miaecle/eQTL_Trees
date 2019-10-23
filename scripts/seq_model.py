#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:41:57 2018

@author: zqwu
"""

from torch import nn
import torch as t
import torch.nn.functional as F
import numpy as np

class SeqModel(nn.Module):
  def __init__(self,
               seq_input_dim=17,
               other_input_dim=320,
               hidden_dim_lstm=128,
               hidden_dim_attention=32,
               hidden_dim_fc=256,
               n_lstm_layers=2,
               n_attention_heads=8,
               dropout_rate=0.5,
               gpu=True):
    super(SeqModel, self).__init__()
    self.seq_input_dim = seq_input_dim
    self.other_input_dim = other_input_dim
    self.hidden_dim_lstm = hidden_dim_lstm
    self.hidden_dim_attention = hidden_dim_attention
    self.hidden_dim_fc = hidden_dim_fc
    self.n_lstm_layers = n_lstm_layers
    self.n_attention_heads = n_attention_heads
    self.dropout_rate = dropout_rate
    self.gpu = gpu

    self.lstm_module = BiLSTM(input_dim=self.seq_input_dim,
                              hidden_dim=self.hidden_dim_lstm,
                              n_layers=self.n_lstm_layers,
                              gpu=self.gpu)
    self.att_module = MultiheadAttention(Q_dim=self.hidden_dim_lstm,
                                         V_dim=self.hidden_dim_lstm,
                                         head_dim=self.hidden_dim_attention,
                                         n_heads=self.n_attention_heads)
    
    self.fc = nn.Sequential(
        nn.Linear(self.hidden_dim_lstm + self.other_input_dim, self.hidden_dim_fc),
        nn.ReLU(True),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.hidden_dim_fc, self.hidden_dim_fc),
        nn.ReLU(True),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.hidden_dim_fc, self.hidden_dim_fc),
        nn.ReLU(True),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.hidden_dim_fc, 2))

  def forward(self, seq_input, other_input, batch_size=1):
    """
    seq_input: seq_len * batch_size * feat_dim1
    other_input: batch_size * feat_dim2
    """
    lstm_outs = self.lstm_module(seq_input, batch_size=batch_size)
    attention_outs = self.att_module(sequence=lstm_outs)
    
    seq_outs = attention_outs.max(0)[0] # Discard the argmax indices
    
    outs = self.fc(t.cat([seq_outs, other_input], 1))
    return F.log_softmax(outs, 1)

  def predict(self, seq_input, other_input, batch_size=1):
    if self.training:
      self.eval()
    assert batch_size == 1
    output = t.exp(self.forward(seq_input, other_input, 1))
    output = output.cpu()
    output = output.data.numpy()
    return output

class TransformerModel(nn.Module):
  def __init__(self,
               seq_input_dim=17,
               other_input_dim=320,
               hidden_dim_attention=16,
               hidden_dim_fc=128,
               n_attention_heads=8,
               n_attention_layers=2,
               dropout_rate=0.5,
               gpu=True):
    super(TransformerModel, self).__init__()
    self.seq_input_dim = seq_input_dim
    self.other_input_dim = other_input_dim
    self.hidden_dim_attention = hidden_dim_attention
    self.hidden_dim_fc = hidden_dim_fc
    self.n_attention_heads = n_attention_heads
    self.n_attention_layers = n_attention_layers
    self.dropout_rate = dropout_rate
    self.gpu = gpu

    
    n_dim_att = self.hidden_dim_attention * self.n_attention_heads
    self.intro_linear = nn.Linear(self.seq_input_dim, n_dim_att)
    
    self.att_modules = []
    self.bn_modules1 = []
    self.feed_forwards = []
    self.bn_modules2 = []
    for i in range(self.n_attention_layers):
      self.att_modules.append(MultiheadAttention(Q_dim=n_dim_att,
                                                 V_dim=n_dim_att,
                                                 head_dim=self.hidden_dim_attention,
                                                 n_heads=self.n_attention_heads))
      self.bn_modules1.append(nn.BatchNorm1d(n_dim_att))
      self.feed_forwards.append(nn.Sequential(nn.Linear(n_dim_att, n_dim_att),
                                              nn.ReLU(True),
                                              nn.Linear(n_dim_att, n_dim_att)))
      self.bn_modules2.append(nn.BatchNorm1d(n_dim_att))
      
    self.att_modules = nn.ModuleList(self.att_modules)
    self.bn_modules1 = nn.ModuleList(self.bn_modules1)
    self.feed_forwards = nn.ModuleList(self.feed_forwards)
    self.bn_modules2 = nn.ModuleList(self.bn_modules2)
    
    self.fc = nn.Sequential(
        nn.Linear(n_dim_att + self.other_input_dim, self.hidden_dim_fc),
        nn.ReLU(True),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.hidden_dim_fc, self.hidden_dim_fc),
        nn.ReLU(True),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.hidden_dim_fc, self.hidden_dim_fc),
        nn.ReLU(True),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.hidden_dim_fc, 2))

  def forward(self, seq_input, other_input, batch_size=1):
    """
    seq_input: seq_len * batch_size * feat_dim1
    other_input: batch_size * feat_dim2
    """
    feat_input = self.intro_linear(seq_input)
    for i in range(self.n_attention_layers):
      attention_out = self.att_modules[i](sequence=feat_input)
      bn_in = (attention_out + feat_input).permute(1, 2, 0).contiguous()
      ff_in = self.bn_modules1[i](bn_in).permute(2, 0, 1)
      ff_out = self.feed_forwards[i](ff_in)
      bn2_in = (ff_in + ff_out).permute(1, 2, 0).contiguous()
      feat_input = self.bn_modules2[i](bn2_in).permute(2, 0, 1)
      
    seq_outs = feat_input.max(0)[0] # Discard the argmax indices
    outs = self.fc(t.cat([seq_outs, other_input], 1))
    return F.log_softmax(outs, 1)

  def predict(self, seq_input, other_input, batch_size=1):
    if self.training:
      self.eval()
    assert batch_size == 1
    output = t.exp(self.forward(seq_input, other_input, 1))
    output = output.cpu()
    output = output.data.numpy()
    return output

class SimpleModel(nn.Module):
  def __init__(self,
               input_dim=477,
               hidden_dim_fc=256,
               dropout_rate=0.1,
               gpu=True):
    super(SimpleModel, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim_fc = hidden_dim_fc
    self.dropout_rate = dropout_rate
    self.gpu = gpu
    self.fc = nn.Sequential(
        nn.Linear(self.input_dim, self.hidden_dim_fc),
        nn.ReLU(True),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.hidden_dim_fc, self.hidden_dim_fc),
        nn.ReLU(True),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.hidden_dim_fc, self.hidden_dim_fc//2),
        nn.ReLU(True),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.hidden_dim_fc//2, 2))

  def forward(self, inputs, **kwargs):
    """
    inputs: batch_size * input_dim
    """
    outs = self.fc(inputs)
    return F.log_softmax(outs, 1)

  def predict(self, inputs, **kwargs):
    if self.training:
      self.eval()
    output = t.exp(self.forward(inputs, **kwargs))
    output = output.cpu()
    output = output.data.numpy()
    return output
    
class BiLSTM(nn.Module):
  def __init__(self, input_dim=256, hidden_dim=128, n_layers=2, gpu=True):
    """ A wrapper of pytorch bi-lstm module """
    super(BiLSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    self.lstm = nn.LSTM(input_dim, self.hidden_dim//2,
                        num_layers=self.n_layers, bidirectional=True)
    self.gpu = gpu
    self.hidden = self._init_hidden()

  def _init_hidden(self, batch_size=1):
    if self.gpu:
      return (t.randn(2*self.n_layers, batch_size, self.hidden_dim//2).cuda(),
              t.randn(2*self.n_layers, batch_size, self.hidden_dim//2).cuda())
    else:
      return (t.randn(2*self.n_layers, batch_size, self.hidden_dim//2),
              t.randn(2*self.n_layers, batch_size, self.hidden_dim//2))

  def forward(self, sequence, batch_size=1):
    # Get the emission scores from the BiLSTM
    # sequence: seq_len * batch_size * feat_dim
    self.hidden = self._init_hidden(batch_size=batch_size)
    inputs = sequence.reshape((-1, batch_size, self.input_dim))
    lstm_out, self.hidden = self.lstm(inputs, self.hidden)
    lstm_out = lstm_out.view(-1, batch_size, self.hidden_dim)
    return lstm_out

class MultiheadAttention(nn.Module):
  def __init__(self, 
               Q_dim=128,
               V_dim=128,
               head_dim=32, 
               n_heads=8):
    """ As described in https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf """
    super(MultiheadAttention, self).__init__()
    self.Q_dim = Q_dim
    self.K_dim = self.Q_dim
    self.V_dim = V_dim
    self.head_dim = head_dim
    self.n_heads = n_heads

    self.K_linears = nn.ModuleList([nn.Linear(self.K_dim, 
                                              self.head_dim) for i in range(self.n_heads)])
    self.Q_linears = nn.ModuleList([nn.Linear(self.Q_dim, 
                                              self.head_dim) for i in range(self.n_heads)])
    self.V_linears = nn.ModuleList([nn.Linear(self.V_dim, 
                                              self.head_dim) for i in range(self.n_heads)])

    self.post_head_linear = nn.Linear(self.head_dim * self.n_heads, self.Q_dim)
    
    self.fc = nn.Sequential(
        nn.Linear(self.Q_dim, self.Q_dim*4),
        nn.ReLU(True),
        nn.Linear(self.Q_dim*4, self.Q_dim*4),
        nn.ReLU(True),
        nn.Linear(self.Q_dim*4, self.Q_dim))

  def forward(self, sequence=None, K_in=None, Q_in=None, V_in=None):
    # query: seq_len_Q * batch_size * Q_dim
    # key:   seq_len_K * batch_size * Q_dim
    # value: seq_len_K * batch_size * V_dim
    outs = []
    if K_in is None:
      K_in = sequence
    if Q_in is None:
      Q_in = sequence
    if V_in is None:
      V_in = sequence
    for i in range(self.n_heads):
      K = self.K_linears[i](K_in.transpose(0, 1))
      Q = self.Q_linears[i](Q_in.transpose(0, 1))
      V = self.V_linears[i](V_in.transpose(0, 1))
      e = t.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.head_dim)
      a = F.softmax(e, dim=2)
      outs.append(t.matmul(a, V))
    
    att_outs = Q_in.transpose(0, 1) + self.post_head_linear(t.cat(outs, 2))
    outs = att_outs + self.fc(att_outs)
    return outs.transpose(0, 1)
