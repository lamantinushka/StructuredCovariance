import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math


class SimpleLR(nn.Module):
    def __init__(self, n_features, max_rank = 10):
        super(SimpleLR, self).__init__()
        self.max_rank = max_rank
        self.mu = torch.nn.Parameter(torch.randn(n_features))
        self.U = torch.nn.Parameter(torch.randn(n_features, max_rank))
        self.d = torch.nn.Parameter(torch.randn(n_features))
        self.frozen = True
        self.W = torch.randn(n_features)
        
    def _freeze(self):
        self.frozen = True
        
    def _defrost(self):
        self.frozen = False
        
    def forward(self, x, sample = True):
        if sample:
            self.W = self.sample()
            return x @ self.W
        return x @ self.mu
          
    def sample(self):
        if self.frozen:
            return self.W
        k = 1/(math.sqrt(2*(self.max_rank - 1)))
        non_diag = k*self.U @ torch.randn(self.max_rank)
        diag = torch.sqrt(self.d) * torch.randn(1)/math.sqrt(2)
        return self.mu + diag + non_diag
      
    def nll_loss(self, X, y, n_samples = 5):
        probs = torch.zeros(X.shape[0])
        self._defrost()
        y[y == 0] = -1
        for i in range(n_samples):
            logits = self(X)
            logits *= y
            probs += logits.sigmoid().log()
        probs /= n_samples
        return -probs.mean()