import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from slang.utils import *

class LogisticRegression(object):
  
    # We suggest b as a column of W
    
    def __init__(self, inp_dim, max_rank = 5):
        self.max_rank = max_rank
        self.mu = torch.nn.Parameter(torch.zeros(inp_dim))
        self.U = torch.nn.Parameter(torch.randn(inp_dim, max_rank))
        self.d = torch.nn.Parameter(torch.ones(inp_dim, dtype = torch.float))
        self.W = None
        self.frozen = False
        
    def _freeze(self):
        self.frozen = True
        
    def _defrost(self):
        self.frozen = False
        
    def __call__(self, x):
        self.input = x
        if self.W is None:
            raise BaseException('Please sample theta first')
        z = x @ self.W 
        return z
    
    def nll_loss(self, x, y, require_grads = False, N = None, random = True, n_samples = 12):
        """ 
        x - tensor [batch_size x feature_size]
        y - [batch_size] tensor, with classes -1 or +1
        """
        preds = torch.zeros(n_samples, x.shape[0])
        if require_grads:
            grads = torch.zeros(n_samples, x.shape[0], x.shape[1])
        if random:
            for i in range(n_samples):
                self.W = self.sample()
                z = self(x)
                preds[i] = z
                if require_grads:
                    # batch_size x feature_size matrix
                    grads[i] = -torch.mul(x.t(), (y * torch.sigmoid(-z * y))).t()
                    
            nll = -torch.log(torch.sigmoid(z * y.view(1, -1))).mean(0)
            if require_grads:
                self.grads = grads.mean(0)
        else:
            self.W = self.mu
            z = self(x)
            nll = -torch.log(torch.sigmoid(z * y))
        return nll.mean()  
      
    def sample(self):
        if self.frozen:
            y = self.W
        else:
            y = fast_sample(self.mu, self.U, self.d)
        return y

class LogisticRegressionFull(object):
  
    # We suggest b as a column of W
    
    def __init__(self, inp_dim, max_rank = 5):
        self.max_rank = max_rank
        self.mu = torch.nn.Parameter(torch.zeros(inp_dim))
        self.Sigma = torch.nn.Parameter(torch.eye(inp_dim))
        self.W = None
        self.frozen = False
        
    def _freeze(self):
        self.frozen = True
        
    def _defrost(self):
        self.frozen = False
        
    def __call__(self, x):
        self.input = x
        if self.W is None:
            raise BaseException('Please sample theta first')
        z = x @ self.W 
        return z
    
    def nll_loss(self, x, y, require_grads = False, N = None, random = True, n_samples = 12):
        """ 
        x - tensor [batch_size x feature_size]
        y - [batch_size] tensor, with classes -1 or +1
        """
        preds = torch.zeros(n_samples, x.shape[0])
        if require_grads:
            grads = torch.zeros(n_samples, x.shape[0], x.shape[1])
            hess = torch.zeros(n_samples, x.shape[1], x.shape[1])
        if random:
            for i in range(n_samples):
                self.W = self.sample()
                z = self(x)
                preds[i] = z
                if require_grads:
                    # batch_size x feature_size matrix
                    grads[i] = -torch.mul(x.t(), (y * torch.sigmoid(-z * y))).t()
                    hess[i] = N/x.shape[0] * grads[i].t() @ grads[i]
            nll = -torch.log(torch.sigmoid(z * y.view(1, -1))).mean(0)
            if require_grads:
                self.grads = grads.mean(0)
                self.hess = hess.mean(0)
        else:
            self.W = self.mu
            z = self(x)
            nll = -torch.log(torch.sigmoid(z * y))
        return nll.mean()
      
    def sample(self):
        if self.frozen:
            y = self.W
        else:
            y = sample_full(self.mu, self.Sigma)
        return y

class MNISTBnn(nn.Module):
  
    def __init__(self, hidden_size = 10, max_rank = 8):
        super(MNISTBnn, self).__init__()
        inp_dim = 14
        self.max_rank = max_rank
        self.out_dim = 2
        self.shapes = [(inp_dim + 1, hidden_size), 
                       (hidden_size + 1, hidden_size),
                       (hidden_size + 1, self.out_dim)]
        self.sizes = [sh[0]*sh[1] for sh in self.shapes]
        
        self.W1 = torch.randn(*self.shapes[0])
        self.W2 = torch.randn(*self.shapes[1])
        self.W3 = torch.randn(*self.shapes[2])
        
        size = sum(self.sizes)
        
        self.mu = torch.nn.Parameter(torch.zeros(size))
        self.U = torch.nn.Parameter(torch.randn(size, max_rank))
        self.d = torch.nn.Parameter(torch.ones(size))
        self.w = None
        self.frozen = False
        
    def _freeze(self):
        self.frozen = True
        
    def _defrost(self):
        self.frozen = False
        
    def forward(self, x):
        z = torch.cat([x, torch.ones(x.shape[0], 1, dtype = torch.float)], 
                      dim = -1) @ self.W1
        z = torch.cat([torch.relu(z), torch.ones(z.shape[0], 1, dtype = torch.float)], 
                      dim = -1) @ self.W2
        logits = torch.cat([torch.relu(z), torch.ones(z.shape[0], 1, dtype = torch.float)], 
                           dim = -1) @ self.W3
        return logits
      
    def sample(self):
        if self.frozen:
            pass
        else:
            w = fast_sample(self.mu, self.U, self.d)
            self.W1 = w[:self.sizes[0]].view(*self.shapes[0])
            self.W2 = w[self.sizes[0]:-self.sizes[2]].view(*self.shapes[1])
            self.W3 = w[-self.sizes[2]:].view(*self.shapes[2])
            return w
          
    def nll_loss(self, x, y, require_grads = False, n_samples = 4, N = None):
        """ 
        x - tensor [batch_size x feature_size]
        y - one-hot-encoded tensor [batch_size x n_classes]
        """
        preds = torch.zeros(n_samples, x.shape[0], self.out_dim)
        if not require_grads:
            for i in range(n_samples):
                w = self.sample()
                logits = self(x)
                preds[i] = logits.log_softmax(-1)
            log_probs = preds.mean(0)   
            return -(log_probs * y).sum()/x.shape[0]
          
          
        grads = torch.zeros(n_samples, x.shape[0], self.mu.shape[0])
        for i in range(n_samples):
            w = self.sample()
            
            z1 = torch.cat([x, torch.ones(x.shape[0], 1, dtype = torch.float)], dim = -1)
            y1 = z1 @ self.W1
            z2 = torch.cat([torch.relu(y1), torch.ones(y1.shape[0], 1, dtype = torch.float)], dim = -1)
            y2 = z2 @ self.W2
            z3 = torch.cat([torch.relu(y2), torch.ones(y2.shape[0], 1, dtype = torch.float)], dim = -1)
            logits = z3 @ self.W3
            preds[i] = logits.log_softmax(-1)
                
            dnll_dlogits = logits.softmax(-1) - y
                
            dW3 = torch.einsum('ab,ac->abc', z3, dnll_dlogits)
            dz3 = dnll_dlogits @ self.W3.t()
            dy2 = dz3[:, :-1]
            dy2[y2 < 0] = 0
                
            dW2 = torch.einsum('ab,ac->abc', z2, dy2)
            dz2 = dy2 @ self.W2.t()
            dy1 = dz2[:, :-1]
            dy1[y1 < 0] = 0
                
            dW1 = torch.einsum('ab,ac->abc', z1, dy1)
            grads[i][:, :self.sizes[0]] = dW1.view(x.shape[0], -1)
            grads[i][:, self.sizes[0]:-self.sizes[2]] = dW2.view(x.shape[0], -1)
            grads[i][:, -self.sizes[2]:] = dW3.view(x.shape[0], -1)
                
        self.grads = grads.mean(0)
        log_probs = preds.mean(0)  
        return -(log_probs * y).sum()/x.shape[0]