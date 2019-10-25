import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from slgd.utils import *
from slgd.models import LogisticRegressionLangevin, LogisticRegressionFull, MNISTBnn

def make_optimization_step(model, x, y, params, alpha = None, beta = None):
    if beta is None:
        beta = params["beta"]
    lmbd = params["lambda"]
    delta = 1 - beta
    n_samples = params["n_samples_train"]
    nll = model.nll_loss(x, y, require_grads = True, n_samples = n_samples)
    if isinstance(model, LogisticRegression) or isinstance(model, MNISTBnn):
        if alpha is None:
            alpha = params["alpha"]
        n_iter = params["n_iter"]
        max_rank = model.max_rank
        V = fast_eig(model, max_rank = max_rank, beta = beta, n_iter = n_iter)
        delta_d = delta*(model.U**2).sum(dim = 1) + beta*(model.grads**2).sum(dim = 0) - (V**2).sum(dim = 1)
        delta_d[delta_d < 0] = 0
        model.U = torch.nn.Parameter(V)
        model.d = torch.nn.Parameter(beta*lmbd + delta*model.d + delta_d)
        cum_grads = model.grads.sum(axis = 0) + lmbd*model.mu
        delta_mu = fast_inverse(cum_grads.view(-1, 1), model.U, model.d)
        model.mu = torch.nn.Parameter(model.mu - alpha*delta_mu)
    elif isinstance(model, LogisticRegressionFull):
        Sinv = torch.inverse(model.Sigma)
        G = model.grads.t()
        Hess = model.hess
        Sinv = delta*Sinv + beta*(lmbd*torch.eye(model.mu.shape[0]) + Hess)
        model.Sigma = torch.nn.Parameter(torch.inverse(Sinv))
        model.mu = torch.nn.Parameter(model.mu - alpha * model.Sigma @ (lmbd* model.mu + G.mean(axis = -1)*N))
    elif isinstance(model, LogisticRegressionLangevin):
        G = model.grads.t()
        model.V += ((G-G.mean(axis=1)[:, None])@(model.grads-model.grads.mean(axis=0)[None, :]))#*X.shape[0]*(1/params["batch_size"])
        model.T += 1
        model.Sigma = torch.pinverse(model.V/model.T)
        model.mu = torch.nn.Parameter(model.mu - alpha * model.grads.sum(axis=0) - 2*(alpha ** 0.5)*(torch.empty(model.mu.shape).normal_(mean=0., std=1.)))
        model.num += alpha*model.mu
        model.denom += alpha
    return nll
  

def train(model, X, y, params):
    loss = []
    test_nll = []
    batch_size = params["batch_size"]
    n_batches = (X.shape[0] + batch_size - 1)//batch_size 
    t = 0
    if isinstance(model, LogisticRegressionLangevin):
        t=30000
    alpha0 = params["alpha"]
    beta0 = params["beta"]
    pow = params["pow"]
    eval_samples = params["n_samples_eval"]
    if pow is None:
        pow = 0.
    for epoch in range(params["n_epoch"]):
        idxes = np.arange(X.shape[0])
        np.random.shuffle(idxes)
        for i in range(n_batches):
            idx_batch = idxes[i*batch_size: (1+i)*batch_size]
            x_batch, y_batch = X[idx_batch], y[idx_batch]
            alpha = alpha0/(1 + t**pow)
            beta = alpha
            nll = make_optimization_step(model, x_batch, y_batch, params, alpha = alpha, beta = beta)
            t += 1
            loss.append(nll.item())
        if isinstance(model, LogisticRegressionLangevin):
            model.mu = model.num/model.denom
        if epoch % 10 == 0:
            test_nll.append(model.nll_loss(X_test, y_test, 
                                           require_grads = False, 
                                           n_samples = eval_samples).item())
    return loss, test_nll
  

def train(model, X, y, params, X_test = None, y_test = None):
    loss = []
    if X_test is not None:
        test_nll = []
    batch_size = params["batch_size"]
    n_batches = (X.shape[0] + batch_size - 1)//batch_size 
    t = 0
    alpha0 = params["alpha"]
    beta0 = params["beta"]
    pow = params["pow"]
    eval_samples = params["n_samples_eval"]
    if pow is None:
        pow = 0.
    for epoch in range(params["n_epoch"]):
        idxes = np.arange(X.shape[0])
        np.random.shuffle(idxes)
        for i in range(n_batches):
            idx_batch = idxes[i*batch_size: (1+i)*batch_size]
            x_batch, y_batch = X[idx_batch], y[idx_batch]
            alpha = alpha0/(1 + t**pow)
            beta = beta0/(1 + t**pow)
            nll = make_optimization_step(model, x_batch, y_batch, params, alpha = alpha, beta = beta)
            t += 1
            loss.append(nll.item())
        if epoch % 20 == 0 and (X_test is not None):
            test_nll.append(model.nll_loss(X_test, y_test, 
                                           require_grads = False, 
                                           n_samples = eval_samples).item())
    return loss, test_nll
