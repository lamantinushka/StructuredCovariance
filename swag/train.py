import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math


def train_simple_lr(model, X, y, params):
    loss = []
    batch_size = params["batch_size"]
    n_batches = (X.shape[0] + batch_size - 1)//batch_size 
    optimizer = torch.optim.Adam(model.parameters(), lr = params["lr"])
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(params["n_epoch"]):
        idxes = np.arange(X.shape[0])
        np.random.shuffle(idxes)
        for i in range(n_batches):
            idx_batch = idxes[i*batch_size: (1+i)*batch_size]
            x_batch, y_batch = X[idx_batch], y[idx_batch]
            logits = model(x_batch, sample = False)
            nll = criterion(logits, y_batch)
            nll.backward()
            optimizer.step()
            loss.append(nll.item())
    return loss

def sample_batch(X, y, batch_size):
    idxes = np.random.choice(len(y), size = batch_size, replace=False) 
    return X[idxes], y[idxes]

def run_swag_lr(model, X, y, params):
    loss = []
    first_moment = model.mu
    second_moment = model.mu**2
    U = torch.zeros(model.mu.shape[0], model.max_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr = params["lr_swag"])
    criterion = nn.BCEWithLogitsLoss()
    N = 1
    C = params["C"]
    for i in range(1, C*params["n_steps_swag"] + 1):
        x_batch, y_batch = sample_batch(X, y, params["batch_size"])
        logits = model(x_batch, sample = False)
        nll = criterion(logits, y_batch)
        nll.backward()
        optimizer.step()
        if i % C == 0:
            first_moment = (first_moment*N + model.mu)/(N+1)
            second_moment = (second_moment*N + model.mu**2)/(N+1)
            N += 1
            loss.append(nll.item())
    U = torch.zeros_like(model.U)
    for i in range(1, C*model.max_rank + 1):
        x_batch, y_batch = sample_batch(X, y, params["batch_size"])
        logits = model(x_batch, sample = False)
        nll = criterion(logits, y_batch)
        nll.backward()
        optimizer.step()
        if i % C == 0:
            first_moment = (first_moment*N + model.mu)/(N+1)
            second_moment = (second_moment*N + model.mu**2)/(N+1)
            U[:, i//C - 1] = model.mu - first_moment
            N += 1
            loss.append(nll.item())
    d = second_moment - first_moment**2
    return first_moment, U, d, loss
