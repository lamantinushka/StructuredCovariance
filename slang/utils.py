import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math

def fast_inverse(g, U, d):
  
  # g - n_features x batch_size 
  # U - n_features x max_rank
  # d - n_features
  
  # output  n_features x batch_size
  
    dInv = 1/d.view(-1, 1)
    I_k = torch.eye(U.shape[1], dtype=U.dtype, device=U.device)
    Kinv = I_k + U.t() @ (dInv * U)
    s1 = U.t() @ (dInv * g)
    s2, _ = torch.solve(s1, Kinv)
    return (dInv*(g - (U @ s2))).view(-1)

def fast_sample(mu, U, d):
  
  # mu, d - n_features
  # U - n_features x max_rank
  
  # output - n_features

    dinv = 1 /(d.view(-1, 1)**(0.5))
    eps = torch.randn_like(dinv)
    V = U * dinv
    M = V.t() @ V
    A = torch.cholesky(M)
    B = torch.cholesky(torch.eye(M.shape[0]) + M)
    Ainv = torch.inverse(A)
    C = Ainv.t() @ (B - torch.eye(B.shape[0])) @ Ainv
    Kinv = C + M
    s1 = V.t() @ eps
    s2, _ = torch.solve(s1, Kinv)
    y = eps * dinv - V @ s2
    return y.view(-1) + mu
  
def sample_full(mu, Sigma):
    dist = torch.distributions.MultivariateNormal(mu, Sigma)
    y = dist.sample()
    return y

def orthogonalize(A):
    (Q, _) = torch.qr(A)
    return Q
  
def svd_thin_matrix(A):
    (e, V) = torch.symeig(A @ A.t(), eigenvectors=True)
    e[e < 0] = 1e-20
    Sigma = torch.sqrt(e)
    SigInv = 1/Sigma 
    SigInv[torch.isnan(SigInv)] = 0
    U = A.t() @ (V * SigInv)
    return U, Sigma, V

def fast_eig(model, N, max_rank=6, n_iter=4, L=None, beta = 0.9):
  
    n = model.U.shape[0]
  
    def f(x):
        U = model.U
        G = model.grads.t()
        f_val = (1 - beta) * U @(U.t() @ x) + beta * G @(G.t() @ x) * N/G.shape[1]
        return f_val    
        
    if L is None:
        L = max_rank+2

    def nystrom(Q, anorm):        
        anorm = .1e-6 * anorm * math.sqrt(1. * n)
        E = f(Q) + anorm * Q
        R = Q.t() @ E
        R = (R + R.t()) / 2
        R = torch.cholesky(R, upper=False) # Cholesky
        (tmp, _) = torch.solve(E.t(), R) # Solve
        V, d, _ = svd_thin_matrix(tmp)
        d = d * d - anorm
        return d, V

    Q = 2*(torch.rand((n, L), dtype=torch.float)-0.5)
    for _ in range(max(0, n_iter-1)):
        Q = orthogonalize(f(Q))
    oldQ = Q
    Q = f(Q)
    anorm = torch.max(torch.norm(Q, dim=0)/torch.norm(oldQ, dim=0))
    Q = orthogonalize(Q)

    d, V = nystrom(Q, anorm)

    (_, idx) = torch.abs(d).sort()
    idx = idx[(L-max_rank):]
    return V[:, idx]*torch.sqrt(abs(d[idx]))

