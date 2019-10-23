import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from slang.utils import *
from slang.models import LogisticRegression, LogisticRegressionFull
from slang.train import *
import argparse
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser()

parser.add_argument("--do_full_model", default = True, type = bool)
parser.add_argument("--task", default = "australian", type = str)
parser.add_argument("--batch_size", default = 64, type = int)
parser.add_argument("--n_epoch", default = 1000, type = int)
parser.add_argument("--beta", default = 0.03, type = float)
parser.add_argument("--alpha", default = 0.03, type = float)
parser.add_argument("--lambda", default = 1., type = float)
parser.add_argument("--max_ranks", nargs='+', default = [1, 5, 10], type = int)
parser.add_argument("--n_iter", default = 4, type = int)
parser.add_argument("--pow", default = 0.51, type = float)
parser.add_argument("--n_samples_train", default = 12, type = int)
parser.add_argument("--n_samples_eval", default = 100, type = int)
args = parser.parse_args()
params = vars(args)


if params["task"] == "australian":
    src = 'data/australian_presplit/australian_scale_'
    
elif params["task"] == "cancer":
    src = 'data/breast_cancer_presplit/breast_cancer_scale_'
    
else:
    print("Sorry, I can do only australian and cancer for now")

X_train = pd.read_csv(src + 'X_tr.csv', delim_whitespace=True)
y_train = pd.read_csv(src + 'y_tr.csv', delim_whitespace=True)
X_test = pd.read_csv(src + 'X_te.csv', delim_whitespace=True)
y_test = pd.read_csv(src + 'y_te.csv', delim_whitespace=True)

X = torch.tensor(X_train.values, dtype = torch.float)
X = torch.cat([X, torch.ones(X.shape[0], 1, dtype = torch.float)], dim = -1)
y = torch.tensor(y_train.values, dtype = torch.float).view(-1)
y[y == 0] = -1
params["N"] = X.shape[0]
    
X_test = torch.tensor(X_test.values, dtype = torch.float)
X_test = torch.cat([X_test, torch.ones(X.shape[0], 1, dtype = torch.float)], dim = -1)
y_test = torch.tensor(y_test.values, dtype = torch.float).view(-1)
y_test[y_test == 0] = -1


losses = []
test_nlls = []
covars = []
final_nlls = []
for rank in params["max_ranks"]:
    model = LogisticRegression(X.shape[1], max_rank = rank)
    l, tl = train(model, X, y, params, X_test = X_test, y_test = y_test)
    covars.append(model.U @ model.U.t() + torch.diag(model.d))
    losses.append(l)
    test_nlls.append(tl)
    final_nlls.append(model.nll_loss(X_test, y_test, 
                                    require_grads = False, 
                                    n_samples = 2000).item())
    
full_model = LogisticRegressionFull(X.shape[1])
lf, tlf = train(full_model, X, y, params, X_test = X_test, y_test = y_test)
covars.append(torch.inverse(full_model.Sigma))
final_nlls.append(full_model.nll_loss(X_test, y_test, 
                                      require_grads = False, 
                                      n_samples = 2000).item())

print(final_nlls)

dir = 'experiments/SLANG/lr'
if not os.path.exists(dir):
    os.makedirs(dir)

fig, axes = plt.subplots(ncols=2, figsize=(10, 3))
for i, l in enumerate(losses):
    axes[0].plot(np.log(np.arange(1, len(l[::20]) + 1)), l[::20], label = "SLANG-{}".format(params["max_ranks"][i]))
axes[0].plot(np.log(np.arange(1, len(lf[::20]) + 1)), lf[::20], label = 'Full-Gaussian')
axes[0].legend()
axes[0].set_title('Train Loss')


for i, l in enumerate(test_nlls):
    axes[1].plot(np.log(np.arange(1, len(l) + 1)), l, label = "SLANG-{}".format(params["max_ranks"][i]))
axes[1].plot(np.log(np.arange(1, len(tlf) + 1)), tlf, label = 'Full-Gaussian')
axes[1].legend()
axes[1].set_title('Test NLLS')

fig.savefig(dir + '/NLLS.png')

names = ['SLANG-{}'.format(rank) for rank in params["max_ranks"]]
if params["do_full_model"]:
    names += ["Full"]
for i, covar in enumerate(covars):
    plt.imsave(dir + '/' + names[i] + 'covar_inverse.png', covar.detach())
    plt.imsave(dir + '/' + names[i] + 'covar.png', torch.inverse(covar).detach())
