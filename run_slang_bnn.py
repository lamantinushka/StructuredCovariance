import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from slang.utils import *
from slang.models import MNISTBnn
from slang.train import *
import matplotlib.pyplot as plt
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument("--task", default = "australian", type = str)
parser.add_argument("--hidden_size", default = 20, type = int)
parser.add_argument("--batch_size", default = 64, type = int)
parser.add_argument("--n_epoch", default = 1000, type = int)
parser.add_argument("--beta", default = 0.03, type = float)
parser.add_argument("--alpha", default = 0.03, type = float)
parser.add_argument("--lambda", default = 1., type = float)
parser.add_argument("--max_ranks", nargs='+', default = [5, 10, 20], type = int)
parser.add_argument("--n_iter", default = 4, type = int)
parser.add_argument("--pow", default = 0.2, type = float)
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
y = torch.tensor(y_train.values, dtype = torch.long).view(-1)
y_ohe = torch.zeros(len(X), 2)
y_ohe[np.arange(len(y)), y] = 1
params["N"] = X.shape[0]

X_test = torch.tensor(X_test.values, dtype = torch.float)
y_test = torch.tensor(y_test.values, dtype = torch.long).view(-1)
y_ = torch.zeros(len(y_test), 2)
y_[np.arange(len(y_test)), y_test] = 1
y_test = y_

losses = []
test_nlls = []
final_nlls = []
for rank in params["max_ranks"]:
    model = MNISTBnn(inp_dim = X.shape[1], max_rank = rank, hidden_size = params["hidden_size"])
    l, tl = train(model, X, y_ohe, params, X_test = X_test, y_test = y_test)
    losses.append(l)
    test_nlls.append(tl)
    final_nlls.append(model.nll_loss(X_test, y_test, 
                                    require_grads = False, 
                                    n_samples = 2000).item())

print(final_nlls)

dir = 'experiments/SLANG/bnn'
if not os.path.exists(dir):
    os.makedirs(dir)

fig, axes = plt.subplots(ncols=2, figsize=(10, 3))
for i, l in enumerate(losses):
    axes[0].plot(np.log(np.arange(1, len(l[::20]) + 1)), l[::20], label = "SLANG-{}".format(params["max_ranks"][i]))
axes[0].legend()
axes[0].set_title('Train Loss')


for i, l in enumerate(test_nlls):
    axes[1].plot(np.log(np.arange(1, len(l) + 1)), l, label = "SLANG-{}".format(params["max_ranks"][i]))
axes[1].legend()
axes[1].set_title('Test NLLS')

fig.savefig(dir + '/NLLS.png')
