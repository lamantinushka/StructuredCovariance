# StructuredCovariance
In this project we investigate methods of non-diagonal covariance approximation of the variational gaussian posterior of the machine learning models (logistic regression and deep neural networks). The project is mostly based on two papers:
- SLANG: Fast Structured Covariance Approximations for Bayesian Deep Learning with Natural Gradient. https://arxiv.org/pdf/1811.04504.pdf
- A Simple Baseline for Bayesian Uncertainty in Deep Learning. https://arxiv.org/pdf/1902.02476.pdf

**Project Proposal**: [pdf](https://drive.google.com/file/d/1FFzTuCEyT-UvA8euSIRIvqpneYRZVzNw/view?usp=sharing)

**Presentation**: [pdf](https://drive.google.com/file/d/19xzaSg0eyreZk63jjm5FrRT7JLmkN8Vq/view?usp=sharing)

## Data and models
- data may be found by the link: [data](https://drive.google.com/file/d/1qSbHIVRQlq8rHkzAax1OLF4XHX5mPpy7/view?usp=sharing). 
Please unzip arxive to your StructuredCovariance folder
- models for UQ notebook: [models](https://drive.google.com/file/d/1Go0JiqKR2RB__LfLR5hGilpbVWXviZXt/view?usp=sharing).
You can copy files to your Gdrive and simply mount it to the notebook.

## SLANG
Is a variational inference method based on the idea of natural gradients stochastic approximation. You can find more about VOGN (Variational Online Gauss-Newton) method in the paper Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam https://arxiv.org/pdf/1806.04854.pdf. SLANG approximate inverse covariance matrix as diagonal + low-rank matrix. You can find detailed linalg routine in the paper. In our implementation we strictly followed the described methods.

### How to reproduce?
#### Simple example
Simple examples with illustrated covariance matrixes can be found in `slang_lr_example.ipynb`
#### Experiments
- You can reproduce the results for logistic regression for Australian dataset with the following command:

`python3 run_slang_lr.py`

- Set --do_full_model=True if you'd like to compare results to the Full-Gaussian model obtained by VOGN method.

- Use similar command for BNN:

`python3 run_slang_bnn.py`

- For detailed explanation of the parameters please look at the simple examples.

## SWAG 

Is a semi-bayesian approach to the posterior covariance approximation from the empirical distribution of the weights obtained during the SGD. In the paper such posterior is shown to almost Gaussian. SWAG requires pretrained models.

### How to reproduce?
#### Simple example
Simple examples with illustrated covariance matrix can be found in `swag_lr_example.ipynb`
#### Experiments
- Experiment with uncertainty quantification can be found in `UQ.ipynb`
- You can use pretrained models or train them yourself. It takes several minutes on GPU.
 
