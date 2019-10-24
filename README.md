# WeightPrior

## Paper
**The Deep Weight Prior**
[arXiv:1810.06943](https://arxiv.org/abs/1810.06943)

## Project Proposal
* [Overleaf Document](https://www.overleaf.com/read/njvynwkrxzvd)
* [PDF](https://github.com/matyushinleonid/WeightPrior/blob/master/BMML_Project_Proposal.pdf)

## Goals
We are going to reproduce considered in the paper experiments and prove (or disprove) that for dwp we have that

* dwp improve the performance of Bayesian neural networks in case of limited data
* initialization of weights with samples from dwp accelerates training of conventional convolutional neural networks, such that training procedure becomes faster than, say, in uniform initialization case, or in Xavier initialization case

## Members
* Leonid Matyushin

## How-to Reproduce Experiments
* run [Auxiliary CNN Training](https://github.com/matyushinleonid/WeightPrior/blob/master/get_kernels.ipynb)
* run [Experiments](https://github.com/matyushinleonid/WeightPrior/blob/master/train_vae.ipynb)
