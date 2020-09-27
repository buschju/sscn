# SSCN
![alt text](img/model.png)

PyTorch-implementation of the *Siamese Subspace Clustering (SSCN)* model proposed in the paper:

[Learning Self-Expression Metrics for Scalable and Inductive Subspace Clustering]()  
Busch, J., Faerman, E., Schubert, M. and Seidl, T.  
2020

## Setup
Clone this repository, navigate into the root directory and run `python setup.py install`.

## Demo
We provide a demonstration of the inner workings of our model on a small toy dataset. Please check out the notebook `src/demo.ipynb`.

## Running Experiments
- To run experiments or to reproduce the results reported in the paper, you can use the script `src/run_experiment.py`.
- Parameters need to be specified in a config-file in *JSON*-syntax. We uploaded the config-filed used in our experiments into the folder `config`.
- Results will be tracked by *MLFlow*. We uploaded the results from our runs which can be explored using the notebook `src/evaluate_results.ipynb`.
- Auto-encoders can be trained within the pipeline or pre-trained using the script `src/pretrain_autoencoder.py`. We uploaded the auto-encoder used in our runs into the folder `trained_models`.
