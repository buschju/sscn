# SSCN
![alt text](img/model.png)

PyTorch-implementation of the *Siamese Subspace Clustering (SSCN)* model proposed in the paper:

[Learning Self-Expression Metrics for Scalable and Inductive Subspace Clustering](https://arxiv.org/pdf/2009.12875.pdf)  
Busch, J., Faerman, E., Schubert, M. and Seidl, T.  
2020

## Setup
Clone this repository, navigate into the root directory and run `python setup.py install`.

## Demo
We provide a demonstration of the inner workings of our model on a small toy dataset. Please check out the notebook `src/demo.ipynb`.

## Running Experiments
- To run experiments or to reproduce the results reported in the paper, you can use the script `src/run_experiment.py`.
- Parameters need to be specified in a config-file in *JSON*-syntax. We uploaded the config-files used in our experiments into the folder `config`.
- Results will be tracked by *MLflow*. We uploaded the results from our runs which can be explored using the notebook `src/evaluate_results.ipynb`.
- Auto-encoders can be trained within the pipeline or pre-trained using the script `src/pretrain_autoencoder.py`. We uploaded the auto-encoder used in our runs into the folder `trained_models`.

## Cite
If you use our model or any of the provided code or material, please cite our paper:

```
@article{busch2020learning,
  title={Learning Self-Expression Metrics for Scalable and Inductive Subspace Clustering},
  author={Busch, Julian and Faerman, Evgeniy and Schubert, Matthias and Seidl, Thomas},
  journal={arXiv preprint arXiv:2009.12875},
  year={2020}
}
```
