import argparse
import json
import os
import random
from typing import Any, Dict

import mlflow
import numpy
import torch

from data_utils import load_vision_dataset, load_pretrained_autoencoder
from eval_utils import evaluate_model
from models import get_model_class
from utils import flatten_dictionary


def run_experiment(data_root: str,
                   device: int,
                   config: Dict[str, Any],
                   trained_models_root: str,
                   ) -> None:
    dataset_name = config['dataset_name']
    dataset_parameters = config['dataset_parameters']
    model_name = config['model_name']
    model_parameters = config['model_parameters']
    training_parameters = config['training_parameters']
    training_parameters_autoencoder = config['training_parameters_autoencoder']
    evaluation_parameters = config['evaluation_parameters']
    run_parameters = config['run_parameters']
    inductive = run_parameters['inductive']

    # Load dataset
    train_dataset = load_vision_dataset(data_root=data_root,
                                        dataset_name=dataset_name,
                                        normalize=dataset_parameters['normalize'],
                                        )
    if inductive:
        test_dataset_name = config['test_dataset_name']
        test_dataset = load_vision_dataset(data_root=data_root,
                                           dataset_name=test_dataset_name,
                                           normalize=dataset_parameters['normalize'],
                                           )

    # Fix random seed
    seed = run_parameters['seed']
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize Model
    model = get_model_class(model_name=model_name)(**model_parameters)

    # Move to device
    torch.cuda.set_device(device)
    model = model.to('cuda')

    # Run
    if 'num_runs' in run_parameters:
        num_runs = run_parameters['num_runs']
    else:
        num_runs = 1
    for _ in range(num_runs):
        with mlflow.start_run():
            # Log hyper-parameter values
            params_flat = flatten_dictionary(config)
            mlflow.log_params(params_flat)

            # Log number of parameters
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_metric('num_params', num_params)

            # Reset
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.reset_parameters()

            if run_parameters['load_pretrained_autoencoder']:
                model = load_pretrained_autoencoder(model=model,
                                                    config=config,
                                                    trained_models_root=trained_models_root,
                                                    )
            else:
                model.pretrain_autoencoder(dataset=train_dataset,
                                           **training_parameters_autoencoder,
                                           )

            # Train model
            model.train_model(dataset=train_dataset,
                              **training_parameters,
                              )

            if inductive:
                # Train classifier
                training_parameters_classifier = config['training_parameters_classifier']
                model.train_classifier(dataset=train_dataset,
                                       **training_parameters_classifier,
                                       )

            # Evaluate model
            acc_run, ari_run, nmi_run = evaluate_model(model=model,
                                                       dataset=train_dataset,
                                                       **evaluation_parameters,
                                                       )
            mlflow.log_metric('acc', acc_run)
            mlflow.log_metric('ari', ari_run)
            mlflow.log_metric('nmi', nmi_run)

            memory_gb = torch.cuda.max_memory_allocated(device=device) / 1e9
            mlflow.log_metric('memory_gb', memory_gb)

            if inductive:
                # Evaluate model on separate test set using classifier
                acc_run_test, ari_run_test, nmi_run_test = evaluate_model(model=model,
                                                                          dataset=test_dataset,
                                                                          **evaluation_parameters,
                                                                          )
                mlflow.log_metric('acc_test', acc_run_test)
                mlflow.log_metric('ari_test', ari_run_test)
                mlflow.log_metric('nmi_test', nmi_run_test)

                memory_test_gb = torch.cuda.max_memory_allocated(device=device) / 1e9
                mlflow.log_metric('memory_test_gb', memory_test_gb)


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_root',
                        help="Path to config files",
                        type=str,
                        default='../config',
                        )
    parser.add_argument('--config_name',
                        help="Name of the config file",
                        type=str,
                        )
    parser.add_argument('--data_root',
                        help="Path to data",
                        type=str,
                        default='../data',
                        )
    parser.add_argument('--trained_models_root',
                        help="Path to trained models",
                        type=str,
                        default='../trained_models',
                        )
    parser.add_argument('--device',
                        help="Device index",
                        type=int,
                        default=0,
                        )
    parser.add_argument('--mlflow_uri',
                        help="MLflow tracking URI",
                        type=str,
                        default='../mlflow',
                        )
    parser.add_argument('--mlflow_experiment_name',
                        help="Experiment name used for MLflow results tracking",
                        type=str,
                        default='sscn',
                        )
    args = parser.parse_args()

    # Parse config file
    with open(os.path.join(args.config_root, '{}.json'.format(args.config_name)), 'r') as config_file:
        config = json.load(config_file)

    # Set up MLflow results tracking
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(name=args.mlflow_experiment_name)
    if experiment is None:
        mlflow.create_experiment(args.mlflow_experiment_name)
    mlflow.set_experiment(args.mlflow_experiment_name)

    # Run experiment
    run_experiment(data_root=args.data_root,
                   device=args.device,
                   config=config,
                   trained_models_root=args.trained_models_root,
                   )
