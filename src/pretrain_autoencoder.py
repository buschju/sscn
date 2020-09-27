import argparse
import json
import os
import random

import mlflow
import numpy
import torch

from data_utils import load_vision_dataset, save_trained_model
from models import get_model_class
from models.utils import get_conv_autoencoder

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
    args = parser.parse_args()

    # Parse config file
    with open(os.path.join(args.config_root, '{}.json'.format(args.config_name)), 'r') as config_file:
        config = json.load(config_file)

    # Dump results into default location
    mlflow.set_tracking_uri(args.mlflow_uri)

    # Load dataset
    dataset = load_vision_dataset(data_root=args.data_root,
                                  dataset_name=config['dataset_name'],
                                  normalize=config['dataset_parameters']['normalize'],
                                  )

    # Fix random seed
    seed = config['run_parameters']['seed']
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize Model
    encoder, decoder, _ = get_conv_autoencoder(**config['model_parameters'])
    model = get_model_class(model_name=config['model_name'])(encoder=encoder,
                                                             decoder=decoder,
                                                             )

    # Move to device
    torch.cuda.device(args.device)
    model = model.to('cuda')

    # Pre-train auto-encoder
    model.pretrain_autoencoder(dataset=dataset,
                               **config['training_parameters'],
                               )

    # Save pre-trained auto-encoder
    save_trained_model(model=model,
                       config=config,
                       trained_models_root=args.trained_models_root,
                       )
