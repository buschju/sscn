import os
from typing import Any, Dict, Tuple, Optional

import torch
import torchvision
from torch.nn import Module
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import ToTensor


def load_vision_dataset(data_root: str,
                        dataset_name: str,
                        normalize: bool,
                        ) -> Dataset:
    if normalize:
        transform = ToTensor()
    else:
        transform = None

    dataset_class_name, dataset_part = dataset_name.split('_')
    dataset_class = getattr(torchvision.datasets, dataset_class_name)

    if dataset_class_name in ['MNIST']:
        if dataset_part == 'train':
            dataset = dataset_class(root=data_root,
                                    train=True,
                                    download=True,
                                    transform=transform,
                                    )
        elif dataset_part == 'test':
            dataset = dataset_class(root=data_root,
                                    train=False,
                                    download=True,
                                    transform=transform,
                                    )
        elif dataset_part == 'full':
            training_set = dataset_class(root=data_root,
                                         train=True,
                                         download=True,
                                         transform=transform,
                                         )
            test_set = dataset_class(root=data_root,
                                     train=False,
                                     download=True,
                                     transform=transform,
                                     )
            dataset = ConcatDataset([training_set, test_set])
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset_name))
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))

    return dataset


def get_trained_model_path(config: Dict[str, Any],
                           trained_models_root: str,
                           ) -> str:
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    parameter_dictionaries = [value for value in config.values() if isinstance(value, dict)]

    trained_model_path = os.path.join(trained_models_root,
                                      dataset_name,
                                      )
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)
    filename = '{}_'.format(model_name)
    filename += '_'.join(['{}={}'.format(key, value) for key, value in parameter_dictionaries[0].items()])
    for i in range(1, len(parameter_dictionaries)):
        filename += '_' + '_'.join(['{}={}'.format(key, value) for key, value in parameter_dictionaries[i].items()])
    filename += '.pyt'
    trained_model_path = os.path.join(trained_model_path, filename)

    return trained_model_path


def save_trained_model(model: Module,
                       config: Dict[str, Any],
                       trained_models_root: str,
                       key_prefixes: Optional[Tuple[str, ...]] = None,
                       ) -> None:
    trained_model_path = get_trained_model_path(config=config,
                                                trained_models_root=trained_models_root,
                                                )
    pretrained_keys = model.state_dict().keys() if key_prefixes is None else [key for key in model.state_dict().keys()
                                                                              for key_prefix in key_prefixes if
                                                                              key.startswith(key_prefix)]
    torch.save({key: model.state_dict()[key] for key in pretrained_keys},
               trained_model_path,
               )


def get_autoencoder_config(type: str,
                           config: Dict[str, Any],
                           ) -> Dict[str, Any]:
    config_autoencoder = {}
    config_autoencoder['dataset_name'] = config['dataset_name']
    config_autoencoder['dataset_parameters'] = {key: value for key, value in config['dataset_parameters'].items()
                                                if key in ['normalize',
                                                           ]
                                                }
    config_autoencoder['model_name'] = 'ClusteringAutoencoder'

    if type == 'conv':
        config_autoencoder['model_parameters'] = {key: value for key, value in config['model_parameters'].items()
                                                  if key in ['image_size',
                                                             'in_channels',
                                                             'kernel_size',
                                                             'num_channels',
                                                             'stride',
                                                             'padding',
                                                             ]
                                                  }
    elif type == 'linear':
        return None
    else:
        raise ValueError('Unknown auto-encoder type: {}'.format(type))

    config_autoencoder['training_parameters'] = config['training_parameters_autoencoder']
    config_autoencoder['run_parameters'] = {key: value for key, value in config['run_parameters'].items()
                                            if key in ['seed',
                                                       ]
                                            }

    return config_autoencoder


def load_trained_model_parameters(model: Module,
                                  config: Dict[str, Any],
                                  trained_models_root: str,
                                  ) -> Module:
    trained_model_path = get_trained_model_path(config=config,
                                                trained_models_root=trained_models_root,
                                                )

    loaded_parameters = torch.load(trained_model_path)
    model_parameters = model.state_dict()
    model_parameters.update(loaded_parameters)
    model.load_state_dict(model_parameters)

    return model


def load_pretrained_autoencoder(model: Module,
                                config: Dict[str, Any],
                                trained_models_root: str,
                                ):
    model_name = config['model_name']

    if model_name.endswith('Conv'):
        autoencoder_type = 'conv'
    elif model_name.endswith('Linear'):
        autoencoder_type = 'linear'
    else:
        raise ValueError('Unknown auto-encoder architecture')

    config_autoencoder = get_autoencoder_config(type=autoencoder_type,
                                                config=config
                                                )
    model = load_trained_model_parameters(model=model,
                                          config=config_autoencoder,
                                          trained_models_root=trained_models_root,
                                          )

    return model
