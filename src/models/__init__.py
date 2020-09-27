from typing import Type

from models.dscnet import DscNetConv, DscNetLinear
from models.sscn import SscnConv, SscnLinear
from models.utils import ClusteringAutoencoder

_MODELS = {'ClusteringAutoencoder': ClusteringAutoencoder,
           'DscNetLinear': DscNetLinear,
           'DscNetConv': DscNetConv,
           'SscnLinear': SscnLinear,
           'SscnConv': SscnConv,
           }


def get_model_class(model_name: str,
                    ) -> Type:
    try:
        return _MODELS[model_name]
    except KeyError:
        raise ValueError('Unknown model: {}'.format(model_name))
