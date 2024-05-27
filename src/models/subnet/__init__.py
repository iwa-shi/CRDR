from typing import Dict

from copy import deepcopy

from src.utils.logger import IndentedLog, log_dict_items
from src.utils.registry import (ENCODER_REGISTRY, DECODER_REGISTRY, HYPERENCODER_REGISTRY, 
            HYPERDECODER_REGISTRY, CONTEXTMODEL_REGISTRY, ENTROPYMODEL_REGISTRY)

from .autoencoder import *
from .context_model import *
from .entropy_model import *
from .hyperprior import *
# from .residual_predictor import *


def build_subnet(subnet_opt: Dict, subnet_type: str):
    """

    Args:
        opt (Dict):
        subnet_type (str): e.g., encoder, decoder

    Returns:
        subet[nn.Module]: 

    Example:
        `build_subnet(opt.encoder, 'encoder') -> Encoder`\\
        `build_subnet(opt.entropy_model_z, 'entropy_model') -> EntropyModel`\\
        `build_subnet(opt.discriminator, 'discriminator') -> Discriminator`
    """
    subnet_opt = deepcopy(subnet_opt)
    network_type = subnet_opt.pop('type')
    registry = {
        'encoder': ENCODER_REGISTRY,
        'decoder': DECODER_REGISTRY,
        'hyperencoder': HYPERENCODER_REGISTRY,
        'hyperdecoder': HYPERDECODER_REGISTRY,
        'context_model': CONTEXTMODEL_REGISTRY,
        'entropy_model': ENTROPYMODEL_REGISTRY,
    }[subnet_type]
    subnet = registry.get(network_type)(**subnet_opt)
    log_dict_items(subnet_opt, level='DEBUG', indent=True)
    return subnet
