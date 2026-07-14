#!/usr/bin/env python3
""" Pre-defined networks for use in faceswap """
from .clip import ViT, ViTConfig, TypeModels as TypeModelsViT
from .inception import inception_resnet_v2, override_inception3
from .mobilenet import mobilenet
from .torch_vision import (
    efficientnet_v2_b0, efficientnet_v2_b1, efficientnet_v2_b2, efficientnet_v2_b3,
    patch_legacy
)
from .xception import xception
