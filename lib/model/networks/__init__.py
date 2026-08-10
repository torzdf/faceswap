#!/usr/bin/env python3
""" Pre-defined networks for use in faceswap """
from .clip import vit_b_16, vit_b_32, vit_l_14, vit_l_14_336px
from .inception import inception_resnet_v2, override_inception3
from .mobilenets import mobilenet, mobilenet_v3_small, mobilenet_v3_large
from .nasnet import nasnet_mobile, nasnet_large
from .resnet import resnet50, resnet50_v2, resnet101, resnet101_v2, resnet152, resnet152_v2
from .torch_vision import (
    convnext_xlarge, efficientnet_v2_b0, efficientnet_v2_b1, efficientnet_v2_b2,
    efficientnet_v2_b3, patch_legacy
)
from .xception import xception
