import torch
from torch import nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, RPNHead, _default_anchorgen

def fasterrcnn_resnet101_fpn_v2(num_classes=2, trainable_backbone_layers=3):
    """
    Constructs a Faster-RCNN model with a ResNet-101-FPN backbone. This implementation is a modification of
    torchvision.models.detection.fasterrcnn_resnet50_fpn_v2, which uses a ResNet-50-FPN backbone.

    :param num_classes: The number of output classes.
    :param trainable_backbone_layers: The number of trainable layers, from the back, in the backbone layers.
    :return: A FasterRCNN model that takes "a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range."
    """

    # Generate resnet101 backbone
    backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_backbone_layers,
        norm_layer=nn.BatchNorm2d
    )

    # ToDO: Default anchors for now
    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(
        backbone.out_channels,
        rpn_anchor_generator.num_anchors_per_location()[0],
        conv_depth=2
    )

    # This takes cropped feature maps as input
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7),
        [256, 256, 256, 256],
        [1024],
        norm_layer=nn.BatchNorm2d
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head
    )

    return model