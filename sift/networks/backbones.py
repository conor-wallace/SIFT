from typing import Optional

import torch
import torch.nn as nn
import timm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, FPN, ShapeSpec
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

class ResNet50(Backbone):
    def __init__(self, out_features=None) -> None:
        super().__init__()
        self.model = timm.create_model(
            'resnet50',
            features_only=True,
            pretrained=True
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if out_features is None:
            self._out_features = ["layer4"]
        else:
            self._out_features = out_features
        self._features2idx = {
            "act1": 0,
            "layer1": 1,
            "layer2": 2,
            "layer3": 3,
            "layer4": 4
        }

    @torch.no_grad()
    def forward(self, x: torch.Tensor, pool: Optional[bool] = False) -> torch.Tensor:
        outputs = {}
        feature_maps = self.model(x)
        if pool:
            outputs["global_pool"] = self.global_pool(feature_maps[-1])
        else:
            for f in self._out_features:
                idx = self._features2idx[f]
                outputs[f] = feature_maps[idx]

        return outputs

    def output_shape(self):
        model_info = self.model.feature_info.info
        output_shape = {
            f: ShapeSpec(
                channels=model_info[self._features2idx[f]]["num_chs"],
                stride=model_info[self._features2idx[f]]["reduction"]
            )
            for f in self._out_features
        }
        return output_shape


@BACKBONE_REGISTRY.register()
def build_resnet50_backbone(cfg, input_shape) -> Backbone:
    return ResNet50(cfg.MODEL.RESNETS.OUT_FEATURES)


@BACKBONE_REGISTRY.register()
def build_resnet50_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet50_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


class ConvNext(Backbone):
    def __init__(self, out_features=None) -> None:
        super().__init__()
        self.model = timm.create_model(
            'convnext_base',
            features_only=True,
            pretrained=True
        )
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        if out_features is None:
            self._out_features = ["res5"]
        else:
            self._out_features = out_features
        self._features2idx = {
            "stages.0": 0,
            "stages.1": 1,
            "stages.2": 2,
            "stages.3": 3
        }

    @torch.no_grad()
    def forward(self, x: torch.Tensor, pool: Optional[bool] = False) -> torch.Tensor:
        outputs = {}
        feature_maps = self.model(x)
        if pool:
            outputs["global_pool"] = self.global_pool(feature_maps[-1])
        else:
            for f in self._out_features:
                idx = self._features2idx[f]
                outputs[f] = feature_maps[idx]

        return outputs

    def output_shape(self):
        model_info = self.model.feature_info.info
        return {
            f: ShapeSpec(
                channels=model_info[self._features2idx[f]]["num_channels"],
                stride=model_info[self._features2idx[f]]["reduction"]
            )
            for f in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_convnext_backbone(cfg, input_shape) -> Backbone:
    return ConvNext(cfg.MODEL.RESNETS.OUT_FEATURES)


@BACKBONE_REGISTRY.register()
def build_convnext_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_convnext_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
