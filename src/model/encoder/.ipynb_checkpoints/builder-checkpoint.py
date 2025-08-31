import torch.nn as nn

from .dcformer import decomp_naive, decomp_nano, decomp_small, decomp_tiny
from .vit import Vit3D


def build_vision_tower(config, **kwargs):
    return VisionTower(config)


class VisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.select_layer = config.vision_select_layer # -2
        self.select_feature = config.vision_select_feature # cls_patch
        self.hidden_size = config.dim # 768

        if config.vision_tower == "vit3d":
            self.vision_tower = Vit3D(
                input_size=config.input_size, # 256, 256, 128
                dim=config.dim, # 768
                depth=config.depth, # 12
            )
        elif config.vision_tower == "dcformer":
            self.vision_tower = decomp_small(
                input_size=config.input_size, # 256, 256, 128
            )
            self.low_input_size = self.vision_tower.channels[-2]
            self.high_input_size = self.vision_tower.channels[-1]
        else:
            raise ValueError(f"Unexpected vision tower: {config.vision_tower}")

    def forward(self, images):
        hidden_states = self.vision_tower(images)
        if self.select_layer == 0:
            image_features = hidden_states
        elif self.select_layer < 0:
            image_features = hidden_states[self.select_layer :]
        else:
            raise ValueError(f"Unexpected select layer: {self.select_layer}")

        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
