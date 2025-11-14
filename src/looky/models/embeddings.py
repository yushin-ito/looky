from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F


# Copied from diffusers.models.controlnets.controlnet.zero_module
def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class PoseConditioningEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (32, 64, 256, 512),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
            )

        self.conv_out = zero_module(
            nn.Linear(
                block_out_channels[-1] * 4,
                conditioning_embedding_channels,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        patches = F.unfold(embedding, kernel_size=2, stride=2).transpose(1, 2)
        embedding = self.conv_out(patches)

        return embedding
