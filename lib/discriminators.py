from abc import abstractmethod
from typing import Any

import torch
from torch import nn


class Discriminator(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param x: object from the considered space
        :param y: condition
        None means no condition.
        A discriminator knows the exact type of condition and how to use it.
        If discriminator does not support conditions, it is expected to raise an exception.
        """
        pass


class DCDiscriminator(Discriminator):
    LRELU_SLOPE = 0.2

    def __init__(self, latent_channels: int = 1024, image_channels: int = 3):
        super().__init__()

        self.convs = nn.ModuleList([
            # (B, image_channels, 64, 64)
            nn.Conv2d(image_channels, latent_channels // 8, kernel_size=4, stride=2, padding=1,
                      bias=False),  # (B, latent_channels // 8, 32, 32)
            nn.Conv2d(latent_channels // 8, latent_channels // 4, kernel_size=4, stride=2,
                      padding=1, bias=False),  # (B, latent_channels // 4, 16, 16)
            nn.Conv2d(latent_channels // 4, latent_channels // 2, kernel_size=4, stride=2,
                      padding=1, bias=False),  # (B, latent_channels // 2, 8, 8)
            nn.Conv2d(latent_channels // 2, latent_channels, kernel_size=4, stride=2, padding=1,
                      bias=False),  # (B, latent_channels, 4, 4)
        ])

        self.post_convs = nn.ModuleList(
            [nn.LeakyReLU(self.LRELU_SLOPE)] +
            [
                nn.Sequential(nn.BatchNorm2d(conv.out_channels), nn.LeakyReLU(self.LRELU_SLOPE))
                for conv in self.convs[1:-1]
            ]
        )

        self.final_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """
        :param x: of shape (B, image_channels, 64, 64)
        :returns: of shape (B, latent_channels*4*4)
        """
        if y is not None:
            raise RuntimeError('Condition not supported')

        for conv, post_conv in zip(self.convs[:-1], self.post_convs):
            x = post_conv(conv(x))
        x = self.convs[-1](x)
        x = x.flatten()
        return self.final_act(x)
