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

    def __init__(self, latent_channels: int = 1024, image_channels: int = 3, use_wgan_loss: bool = True):
        super().__init__()

        self.use_wgan_loss = use_wgan_loss

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

        if self.use_wgan_loss:
            self.last_post_conv = nn.BatchNorm2d(latent_channels)
            self.pre_head = nn.LeakyReLU(self.LRELU_SLOPE)
            self.head = nn.Linear(latent_channels * 4 * 4, 1)
        else:
            self.final_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """
        :param x: of shape (B, image_channels, 64, 64)
        :returns: of shape (B, latent_channels*4*4) if wgan loss is used else - (B, 1)
        """
        if y is not None:
            raise RuntimeError('Condition not supported')

        for conv, post_conv in zip(self.convs[:-1], self.post_convs):
            x = post_conv(conv(x))

        if self.use_wgan_loss:
            x = self.last_post_conv(self.convs[-1](x))
            x = x.flatten(start_dim=1)
            x = self.head(self.pre_head(x))
        else:
            x = self.final_act(x)
        return x


class FixedDCDiscriminator(Discriminator):
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
            nn.Conv2d(latent_channels, 1, kernel_size=4,
                      bias=False),  # (B, 1, 1, 1)
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
        :returns: of shape (B, 1)
        """
        if y is not None:
            raise RuntimeError('Condition not supported')

        for conv, post_conv in zip(self.convs[:-1], self.post_convs):
            x = post_conv(conv(x))
        x = self.convs[-1](x)  # (B, 1, 1, 1)
        x = x.flatten(1)  # (B, 1)
        x = self.final_act(x)
        return x
