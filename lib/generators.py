from abc import abstractmethod
from typing import Any

import torch
from torch import nn


class Generator(nn.Module):
    @abstractmethod
    def forward(self, z: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param z: seed/noise for generation
        :param y: condition
        None means no condition.
        A generator knows the exact type of condition and how to use it for generation.
        If generator does not support conditions, it is expected to raise an exception.
        """
        pass


class DCGenerator(Generator):
    """
    Based on
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    https://arxiv.org/pdf/1511.06434v2.pdf
    """
    LATENT_WIDTH = 4

    def __init__(self, noise_dim: int = 100, latent_channels: int = 1024,
                 image_channels: int = 3):
        super().__init__()
        latent_width = self.LATENT_WIDTH

        self.latent_channels = latent_channels

        self.noise_proj = nn.Linear(noise_dim, latent_channels * latent_width * latent_width)

        self.tconvs = nn.ModuleList([
            # (B, latent_channels, 4, 4)
            nn.ConvTranspose2d(latent_channels, latent_channels // 2, kernel_size=4, stride=2,
                               padding=1, bias=False),  # (B, latent_channels // 2, 8, 8)
            nn.ConvTranspose2d(latent_channels // 2, latent_channels // 4, kernel_size=4, stride=2,
                               padding=1, bias=False),  # (B, latent_channels // 4, 16, 16)
            nn.ConvTranspose2d(latent_channels // 4, latent_channels // 8, kernel_size=4, stride=2,
                               padding=1, bias=False),  # (B, latent_channels // 8, 32, 32)
            nn.ConvTranspose2d(latent_channels // 8, image_channels, kernel_size=4, stride=2,
                               padding=1, bias=False),  # (B, image_channels, 64, 64)
        ])

        self.noise_proj_norm = nn.Sequential(nn.BatchNorm2d(latent_channels), nn.ReLU(inplace=True))

        self.post_tconvs = nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(tconv.out_channels), nn.ReLU(inplace=True))
            for tconv in self.tconvs[:-1]
        ])

        self.final_act = nn.Tanh()

    def forward(self, z: torch.Tensor, y=None) -> torch.Tensor:
        """
        :param z: of shape (B, noise_dim)
        :returns: of shape (B, image_channels, 64, 64)
        """
        if y is not None:
            raise RuntimeError('Condition not supported')

        x = self.noise_proj(z).reshape(-1, self.latent_channels, self.LATENT_WIDTH,
                                       self.LATENT_WIDTH)  # (B, latent_channels, 4, 4)
        x = self.noise_proj_norm(x)  # (B, latent_channels, 4, 4)

        for tconv, norm in zip(self.tconvs[:-1], self.post_tconvs):
            x = norm(tconv(x))
        # x: (B, latent_channels // 8, 32, 32)

        x = self.final_act(self.tconvs[-1](x))  # (B, image_channels, 64, 64)

        return x


class FixedDCGenerator(Generator):
    """
    Based on
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    https://arxiv.org/pdf/1511.06434v2.pdf
    """

    def __init__(self, noise_dim: int = 100, latent_channels: int = 1024,
                 image_channels: int = 3):
        super().__init__()
        self.latent_channels = latent_channels

        self.noise_proj = nn.Linear(noise_dim, latent_channels * 4 * 4)

        self.tconvs = nn.ModuleList([
            # (B, noise_dim)
            nn.ConvTranspose2d(noise_dim, latent_channels, kernel_size=4, bias=False),  # (B, latent_channels, 4, 4)
            nn.ConvTranspose2d(latent_channels, latent_channels // 2, kernel_size=4, stride=2,
                               padding=1, bias=False),  # (B, latent_channels // 2, 8, 8)
            nn.ConvTranspose2d(latent_channels // 2, latent_channels // 4, kernel_size=4, stride=2,
                               padding=1, bias=False),  # (B, latent_channels // 4, 16, 16)
            nn.ConvTranspose2d(latent_channels // 4, latent_channels // 8, kernel_size=4, stride=2,
                               padding=1, bias=False),  # (B, latent_channels // 8, 32, 32)
            nn.ConvTranspose2d(latent_channels // 8, image_channels, kernel_size=4, stride=2,
                               padding=1, bias=False),  # (B, image_channels, 64, 64)
        ])

        self.noise_proj_norm = nn.Sequential(nn.BatchNorm2d(latent_channels), nn.ReLU(inplace=True))

        self.post_tconvs = nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(tconv.out_channels), nn.ReLU(inplace=True))
            for tconv in self.tconvs[:-1]
        ])

        self.final_act = nn.Tanh()

    def forward(self, z: torch.Tensor, y=None) -> torch.Tensor:
        """
        :param z: of shape (B, noise_dim)
        :returns: of shape (B, image_channels, 64, 64)
        """
        if y is not None:
            raise RuntimeError('Condition not supported')

        x = z[:, :, None, None]

        for tconv, post_tconv in zip(self.tconvs[:-1], self.post_tconvs):
            x = post_tconv(tconv(x))
        # x: (B, latent_channels // 8, 32, 32)

        x = self.final_act(self.tconvs[-1](x))  # (B, image_channels, 64, 64)

        return x
