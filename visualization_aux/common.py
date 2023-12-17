import numpy as np
import matplotlib.pyplot as plt
import torch

from lib.data import default_image_inverse_transform
from lib.gan import GAN
from lib.utils import get_local_device


def imshow(img, ax=None, cmap=None):
    img = default_image_inverse_transform(img.detach())
    npimg = np.array(img)
    if ax is None:
        plt.imshow(npimg, cmap=cmap)
        plt.show()
    else:
        ax.imshow(npimg, cmap=cmap)


def gen_several_images(gan_model: GAN, n: int = 5, y=None, figsize=(13, 13), imshow_fn=imshow):
    """
    Выводит n изображений, сгенерированных gan_model в строке
    """
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=figsize)
    gan_model.to(get_local_device())
    with torch.no_grad():
        noise_batch = gan_model.gen_noise(n).to(get_local_device())
        gen_batch = gan_model.generator(noise_batch, y)

    if n == 1:
        axs = [axs]
    for i, (tensor, ax) in enumerate(zip(gen_batch, axs)):
        imshow_fn(tensor.cpu(), ax=ax)
        if y is not None and isinstance(y, torch.Tensor):
            ax.set_xlabel(y[i].item())

    plt.show()


def generate_grid(gan_model: GAN, nrows: int, ncols: int, imshow_fn=imshow) -> plt.Figure:
    """
    plots in matplotlib
    """
    gan_model.eval()
    with torch.no_grad():
        cnt = nrows * ncols
        z = gan_model.gen_noise(cnt).to(get_local_device())
        gen_x = gan_model.generator(z).detach()  # (B, 3, 64, 64)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    gen_x = gen_x.reshape(nrows, ncols, *gen_x.shape[1:])
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            ax.axis('off')
            x = gen_x[row, col]
            imshow_fn(x, ax)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    return fig
