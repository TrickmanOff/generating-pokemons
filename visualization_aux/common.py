from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

from lib.data import default_image_inverse_transform
from lib.gan import GAN
from lib.utils import get_local_device


def generate_images(gan_model: GAN, images_cnt: int, batch_size: int = 32,
                    seed: Optional[int] = None) -> List[Image.Image]:
    gen_outputs = []
    batches_cnt = (images_cnt + batch_size - 1) // batch_size
    noise = gan_model.gen_noise(images_cnt, seed=seed).to(get_local_device())
    with torch.no_grad():
        for batch_i in tqdm(range(batches_cnt), desc='Generating images'):
            cur_noise = noise[batch_i*batch_size:(batch_i + 1)*batch_size]
            gen_output = gan_model.generator(cur_noise)
            gen_outputs.append(gen_output)

    gen_outputs = torch.concat(gen_outputs, dim=0).detach().cpu()  # (B, 3, 64, 64)
    return [default_image_inverse_transform(output) for output in gen_outputs]


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


def generate_grid(gan_model: GAN, nrows: int, ncols: int, figsize=(20, 20),
                  batch_size: int = 32,
                  seed: Optional[int] = None) -> plt.Figure:
    """
    plots in matplotlib
    """
    gan_model.eval()
    images = generate_images(gan_model, nrows * ncols, batch_size=batch_size, seed=seed)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            ax.axis('off')
            img = images[row*ncols + col]
            ax.imshow(img)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    return fig
