import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import train
from lib.data import default_image_inverse_transform
from lib.gan import GAN
from lib.utils import download_file, get_local_device
from visualization_aux.common import generate_grid, generate_images


URL_LINKS = {
    'gan_big': 'https://www.googleapis.com/drive/v3/files/1nFUBPYrKDO0_VTF1qRFu3_ApHy8hx9zX?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI',
}


def get_generator(arch_name: str, storage_dir: str) -> GAN:
    if arch_name not in URL_LINKS:
        raise RuntimeError(
            f'Only the following architectures are supported: {", ".join(URL_LINKS.keys())}')

    checkpoint_filename = 'generator_checkpoint.pth'
    checkpoint_filepath = Path(storage_dir) / arch_name / checkpoint_filename

    if not checkpoint_filepath.exists():
        checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)
        print(f'The checkpoint of {arch_name} will be downloaded')
        download_file(URL_LINKS[arch_name], checkpoint_filepath.parent, checkpoint_filepath.name)

    device = get_local_device()
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    noise_dim = checkpoint['noise_dim']
    generator = train.get_generator(noise_dim, arch_name)
    generator.load_state_dict(checkpoint['generator'])

    noise_generator = train.get_noise_generator(noise_dim)

    gan = GAN(generator=generator, discriminator=None, noise_generator=noise_generator)
    gan.eval()
    print('The generator is successfully loaded')
    return gan


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="GAN training script")
    args.add_argument(
        "-s",
        "--storage",
        default='inference/checkpoints',
        type=str,
        help="a directory where the checkpoints are stored",
    )
    args.add_argument(
        "-o",
        "--output",
        default='inference/generated',
        type=str,
        help="a directory with the generated images",
    )
    args.add_argument(
        "-m",
        "--model",
        default='gan_big',
        type=str,
        help="which architecture to use",
    )
    args.add_argument(
        "-b",
        "--batch",
        default=32,
        type=int,
        help="batch size for the generator",
    )
    args.add_argument(
        "-g",
        action="store_true",
        help="generate images as a grid",
    )
    args.add_argument(
        "numbers",
        nargs=argparse.REMAINDER
    )
    args = args.parse_args()

    generator = get_generator(args.model, args.storage)

    output_dirpath = Path(args.output)
    output_dirpath.mkdir(parents=True, exist_ok=True)

    if not args.g:
        n = int(args.numbers[0])

        gen_images = generate_images(generator, images_cnt=n, batch_size=args.batch)

        for i, img in enumerate(gen_images):
            filename = f'{i:03d}.png'
            filepath = output_dirpath / filename
            img.save(filepath, format='png')
    else:
        nrows, ncols = map(int, args.numbers)
        fig = generate_grid(generator, nrows, ncols, batch_size=args.batch)
        filepath = output_dirpath / 'grid.png'
        fig.savefig(filepath, bbox_inches='tight')
