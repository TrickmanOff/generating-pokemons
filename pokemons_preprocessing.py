"""
Created for the https://www.kaggle.com/datasets/zackseliger/pokemon-images-includes-fakemon/ dataset
"""
import argparse
import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_rgba_to_rgb(img):
    img_array = np.array(img)  # (W, H, 4)
    white_mask = img_array[..., 3] == 0
    rgb_img = img.convert('RGB')  # (W, H, 3)
    rgb_img_array = np.array(rgb_img)
    rgb_img_array[white_mask] = [255, 255, 255]
    return Image.fromarray(rgb_img_array)


def crop_white_background(img):
    img_array = np.array(img)  # (W, H, 3)
    white_mask = (img_array == 255).all(axis=-1)
    non_white_rows_mask = ~white_mask.all(axis=1)
    non_white_cols_mask = ~white_mask.all(axis=0)
    cropped_img_array = img_array[non_white_rows_mask][:, non_white_cols_mask]
    return Image.fromarray(cropped_img_array)


def pad_to_square(img, background_pixel=(255, 255, 255)):
    width, height = img.size
    left, top = 0, 0  # padding size
    diff = abs(width - height)
    if width < height:
        left = diff // 2
        width += diff
    elif height < width:
        top = diff // 2
        height += diff

    result = Image.new(img.mode, (width, height), background_pixel)
    result.paste(img, (left, top))
    return result


def main(input_dirpath: Union[str, Path], output_dirpath: Union[str, Path], target_resolution: int = 64):
    images_dirpath = Path(input_dirpath)
    output_dirpath = Path(output_dirpath)

    output_dirpath.mkdir(parents=True, exist_ok=True)

    target_resolution = (target_resolution, target_resolution)

    images_filenames = [filename for filename in sorted(os.listdir(images_dirpath)) if
                        not filename.startswith('.')]

    for i, image_filename in enumerate(tqdm(images_filenames, desc='Processing images')):
        img = Image.open(images_dirpath / image_filename)
        rgb_img = convert_rgba_to_rgb(img)
        cropped_rgb_img = crop_white_background(rgb_img)
        cropped_rgb_img = pad_to_square(cropped_rgb_img)
        res_img = cropped_rgb_img.resize(target_resolution, resample=Image.Resampling.NEAREST)

        output_filename = f'{i + 1:04d}.png'
        res_img.save(output_dirpath / output_filename, format='png')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Pokemons dataset preprocessing")
    args.add_argument(
        "-i",
        "--input",
        type=str,
        help="the path to the directory with images",
    )
    args.add_argument(
        "-o",
        "--output",
        type=str,
        help="the path to the directory for storing the resulting images",
    )
    args.add_argument(
        "-r",
        "--resolution",
        default=64,
        type=int,
        help="target resolution of all images, all images will be squared ",
    )
    args = args.parse_args()

    main(args.input, args.output, args.resolution)
