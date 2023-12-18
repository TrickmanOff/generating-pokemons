import os
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np
import torch
import torch.utils.data
import torchvision
from torch import nn
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms


"""
Датасеты могут быть двух типов:
1. Элемент - число или тензор. В этом случае элемент рассматривается как x в GAN
2. Элемент - tuple длины 2. В этом случае 1-ый элемент tuple - x, 2-й - y (условие)
y - либо число, либо тензор, либо tuple с числами/тензорами

Обёртка в виде UnifiedDatasetWrapper приводит оба типа датасетов ко 2-му (для датасета 1-го типа y = None).
Обёртку следует использовать пользователю.
"""


class RandomDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int, *args, **kwargs):
        sampler = torch.utils.data.sampler.RandomSampler(dataset, replacement=True)
        random_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size,
                                                               drop_last=False)

        super().__init__(dataset, batch_sampler=random_sampler, *args, **kwargs)


def get_random_infinite_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, *args, **kwargs):
    return cycle(RandomDataloader(dataset, batch_size=batch_size, *args, **kwargs))


def collate_fn(els_list: Sequence[Union[Tuple, int, torch.Tensor]]):
    if isinstance(els_list[0], tuple):
        return tuple(collate_fn(list(a)) for a in zip(*els_list))
    elif isinstance(els_list[0], int):
        return torch.Tensor(els_list)
    elif isinstance(els_list[0], torch.Tensor):
        return torch.stack(tuple(els_list))
    elif els_list[0] is None:
        return None
    else:
        raise RuntimeError


def stack_batches(batches_list):
    if isinstance(batches_list[0], tuple):
        return tuple(stack_batches(list(a)) for a in zip(*batches_list))
    elif isinstance(batches_list[0], torch.Tensor):
        return torch.concat(batches_list, dim=0)
    elif batches_list[0] is None:
        return None


def move_batch_to(batch, device):
    if isinstance(batch, tuple):
        return tuple(move_batch_to(subbatch, device) for subbatch in batch)
    elif batch is None:
        return None
    else:
        return batch.to(device)


class UnifiedDatasetWrapper(torch.utils.data.Dataset):
    """
    Обёртка для поддержки датасетов обоих типов
    """
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset
        self.inverse_transform = getattr(dataset, 'inverse_transform', None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, n: int) -> Tuple[Any, Any]:
        element = self.dataset[n]
        if isinstance(element, tuple):
            assert len(element) == 2
            x, y = element
        else:
            x, y = element, None
        return x, y


def get_default_image_transform(dim: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        # Переводим цвета пикселей в отрезок [-1, 1] аффинным преобразованием, изначально они в отрезке [0, 1]
        transforms.Normalize(tuple(0.5 for _ in range(dim)), tuple(0.5 for _ in range(dim)))
    ])


def get_default_image_inverse_transform(dim: int) -> transforms.Compose:
    return transforms.Compose([
        # Переводим цвета пикселей в отрезок [0, 1] аффинным преобразованием, изначально они в отрезке [-1, 1]
        transforms.Normalize(tuple(-1 for _ in range(dim)), tuple(2 for _ in range(dim))),
        transforms.ToPILImage()
    ])


default_image_transform = get_default_image_transform(3)  # 3 corresponds to 3 RGB channels
default_image_inverse_transform = get_default_image_inverse_transform(3)


class SimpleImagesDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, transform: Callable[[Image.Image], torch.Tensor] = None,
                 load_all_in_memory: bool = False):
        index = []
        for filename in sorted(os.listdir(data_dir)):
            if filename.startswith('.'):
                continue
            index.append(str(data_dir / filename))
        self._index = index
        if load_all_in_memory:
            self._index = [Image.open(path) for path in self._index]
        self.load_all_in_memory = True
        self.transform = transform

    def __getitem__(self, item: int) -> torch.Tensor:
        """
        :return: (num_channels, H, W)
        """
        if self.load_all_in_memory:
            img = self._index[item]
        else:
            img = Image.open(self._index[item])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self._index)


def get_simple_images_dataset(dir_path: Union[str, Path], train: bool = True, val_ratio: float = 0.5,
                              transform: Callable[[Image.Image], torch.Tensor] = default_image_transform,
                              load_all_in_memory: bool = False):
    TRAIN_VAL_SPLIT_SEED = 0x3df3fa
    full_dataset = SimpleImagesDataset(Path(dir_path), transform, load_all_in_memory=load_all_in_memory)

    np.random.seed(TRAIN_VAL_SPLIT_SEED)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_ratio)

    all_indices = np.arange(dataset_size)
    val_indices = np.random.choice(all_indices, size=val_size, replace=False)
    val_mask = np.zeros(dataset_size, dtype=bool)
    val_mask[val_indices] = True
    train_indices = all_indices[~val_mask]
    indices = train_indices if train else val_indices

    return Subset(full_dataset, indices)
