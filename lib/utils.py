import numpy as np
import torch

import requests
from tqdm import tqdm


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def get_local_device() -> torch.device:
    return device


def ohe_labels(y: torch.Tensor, classes_cnt: int) -> torch.Tensor:
    batch_size = y.shape[0]
    m = torch.zeros(batch_size, classes_cnt).to(y.device)
    m[np.arange(batch_size), y.long()] = 1
    return m


@torch.no_grad()
def calc_grad_norm(model, norm_type=2) -> float:
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()


def download_file(url, to_dirpath=None, to_filename=None):
    local_filename = to_filename or url.split('/')[-1]
    if to_dirpath is not None:
        to_dirpath.mkdir(exist_ok=True, parents=True)
        local_filename = to_dirpath / local_filename
    chunk_size = 2**20  # in bytes
    with requests.get(url, stream=True) as r:
        if 'Content-length' in r.headers:
            total_size = int(r.headers['Content-length'])
            total = (total_size - chunk_size + 1) // chunk_size
        else:
            total_size = None
        desc = f'Downloading file'
        if total_size is not None:
            desc += f', {total_size / (2**30):.2f}GBytes'
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=total, desc=desc, unit='MBytes'):
                f.write(chunk)
    return local_filename
