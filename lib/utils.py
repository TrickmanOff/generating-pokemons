import numpy as np
import torch


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
