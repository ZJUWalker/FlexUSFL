import torch
import random
import numpy as np
import copy


def set_seed(seed: int, cuda: bool = True, deterministic: bool = True) -> None:
    """
    set_seed: Set the seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(torch.cuda.get_device_name(0))


def fed_average(w: list) -> dict:
    """
    FedAvg: Federated Averaging
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
