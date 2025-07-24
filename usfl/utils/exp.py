import torch
import random
import numpy as np
import copy
from typing import Dict, List


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


@torch.no_grad()
def fed_average(w: List[Dict]) -> Dict:
    """
    FedAvg: Federated Averaging
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


@torch.no_grad()
def fed_avg_params(params: List[List[torch.nn.Parameter]]) -> List[torch.nn.Parameter]:
    """
    FedAvgParams: Federated Averaging of model parameters
    """
    avg_params: List[torch.nn.Parameter] = []
    for i in range(len(params[0])):
        param_list = [p[i] for p in params]
        avg_param = torch.stack(param_list).mean(dim=0)
        avg_params.append(avg_param)
    return avg_params
