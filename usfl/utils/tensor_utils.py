import torch
from typing import Tuple, List, Union


def pad_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor, max_length: int, pad_token_id: int = 0, pad_attention_val: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    _summary_

    Args:
        input_ids (torch.Tensor): _description_
        attention_mask (torch.Tensor): _description_
        max_length (int): _description_
        pad_token_id (int, optional): _description_. Defaults to 0.
        pad_attention_val (int, optional): _description_. Defaults to 0.
        device (_type_, optional): _description_. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    b, s = input_ids.shape
    if s >= max_length:
        # print(f"Input length {s} is already greater than or equal to max length {max_length}.")
        return input_ids, attention_mask
    else:
        padding_length = max_length - s
        padded_input_ids = torch.cat([input_ids, torch.full((b, padding_length), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.full((b, padding_length), pad_attention_val, dtype=attention_mask.dtype, device=attention_mask.device)], dim=-1)

    return padded_input_ids, attention_mask
