import os
import numpy as np
import argparse
import time
import copy
import warnings
import logging
import gc

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from usfl.client import Client
from usfl.utils.log_utils import create_logger
from usfl.utils.load_utils import load_client, load_dataset, get_model_layer_num
from usfl.utils.exp import set_seed, fed_average

SEED = 1234
warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)


def client_worker(rank: int, args: dict):
    set_seed(SEED)
    batch_per_sync = args["batch_per_sync"]
    dataset_name = args["dataset"]
    num_clients = args["num_clients"]
    model_name = args["model"]
    model_dir = os.path.join("/share/models", model_name)
    split_point = args["split_point"]
    device = f"cuda:{rank % torch.cuda.device_count()}"
    logger = create_logger(f"client_{rank}.log", console_output=True)
    # ---------------load model and tokenizer --------------------------
    head, tail, tokenizer = load_client(model_dir, args, split_point)
    # -----------------load dataset------------------------------------
    client_dataloaders = load_dataset(dataset_name, tokenizer, list(range(num_clients)))
    data = client_dataloaders[rank]

    client = Client(
        client_args=args,
        client_id=rank,
        head_model=head,
        tail_model=tail,
        tokenizer=tokenizer,
        client_device=device,
        train_logger=logger,
        dataset_train=data["train"],
        dataset_test=data["test"],
    )
    client.train_epoch()
    # w_head, w_tail, avg_loss, peak_memory, client_id = client.train_batches(0, batch_per_sync)
    # result = {"client_id": client_id, "avg_loss": avg_loss, "peak_memory": peak_memory}
    # queue.put(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8000, help="port to listen")
    parser.add_argument("-CK", "--use_sl_ckpt", action="store_true", help="use split l with checkpoint")
    parser.add_argument("-L", "--use_lora", action="store_true", help="use lora")
    parser.add_argument("-Q4", "--use_qlora_4bit", action="store_true", help="use qlora 4bit")
    parser.add_argument("-Q8", "--use_qlora_8bit", action="store_true", help="use qlora 8bit")
    parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="model card")
    parser.add_argument("-B", "--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("-SQ", "--max_sql", type=int, default=512, help="max sequence length")
    parser.add_argument("-NC", "--num_clients", type=int, default=2)
    parser.add_argument("-S", "--step", type=int, default=20)
    parser.add_argument("-V", "--version", type=str, default="v1")
    parser.add_argument("-BPS", "--batch_per_sync", type=int, default=2)
    parser.add_argument("-BPSC", "--batch_per_sync_client", type=int, default=1)
    parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
    parser.add_argument("-E", "--epoch", type=int, default=1)
    parser.add_argument("-SP", "--split_point", type=int, default=3)
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    client_args = parser.parse_args()
    client_args = vars(client_args)
    num_clients = client_args["num_clients"]
    # mp.set_start_method("spawn", force=True)
    print("create client processes")
    mp.spawn(
        client_worker,
        args=(client_args,),
        nprocs=num_clients,
        join=True,
    )
    print("client processes done")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
