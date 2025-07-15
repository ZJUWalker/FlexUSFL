import os
import numpy as np
import argparse
import time
import copy
import warnings
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoConfig

from usfl.client import Client
from usfl.utils.log_utils import create_logger
from usfl.utils.load_utils import load_client, load_dataset, get_model_layer_num
from usfl.socket import SocketCommunicator
from usfl.utils.exp import set_seed, fed_average
from usfl.utils.dataset.exp import AverageMeter

warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)
SEED = 1234


def multi_process_client_task_test(
    client: Client,
    batch_start: int,
    batch_per_sync: int,
    w_locals_head_models: list,
    w_locals_tail_models: list,
    peak_memory_list: list,
    avg_loss_list: list,
):
    w_client_head, w_client_tail, avg_loss, peak_memory, client_id = client.train_batches(batch_start, batch_per_sync)
    w_locals_head_models.append(copy.deepcopy(w_client_head))
    w_locals_tail_models.append(copy.deepcopy(w_client_tail))
    peak_memory_list.append((client_id, peak_memory))
    avg_loss_list.append((client_id, avg_loss))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8000, help="port to listen")
    parser.add_argument("-S", "--step", type=int, default=20, help="number of steps to profile")
    parser.add_argument("-CK", "--use_sl_ckpt", action="store_true", help="use split l with checkpoint")
    parser.add_argument("-L", "--use_lora", action="store_true", help="use lora")
    parser.add_argument("-Q4", "--use_qlora_4bit", action="store_true", help="use qlora 4bit")
    parser.add_argument("-Q8", "--use_qlora_8bit", action="store_true", help="use qlora 8bit")
    parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="model card")
    parser.add_argument("-B", "--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("-SQ", "--max_sql", type=int, default=512, help="max sequence length")
    parser.add_argument("-NC", "--num_clients", type=int, default=1, help="number of clients")
    parser.add_argument("-BF", "--buffer_size", type=int, default=4096, help="buffer size for socket communication")
    parser.add_argument("-V", "--version", type=str, default="v1", help="version of the code")
    args = parser.parse_args()
    client_args = vars(args)
    set_seed(SEED)
    # ================================init variables=====================================
    available_devices = [0, 1, 2]
    epochs = 1
    frac = 1
    lr = 0.00005
    batch_per_sync = 20
    dataset_name = "gsm8k"
    client_ids = [i for i in range(client_args["num_clients"])]
    train_logger = create_logger(
        log_file_name="train.log",
        console_output=True,
    )
    matrix_logger = create_logger(
        log_file_name="matrix.log",
        console_output=True,
    )
    # ================================load model and dataset======================================
    model_dir = os.path.join("/share/models", client_args["model"])
    split_point = get_model_layer_num(model_dir) // 5
    head, tail, tokenizer = load_client(model_dir, client_args, split_point)
    # =====================================================================
    client_dataloaders = load_dataset(dataset_name, tokenizer, client_ids)
    for client_id, dataloaders in client_dataloaders.items():
        train_logger.info(f"Client {client_id}:")
        train_logger.info(f"Train dataset size: {len(dataloaders['train'])}")
        train_logger.info(f"Test dataset size: {len(dataloaders['test'])}")

    w_glob_head = head.state_dict()
    w_glob_tail = tail.state_dict()
    # =====================================================================
    train_logger.info(
        "Training... \n{} client start split_point: {} ,buffer size: {}B ,batch size: {} ,use lora {} ,use qlora {} ,use checkpoint {}".format(
            client_args["model"],
            split_point,
            client_args["buffer_size"],
            client_args["batch_size"],
            client_args["use_lora"],
            ("qlora_4bit" if client_args["use_qlora_4bit"] else "qlora_8bit" if client_args["use_qlora_8bit"] else "no_quantization"),
            client_args["use_sl_ckpt"],
        )
    )
    # ================================init clients=====================================
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    # =====================================================================
    eval_losses = []
    for iter in range(epochs):
        m = max(int(frac * client_args["num_clients"]), 1)
        idxs_users = np.random.choice(client_ids, m, replace=False)
        num_batches = min([len(client_dataloaders[idx]["train"]) for idx in idxs_users])

        for batch_start in range(0, num_batches, batch_per_sync):
            train_logger.info(f"batch_start: {batch_start}, batch_per_sync: {batch_per_sync}")
            w_locals_head_models = manager.list()
            w_locals_tail_models = manager.list()
            peak_memory_list = manager.list()
            avg_loss_list = manager.list()
            # client = Client(
            #     client_args=client_args,
            #     client_id=0,
            #     head_model=copy.deepcopy(head),
            #     tail_model=copy.deepcopy(tail),
            #     tokenizer=tokenizer,
            #     client_device=f"cuda:{available_devices[0]}",
            #     lr=lr,
            #     train_logger=train_logger,
            #     dataset_train=client_dataloaders[0]["train"],
            #     dataset_test=client_dataloaders[0]["test"],
            # )
            # # eval_loss = client.evaluating_batches(client_dataloaders)
            # train_logger.info(f"Client 0, eval_loss: {eval_loss:.4f}")
            # eval_losses.append(eval_loss)

            # if hasattr(client, "head_model") and client.head_model is not None:
            #     client.head_model.cpu()
            # if hasattr(client, "tail_model") and client.tail_model is not None:
            #     client.tail_model.cpu()
            # del client
            # gc.collect()
            # torch.cuda.empty_cache()

            processes = []
            for client_id in idxs_users:
                client = Client(
                    client_args=client_args,
                    client_id=client_id,
                    head_model=copy.deepcopy(head),
                    tail_model=copy.deepcopy(tail),
                    tokenizer=tokenizer,
                    client_device=f"cuda:{available_devices[client_id % len(available_devices)]}",
                    lr=lr,
                    train_logger=train_logger,
                    dataset_train=client_dataloaders[client_id]["train"],
                    dataset_test=client_dataloaders[client_id]["test"],
                )
                process = mp.Process(
                    target=multi_process_client_task_test,
                    args=(
                        client,
                        batch_start,
                        batch_per_sync,
                        w_locals_head_models,
                        w_locals_tail_models,
                        peak_memory_list,
                        avg_loss_list,
                    ),
                    name=f"client_{client_id}",
                )
                processes.append(process)

            start_time = time.time()
            for process in processes:
                print(f"start process {process.name}")
                process.start()
            for process in processes:
                process.join()
            end_time = time.time()
            batches_time = end_time - start_time

            w_glob_head_model = fed_average(w_locals_head_models)
            train_logger.info("w_glob_head_model is updated")
            w_glob_tail_model = fed_average(w_locals_tail_models)
            train_logger.info("w_glob_tail_model is updated")
            head.load_state_dict(w_glob_head_model)
            train_logger.info("load w_glob_head_model to net_glob_head")
            tail.load_state_dict(w_glob_tail_model)
            train_logger.info("load w_glob_tail_model to net_glob_tail")

            matrix_logger.info(f"{'-' *10}Epoch {iter}, step {batch_start+batch_per_sync}, clients {idxs_users}{'-' *10}")
            for client in idxs_users:
                matrix_logger.info(
                    f"Client {client}, "
                    f"batches_time: {batches_time:.4f}s, "
                    f"peak_memory: {peak_memory_list[client_ids.index(client)][1]:.4f}GB, "
                    f"avg_loss: {avg_loss_list[client_ids.index(client)][1]:.4f}"
                )

    np.save(
        f"eval_losses_num_clients_{client_args['num_clients']}_version_{client_args['version']}.npy",
        np.array(eval_losses),
    )


if __name__ == "__main__":
    main()
