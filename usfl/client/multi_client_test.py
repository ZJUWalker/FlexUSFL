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

from usfl.utils.log_utils import create_logger
from usfl.utils.load_utils import load_client, load_dataset
from usfl.socket import SocketCommunicator
from usfl.utils.exp import set_seed, fed_average
from usfl.utils.dataset.exp import AverageMeter

warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)


class Client(object):
    def __init__(
        self,
        client_args: dict,
        client_id: int,
        head_model: nn.Module,
        tail_model: nn.Module,
        tokenizer: AutoTokenizer,
        client_device: str,
        lr: float,
        train_logger: logging.Logger,
        dataset_train: Dataset,
        dataset_test: Dataset,
    ):
        self.client_device = client_device
        self.client_args = client_args
        self.client_id = client_id
        self.head_model = head_model.to(self.client_device)
        self.tail_model = tail_model.to(self.client_device)
        self.tokenizer = tokenizer
        self.lr = lr
        self.train_logger = train_logger
        self.train_loader = dataset_train
        self.test_loader = dataset_test
        self.local_ep = 1
        self.optimizer_head = torch.optim.Adam(self.head_model.parameters(), lr=self.lr)
        self.optimizer_tail = torch.optim.Adam(self.tail_model.parameters(), lr=self.lr)
        self.avg_loss = AverageMeter()
        self.simulate_delay = True

    def train_batches(self, batch_start: int, batch_per_sync: int):
        self.head_model.train()
        self.tail_model.train()
        torch.cuda.reset_peak_memory_stats(device=self.client_device)
        self.avg_loss.reset()

        with SocketCommunicator(
            host="localhost",
            port=self.client_args["port"],
            is_server=False,
            buffer_size=4096,
            similuate_delay=self.simulate_delay,
        ) as client:
            for iter in range(self.local_ep):
                for batch_idx, batch in enumerate(self.train_loader):
                    if batch_idx < batch_start:
                        continue
                    if batch_idx >= batch_start + batch_per_sync:
                        break
                    input_ids = batch["input_ids"].to(self.client_device)
                    attention_mask = batch["attention_mask"].to(self.client_device)
                    self.optimizer_head.zero_grad()
                    self.optimizer_tail.zero_grad()
                    labels = batch["labels"].to(self.client_device) if "labels" in batch else input_ids
                    head_outs = self.head_model.forward(input_ids, attention_mask)
                    head_outs[0].requires_grad_(True)
                    attention_mask = head_outs[1]
                    head_out_to_server = {
                        "client_id": self.client_id,
                        "activation": head_outs[0].cpu(),
                        "attention_mask": (head_outs[1].cpu() if head_outs[1] is not None else None),
                        "position_embeddings": ([ho.float().cpu().tolist() for ho in head_outs[2]] if len(head_outs) > 2 else None),
                        "is_training": True,
                    }
                    if head_out_to_server["attention_mask"] is None:
                        head_out_to_server.pop("attention_mask")
                    if head_out_to_server["position_embeddings"] is None:
                        head_out_to_server.pop("position_embeddings")
                    client.send(head_out_to_server)
                    self.train_logger.info(f"[Client {self.client_id}] send head_out_to_server")
                    server_forward_output = client.receive()
                    self.train_logger.info(
                        "[Client {0}] receive server_activation: {1}".format(self.client_id, server_forward_output["server_activation"].shape)
                    )
                    activation_from_server = torch.tensor(
                        server_forward_output["server_activation"],
                        device=self.client_device,
                        dtype=torch.float32,
                        requires_grad=True,
                    )
                    output = self.tail_model.forward(
                        hidden_states=activation_from_server,
                        attention_mask=attention_mask,
                        position_embeddings=head_outs[2] if len(head_outs) > 2 else None,
                        labels=labels,
                    )
                    loss = output.loss
                    self.avg_loss.update(loss.item())
                    self.train_logger.info(f"[Client {self.client_id}] compute loss: {loss.item()}")
                    loss.backward()
                    grads_to_server = activation_from_server.grad.cpu()
                    tail_grads_to_server = {
                        "client_id": self.client_id,
                        "gradient": grads_to_server,
                    }
                    # time.sleep(2)
                    client.send(tail_grads_to_server)
                    self.train_logger.info(f"[Client {self.client_id}] send tail_grads_to_server")
                    server_backward_output = client.receive()
                    self.train_logger.info(f"[Client {self.client_id}] receive server_gradient")
                    grads_from_server = torch.tensor(
                        server_backward_output["server_gradient"],
                        device=self.client_device,
                        dtype=torch.float32,
                    )
                    head_outs[0].backward(grads_from_server)
                    self.optimizer_head.step()
                    self.optimizer_tail.step()
            # 发送聚合信号并等待服务器响应
            client.send(
                {
                    "client_id": self.client_id,
                    "aggregate": True,
                    "step": batch_start + batch_per_sync,
                }
            )
            self.train_logger.info(f"[Client {self.client_id}] send aggregate signal")
            # server_aggregate_status = client.receive()

        peak_memory = torch.cuda.max_memory_allocated(device=self.client_device) / 1024**3

        self.head_model.cpu()
        self.tail_model.cpu()
        return (
            self.head_model.state_dict(),
            self.tail_model.state_dict(),
            self.avg_loss.avg,
            peak_memory,
            self.client_id,
        )

    def evaluating_batches(self, client_dataloaders):
        self.head_model.eval()
        self.tail_model.eval()

        eval_loss = AverageMeter()
        eval_loss.reset()
        self.train_logger.info(f"[Client {self.client_id}] start evaluating")
        with SocketCommunicator(
            host="localhost",
            port=self.client_args["port"],
            is_server=False,
            buffer_size=4096,
            similuate_delay=True,
        ) as client:
            for client_id, dataloaders in client_dataloaders.items():
                for batch_idx, batch in enumerate(dataloaders["test"]):
                    input_ids = batch["input_ids"].to(self.client_device)
                    attention_mask = batch["attention_mask"].to(self.client_device)
                    labels = batch["labels"].to(self.client_device) if "labels" in batch else input_ids
                    head_outs = self.head_model.forward(input_ids, attention_mask)
                    head_outs[0].requires_grad_(True)
                    attention_mask = head_outs[1]
                    head_out_to_server = {
                        "client_id": self.client_id,
                        "activation": head_outs[0].cpu(),
                        "attention_mask": (head_outs[1].cpu() if head_outs[1] is not None else None),
                        "position_embeddings": ([ho.float().cpu().tolist() for ho in head_outs[2]] if len(head_outs) > 2 else None),
                        "is_training": False,
                    }
                    if head_out_to_server["attention_mask"] is None:
                        head_out_to_server.pop("attention_mask")
                    if head_out_to_server["position_embeddings"] is None:
                        head_out_to_server.pop("position_embeddings")
                    client.send(head_out_to_server)
                    server_forward_output = client.receive()
                    activation_from_server = torch.tensor(
                        server_forward_output["server_activation"],
                        device=self.client_device,
                        dtype=torch.float32,
                        requires_grad=True,
                    )
                    output = self.tail_model.forward(
                        hidden_states=activation_from_server,
                        attention_mask=attention_mask,
                        position_embeddings=head_outs[2] if len(head_outs) > 2 else None,
                        labels=labels,
                    )
                    loss = output.loss
                    # print(f"loss: {loss.item()}")
                    eval_loss.update(loss.item())
                    if batch_idx % 50 == 0:
                        self.train_logger.info(f"[Client {self.client_id}] evaluating batch {batch_idx}, loss: {eval_loss.avg}")
            return eval_loss.avg


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


if __name__ == "__main__":
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
    parser.add_argument("--num_clients", type=int, default=1, help="number of clients")
    parser.add_argument("--buffer_size", type=int, default=4096, help="buffer size for socket communication")
    parser.add_argument("--version", type=str, default="v1", help="version of the code")
    args = parser.parse_args()
    client_args = vars(args)
    # =====================================================================
    train_logger = create_logger(
        log_file_name="train.log",
        console_output=True,
    )
    matrix_logger = create_logger(
        log_file_name="matrix.log",
        console_output=True,
    )
    # =====================================================================
    torch.cuda.init()
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    SEED = 1234
    set_seed(SEED)
    # =====================================================================
    available_devices = [0, 1, 2]
    epochs = 1
    frac = 1
    lr = 0.00005
    dataset_name = "gsm8k"
    client_ids = [i for i in range(client_args["num_clients"])]
    # =====================================================================
    model_dir = os.path.join("/share/models", client_args["model"])
    split_point = AutoConfig.from_pretrained(model_dir).num_hidden_layers // 5
    head, tail, tokenizer = load_client(model_dir, client_args, split_point)
    # =====================================================================
    client_dataloaders = load_dataset(dataset_name, tokenizer, client_ids)
    for client_id, dataloaders in client_dataloaders.items():
        train_logger.info(f"Client {client_id}:")
        train_logger.info(f"  Train dataset size: {len(dataloaders['train'])}")
        train_logger.info(f"   Test dataset size: {len(dataloaders['test'])}")

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
    # =====================================================================
    eval_losses = []
    for iter in range(epochs):
        m = max(int(frac * client_args["num_clients"]), 1)
        idxs_users = np.random.choice(client_ids, m, replace=False)
        num_batches = min([len(client_dataloaders[idx]["train"]) for idx in idxs_users])
        batch_per_sync = 20

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
