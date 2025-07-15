import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from usfl.socket import SocketCommunicator
from usfl.utils.dataset.exp import AverageMeter


class Client(object):
    def __init__(
        self,
        client_args: dict,
        client_id: int,
        head_model: nn.Module,
        tail_model: nn.Module,
        tokenizer: AutoTokenizer,
        client_device: str,
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
        self.train_logger = train_logger
        self.train_loader = dataset_train
        self.test_loader = dataset_test
        self.local_ep = client_args["epoch"]
        self.lr = client_args['learning_rate']
        if client_id == 0:
            print(
                f"[Client {client_id}] after model loaded,cuda memory: {torch.cuda.memory_allocated(device=client_device) / 1024**3:.4f} GB,max memory: {torch.cuda.max_memory_allocated(device=client_device) / 1024**3:.4f} GB"
            )
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
            for _ in range(self.local_ep):
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
                        "position_embeddings": ([ho.float().cpu() for ho in head_outs[2]] if len(head_outs) > 2 else None),
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
                    self.train_logger.info(
                        f"[Client {self.client_id}] update model,cuda memory: {torch.cuda.memory_allocated(device=self.client_device) / 1024**3:.4f} GB"
                    )
                    torch.cuda.empty_cache()
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
        client.send(
            {
                "client_id": self.client_id,
                "aggregate_client": True,
                "head_state_dict": self.head_model.state_dict(),
                "tail_state_dict": self.tail_model.state_dict(),
            }
        )
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
