import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from usfl.socket import SocketCommunicator
from usfl.utils.dataset.exp import AverageMeter
from typing import Dict, List, Tuple
from usfl.utils.tensor_utils import pad_inputs
from usfl.utils.timestamp_recorder import GanttChartData
import json
from dataclasses import asdict
import os


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
        batch_num: int,
        lag_ratio: float = 0.0,
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
        self.batch_num = batch_num
        self.local_ep = client_args["epoch"]
        self.lr = client_args["learning_rate"]
        if client_id == 0:
            print(
                f"[Client {client_id}] after model loaded,cuda memory: {torch.cuda.memory_allocated(device=client_device) / 1024**3:.4f} GB,max memory: {torch.cuda.max_memory_allocated(device=client_device) / 1024**3:.4f} GB"
            )
        self.optimizer_head = torch.optim.Adam(self.head_model.parameters(), lr=self.lr)
        self.optimizer_tail = torch.optim.Adam(self.tail_model.parameters(), lr=self.lr)
        self.avg_loss = AverageMeter()
        self.simulate_delay = True
        self.compute_time = 0
        self.profile_data: List[GanttChartData] = [GanttChartData(batch_idx=i, client_id=client_id) for i in range(self.batch_num)]
        self.lag_ratio = lag_ratio
        self._did_global_barrier = False
        self._barrier_times = 0  # 新增：记录已经同步了几次

    def train_epoch(self, barrier=None):
        """
        Train the client model
        """
        # self.head_model.train()
        # self.tail_model.train()

        torch.cuda.reset_peak_memory_stats(device=self.client_device)
        self.avg_loss.reset()
        batch_per_sync = self.client_args["batch_per_sync"]
        print("max seq len: ", self.client_args["max_seq_len"])
        with SocketCommunicator(
            host="localhost",
            port=self.client_args["port"],
            is_server=False,
            buffer_size=4096,
            similuate_delay=self.simulate_delay,
            lag_ratio=self.lag_ratio,
        ) as client:
            start_time = time.time()

            for epoch in range(self.local_ep):
                self.train_logger.info(f"[Client {self.client_id}] start train epoch {epoch+1}, data loader len: {len(self.train_loader)}")
                n = 0
                print(f"[Client {self.client_id}] connected to server, time={time.time()}")
                for batch_idx, batch in enumerate(self.train_loader, 0):

                    loss = self.train_batch(batch, client, batch_idx, barrier)
                    self.train_logger.info(f"[Client {self.client_id}] train epoch {epoch+1}, batch {batch_idx}/{self.batch_num}, loss: {loss:.4f}")

                    if batch_idx % batch_per_sync == 0 or batch_idx == self.batch_num:
                        self.train_logger.info(f"[Client {self.client_id} {batch_idx//batch_per_sync}-th Aggregation] send aggregate_client signal")
                        head_params = [p.cpu() for p in self.head_model.parameters() if p.requires_grad]
                        tail_params = [p.cpu() for p in self.tail_model.parameters() if p.requires_grad]
                        # self.train_logger.info(f"[Client {self.client_id}] prepare params to send {[p.shape for p in head_params + tail_params]}")
                        # peak_memory = torch.cuda.max_memory_allocated(device=self.client_device) / 1024**3
                        client.send(
                            {
                                "client_id": self.client_id,
                                "aggregate": True,
                                "step": batch_idx,
                                "loss": self.avg_loss.avg,
                                "head_params": head_params,
                                "tail_params": tail_params,
                            }
                        )
                        # self.train_logger.info(
                        #     f"[Client {self.client_id}] send aggregate_client signal, batch_idx: {batch_idx}, avg_loss: {self.avg_loss.avg}, peak_memory: {peak_memory:.4f} GB"
                        # )
                        client_aggregate_result = client.receive()  # blocking util receive server aggregate finished signal
                        self.train_logger.info(
                            f"[Client {self.client_id} Aggregated],avg loss {client_aggregate_result['loss']:.4f},"
                            f"max memory: {torch.cuda.max_memory_allocated(device=self.client_device) / 1024**3:.4f} GB"
                            f"max memory reserved: {torch.cuda.max_memory_reserved(device=self.client_device) / 1024**3:.4f} GB"
                            f"time used:{time.time() - start_time:.2f} s"
                        )
                        # update model
                        self._update_model_params(self.head_model, client_aggregate_result["head_params"])
                        self._update_model_params(self.tail_model, client_aggregate_result["tail_params"])
                        self.avg_loss.reset()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device=self.client_device)
                        # self.train_logger.info(f"[Client {self.client_id}] updated model")
                        # if batch_idx == batch_per_sync * 1:
                        #     break  # for test only

                    # if batch_idx == 10:
                    #     break
                    n += 1
                    if n >= 5:
                        client.send("stop")
                        # print(self.profile_data[])
                        data_to_save = [asdict(x) for x in self.profile_data[:5]]
                        save_dir = os.path.join(
                            "./vis",
                            f"version_{self.client_args['version']}",
                            f"model_{self.client_args['model'].split('/')[-1]}",
                            f"dataset_{self.client_args['dataset']}",
                            f"lag_{self.client_args['lag_ratio']}",
                            f"client_num_{self.client_args['num_clients']}",
                            # f"bps_{self.client_args['batch_per_sync']}",
                            f"order_{self.client_args['queue_order']}",
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"client_{self.client_id}_profile_data_sample.json")
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                        break
                    # if batch_idx == self.batch_num:
                    #     break
            end_time = time.time()
            self.train_logger.info(
                f"[Client {self.client_id} Finished] train epoch time: {end_time - start_time:.2f} s, compute time: {self.compute_time:.2f} s"
            )
        pass

    def _update_model_params(self, model: nn.Module, new_params: List[nn.Parameter]):
        i = 0
        for p in model.parameters():
            if p.requires_grad:
                p.data.copy_(new_params[i].data, non_blocking=True)
                i += 1
        torch.cuda.synchronize(device=self.client_device)
        pass

    def train_batch(self, batch: Dict, client: SocketCommunicator, batch_idx: int, barrier=None):
        # print(f"Client {self.client_id} start training batch_idx={batch_idx}, time={time.time()}")
        input_ids = batch["input_ids"].to(self.client_device)
        attention_mask = batch["attention_mask"].to(self.client_device)

        input_ids, attention_mask = pad_inputs(input_ids, attention_mask, self.client_args["max_seq_len"])
        labels = input_ids
        # Forward pass through head model
        print(f"Client {self.client_id} loaded data to device, batch_idx={batch_idx}, time={time.time()}")
        torch.cuda.current_stream().synchronize()

        if barrier is not None and self._barrier_times < 1:
            print(
                f"[Client {self.client_id}] barrier before forward pass " f"(#{self._barrier_times + 1}), batch_idx={batch_idx}, time={time.time()}"
            )
            barrier.wait()
            self._barrier_times += 1

        print(f"Client {self.client_id} passed barrier, batch_idx={batch_idx}, time={time.time()}")
        self.profile_data[batch_idx].head_fwd_timestamp[0] = time.time()
        print(f"Client {self.client_id} start head forward, batch_idx={batch_idx}, time={time.time()}")
        head_outs = self.head_model.forward(input_ids, attention_mask)
        head_outs[0].requires_grad_(True)
        attention_mask = head_outs[1]
        torch.cuda.current_stream().synchronize()
        if self.lag_ratio > 1.0:
            delay_time = (time.time() - self.profile_data[batch_idx].head_fwd_timestamp[0]) * (self.lag_ratio - 1.0)
            # print(f"Client {self.client_id} simulating lag for {delay_time:.2f} seconds")
            time.sleep(delay_time)
        self.profile_data[batch_idx].head_fwd_timestamp[1] = time.time()

        # Send head outputs to server
        torch.cuda.current_stream().synchronize()
        self.profile_data[batch_idx].head_fwd_send_timestamp[0] = time.time()
        head_out_to_server = {
            "client_id": self.client_id,
            "batch_idx": batch_idx,
            "activation": head_outs[0].cpu(),
            "attention_mask": (head_outs[1].cpu() if head_outs[1] is not None else None),
            "position_embeddings": ([ho.float().cpu() for ho in head_outs[2]] if len(head_outs) > 2 else None),
            "is_training": True,
        }
        # print(f"activation shape: {head_out_to_server['activation'].shape}")
        if head_out_to_server["attention_mask"] is None:
            head_out_to_server.pop("attention_mask")
        if head_out_to_server["position_embeddings"] is None:
            head_out_to_server.pop("position_embeddings")

        mb, head_send_time = client.send(head_out_to_server)
        print(f"Client {self.client_id} sent head_out to server, batch_idx={batch_idx}, size={mb:.4f} MB, send_time={head_send_time:.4f} s")

        torch.cuda.current_stream().synchronize()
        self.profile_data[batch_idx].head_fwd_send_timestamp[1] = time.time()
        print(f"client {self.client_id} sent head_out to server, batch_idx={batch_idx}, timestamp={time.time()}")
        # self.train_logger.info(f"[Client {self.client_id}] send head_out_to_server")
        server_forward_output = client.receive()

        # Forward pass through tail model
        # self.train_logger.info("[Client {0}] receive server_activation: {1}".format(self.client_id, server_forward_output["server_activation"].shape))
        torch.cuda.current_stream().synchronize()
        self.profile_data[batch_idx].tail_fwd_timestamp[0] = time.time()
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
        if self.lag_ratio > 1.0:
            delay_time = (time.time() - self.profile_data[batch_idx].tail_fwd_timestamp[0]) * (self.lag_ratio - 1.0)
            # print(f"Client {self.client_id} simulating lag for {delay_time:.2f} seconds")
            time.sleep(delay_time)

        torch.cuda.current_stream().synchronize()
        self.profile_data[batch_idx].tail_fwd_timestamp[1] = time.time()
        torch.cuda.empty_cache()
        loss = output.loss
        self.avg_loss.update(loss.item())
        # self.train_logger.info(f"[Client {self.client_id}] compute loss: {loss.item()}")

        # Backward pass through tail model
        torch.cuda.current_stream().synchronize()
        self.profile_data[batch_idx].tail_bwd_timestamp[0] = time.time()
        loss.backward()
        torch.cuda.current_stream().synchronize()
        torch.cuda.empty_cache()
        grads_to_server = activation_from_server.grad.cpu()
        if self.lag_ratio > 1.0:
            delay_time = (time.time() - self.profile_data[batch_idx].tail_bwd_timestamp[0]) * (self.lag_ratio - 1.0)
            # print(f"Client {self.client_id} simulating lag for {delay_time:.2f} seconds")
            time.sleep(delay_time)
        self.profile_data[batch_idx].tail_bwd_timestamp[1] = time.time()

        # send tail gradients to server
        self.profile_data[batch_idx].tail_bwd_send_timestamp[0] = time.time()
        tail_grads_to_server = {
            "client_id": self.client_id,
            "batch_idx": batch_idx,
            "gradient": grads_to_server,
        }
        # time.sleep(2)
        client.send(tail_grads_to_server)
        self.profile_data[batch_idx].tail_bwd_send_timestamp[1] = time.time()

        # self.train_logger.info(f"[Client {self.client_id}] send tail_grads_to_server")
        server_backward_output = client.receive()
        # self.train_logger.info(f"[Client {self.client_id}] receive server_gradient")

        # Backward pass through head model and update
        torch.cuda.current_stream().synchronize()
        self.profile_data[batch_idx].head_bwd_timestamp[0] = time.time()
        grads_from_server = torch.tensor(
            server_backward_output["server_gradient"],
            device=self.client_device,
            dtype=torch.float32,
        )
        head_outs[0].backward(grads_from_server)
        torch.cuda.current_stream().synchronize()
        if self.lag_ratio > 1.0:
            delay_time = (time.time() - self.profile_data[batch_idx].head_bwd_timestamp[0]) * (self.lag_ratio - 1.0)
            # print(f"Client {self.client_id} simulating lag for {delay_time:.2f} seconds")
            time.sleep(delay_time)
        self.profile_data[batch_idx].head_bwd_timestamp[1] = time.time()

        torch.cuda.current_stream().synchronize()
        self.profile_data[batch_idx].client_step_timestamp[0] = time.time()
        self.optimizer_head.step()
        self.optimizer_tail.step()
        self.optimizer_head.zero_grad()
        self.optimizer_tail.zero_grad()

        torch.cuda.current_stream().synchronize()
        if self.lag_ratio > 1.0:
            delay_time = (time.time() - self.profile_data[batch_idx].client_step_timestamp[0]) * (self.lag_ratio - 1.0)
            # print(f"Client {self.client_id} simulating lag for {delay_time:.2f} seconds")
            time.sleep(delay_time)
        self.profile_data[batch_idx].client_step_timestamp[1] = time.time()
        torch.cuda.empty_cache()
        # self.train_logger.info(
        #     f"[Client {self.client_id}] update model,cuda memory: {torch.cuda.memory_allocated(device=self.client_device) / 1024**3:.4f} GB"
        # )
        # torch.cuda.empty_cache()

        # print(self.profile_data[:batch_idx])

        return loss

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
                "head_params": list(filter(lambda p: p.requires_grad, self.head_model.parameters())),
                "tail_params": list(filter(lambda p: p.requires_grad, self.tail_model.parameters())),
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
