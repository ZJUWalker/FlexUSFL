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
from contextlib import contextmanager


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
        enable_profiling: bool = True,
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
        self.enable_profiling = enable_profiling
        if enable_profiling:
            self.profile_data: List[GanttChartData] = [GanttChartData(batch_idx=i, client_id=client_id) for i in range(self.batch_num)]
        else:
            print("profiling disabled")
        self.lag_ratio = lag_ratio
        self._did_global_barrier = False
        self._barrier_times = 0  # 新增：记录已经同步了几次
        self.start_time = time.time()
        print(f"client {self.client_id} train_loader len: {len(self.train_loader)}")

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

        MAX_ROUNDS = 6
        current_round = 0
        stop_training = False  # 用于跳出外层 epoch 循环

        with SocketCommunicator(
            host="localhost",
            port=self.client_args["port"],
            is_server=False,
            buffer_size=4096,
            similuate_delay=self.simulate_delay,
            lag_ratio=self.lag_ratio,
        ) as client:
            start_time = time.time()
            aggregated = False
            for epoch in range(self.local_ep):

                if stop_training:
                    break

                self.train_logger.info(f"[Client {self.client_id}] start train epoch {epoch+1}, data loader len: {len(self.train_loader)}")
                print(f"[Client {self.client_id}] connected to server, time={time.time()}")
                for batch_idx, batch in enumerate(self.train_loader, 0):
                    if aggregated:
                        barrier.wait()
                        if self.enable_profiling:
                            self.profile_data[batch_idx - 1].client_fed_avg_timestamp[1] = time.time()
                        aggregated = False
                    loss = self.train_batch(batch, client, batch_idx, barrier)
                    self.train_logger.info(
                        f"[Client {self.client_id}] train epoch {epoch+1}, batch {batch_idx}/{self.batch_num}, loss: {loss:.4f}"
                    )

                    if batch_idx % batch_per_sync == 0 or batch_idx == self.batch_num:
                        self.train_logger.info(
                            f"[Client {self.client_id} {batch_idx//batch_per_sync}-th Aggregation] send aggregate_client signal"
                        )
                        head_params = [p.cpu() for p in self.head_model.parameters() if p.requires_grad]
                        tail_params = [p.cpu() for p in self.tail_model.parameters() if p.requires_grad]

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
                        client_aggregate_result = client.receive()  # blocking util receive server aggregate finished signal
                        self.train_logger.info(
                            f"[Client {self.client_id} Aggregated],avg loss {client_aggregate_result['loss']:.4f},"
                            f"max memory: {torch.cuda.max_memory_allocated(device=self.client_device) / 1024**3:.4f} GB"
                            f"max memory reserved: {torch.cuda.max_memory_reserved(device=self.client_device) / 1024**3:.4f} GB"
                            f"time used:{time.time() - self.start_time:.2f} s"
                        )
                        # update model
                        self._update_model_params(self.head_model, client_aggregate_result["head_params"])
                        self._update_model_params(self.tail_model, client_aggregate_result["tail_params"])
                        self.train_logger.info(f"[Client {self.client_id}] updated model")
                        self.avg_loss.reset()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device=self.client_device)
                        if self.enable_profiling:
                            torch.cuda.synchronize(device=self.client_device)
                            self.profile_data[batch_idx].client_fed_avg_timestamp[0] = client_aggregate_result["aggregate_start_time"]
                        aggregated = True

                        current_round += 1
                        self.train_logger.info(f"[Client {self.client_id}] Completed Round {current_round}/{MAX_ROUNDS}")

                        if current_round >= MAX_ROUNDS:
                            self.train_logger.info(f"[Client {self.client_id}] Reached {MAX_ROUNDS} rounds. Stopping...")

                            # 发送停止信号给 Server (Server 需要字典格式)
                            client.send({"stop": True, "client_id": self.client_id})

                            # 保存 Profiling 数据 (如果需要)
                            if self.enable_profiling:
                                data_to_save = [asdict(x) for x in self.profile_data[: batch_idx + 1]]  # 只保存跑过的数据
                                save_dir = os.path.join(
                                    "./vis",
                                    f"version_{self.client_args['version']}",
                                    f"model_{self.client_args['model'].split('/')[-1]}",
                                    f"dataset_{self.client_args['dataset']}",
                                    f"lag_{self.client_args['lag_ratio']}",
                                    f"client_num_{self.client_args['num_clients']}",
                                    f"order_{self.client_args['queue_order']}",
                                )
                                os.makedirs(save_dir, exist_ok=True)
                                save_path = os.path.join(save_dir, f"client_{self.client_id}_profile_data.json")
                                with open(save_path, "w", encoding="utf-8") as f:
                                    json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                                self.train_logger.info(f"Profile data saved to {save_path}")

                            stop_training = True  # 设置标志位
                            break  # 跳出 batch 循环

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
        # print(f"[Client {self.client_id}] start train batch {batch_idx}")
        input_ids = batch["input_ids"].to(self.client_device)
        attention_mask = batch["attention_mask"].to(self.client_device)

        input_ids, attention_mask = pad_inputs(input_ids, attention_mask, self.client_args["max_seq_len"])
        labels = input_ids
        # Forward pass through head model
        if barrier is not None and self._barrier_times < 2:
            barrier.wait()
            self._barrier_times += 1
            self.start_time = time.time()  # 记录同步时间

        with self._profile_scope(batch_idx, "head_fwd_timestamp") as head_fwd_start_time:
            head_outs = self.head_model.forward(input_ids, attention_mask)
            head_outs[0].requires_grad_(True)
            attention_mask = head_outs[1]
            torch.cuda.current_stream().synchronize()
            if self.lag_ratio > 1.0:
                actual_compute_time = time.time() - head_fwd_start_time
                delay_time = actual_compute_time * (self.lag_ratio - 1.0)
                time.sleep(delay_time)

        # Send head outputs to server
        with self._profile_scope(batch_idx, "head_fwd_send_timestamp"):
            head_out_to_server = {
                "client_id": self.client_id,
                "batch_idx": batch_idx,
                "activation": head_outs[0].cpu(),
                "attention_mask": (head_outs[1].cpu() if head_outs[1] is not None else None),
                "position_embeddings": ([ho.float().cpu() for ho in head_outs[2]] if len(head_outs) > 2 else None),
                "is_training": True,
                "head_fwd_start_time": head_fwd_start_time,
            }
            if head_out_to_server["attention_mask"] is None:
                head_out_to_server.pop("attention_mask")
            if head_out_to_server["position_embeddings"] is None:
                head_out_to_server.pop("position_embeddings")

            mb, head_send_time = client.send(head_out_to_server)

        server_forward_output = client.receive()
        # print(f"[Client {self.client_id}] received server output batch {batch_idx}")
        # Forward pass through tail model
        with self._profile_scope(batch_idx, "tail_fwd_timestamp") as tail_start_time:
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
                torch.cuda.current_stream().synchronize()  # 必须先同步，确保 forward 跑完了
                # 使用 tail_start_time 而不是 profile_data
                actual_compute_time = time.time() - tail_start_time
                delay_time = actual_compute_time * (self.lag_ratio - 1.0)
                time.sleep(delay_time)

        torch.cuda.empty_cache()
        loss = output.loss
        self.avg_loss.update(loss.item())
        # self.train_logger.info(f"[Client {self.client_id}] compute loss: {loss.item()}")

        # Backward pass through tail model
        with self._profile_scope(batch_idx, "tail_bwd_timestamp") as tail_bwd_start_time:
            loss.backward()
            torch.cuda.current_stream().synchronize()
            torch.cuda.empty_cache()
            grads_to_server = activation_from_server.grad.cpu()
            if self.lag_ratio > 1.0:
                actual_compute_time = time.time() - tail_bwd_start_time
                delay_time = actual_compute_time * (self.lag_ratio - 1.0)
                time.sleep(delay_time)

        # send tail gradients to server
        with self._profile_scope(batch_idx, "tail_bwd_send_timestamp"):
            tail_grads_to_server = {
                "client_id": self.client_id,
                "batch_idx": batch_idx,
                "gradient": grads_to_server,
                "tail_start_time": tail_start_time,
            }
            # time.sleep(2)
            client.send(tail_grads_to_server)

        server_backward_output = client.receive()

        # Backward pass through head model and update
        with self._profile_scope(batch_idx, "head_bwd_timestamp") as head_bwd_start_time:
            grads_from_server = torch.tensor(
                server_backward_output["server_gradient"],
                device=self.client_device,
                dtype=torch.float32,
            )
            head_outs[0].backward(grads_from_server)
            torch.cuda.current_stream().synchronize()
            if self.lag_ratio > 1.0:
                delay_time = (time.time() - head_bwd_start_time) * (self.lag_ratio - 1.0)
                time.sleep(delay_time)

        with self._profile_scope(batch_idx, "client_step_timestamp") as client_step_start_time:
            self.optimizer_head.step()
            self.optimizer_tail.step()
            self.optimizer_head.zero_grad()
            self.optimizer_tail.zero_grad()

            torch.cuda.current_stream().synchronize()
            if self.lag_ratio > 1.0:
                delay_time = (time.time() - client_step_start_time) * (self.lag_ratio - 1.0)
                time.sleep(delay_time)

        torch.cuda.empty_cache()

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
                        "[Client {0}] receive server_activation: {1}".format(
                            self.client_id, server_forward_output["server_activation"].shape
                        )
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

    @contextmanager
    def _profile_scope(self, batch_idx: int, attr_name: str):
        # -------------------------------------------------------
        # 1. 强制同步：确保 GPU 完成了之前的任务
        # -------------------------------------------------------
        if "cuda" in str(self.client_device):
            torch.cuda.synchronize(self.client_device)

        # -------------------------------------------------------
        # 2. 只有同步完成后，才记录开始时间 (这就是你要的 Accurate Start Time)
        # -------------------------------------------------------
        start_time = time.time()

        # 3. 将这个精准的开始时间 yield 给 with 语句内部
        try:
            yield start_time
        finally:
            # 4. 业务代码跑完后，再次同步并记录结束时间
            if "cuda" in str(self.client_device):
                torch.cuda.synchronize(self.client_device)
            end_time = time.time()

            # 5. 如果开启了 Profiling，则写入日志
            if self.enable_profiling:
                try:
                    if batch_idx < len(self.profile_data):
                        record = self.profile_data[batch_idx]
                        getattr(record, attr_name)[0] = start_time
                        getattr(record, attr_name)[1] = end_time
                except Exception:
                    pass
