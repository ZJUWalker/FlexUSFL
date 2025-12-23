import queue
import logging
import time
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass, field, asdict
import json
import os
import torch.utils.checkpoint
from usfl.server.server_base import see_mem
from usfl.server.server_v2 import ServerV2
from usfl.utils.timestamp_recorder import GanttChartData


class ServerV3(ServerV2):

    def __init__(
        self,
        server_args: Dict[str, Any],
        server_model: nn.Module,
        server_device: torch.device,
        lr: float,
        num_clients: int,
        checkpoint_mode: str = 'no',
        optimizer_clz: torch.optim.Optimizer.__class__ = torch.optim.AdamW,
        logger: logging.Logger = None,
        matrix_logger: logging.Logger = None,
        enforce_sync: bool = True,
    ):

        super().__init__(
            server_args=server_args,
            server_device=server_device,
            server_model=server_model,
            lr=lr,
            num_clients=num_clients,
            optimizer_clz=optimizer_clz,
            logger=logger,
            matrix_logger=matrix_logger,
        )
        self.checkpoint_mode = checkpoint_mode
        # self.checkpoint_clients=[]
        self.server_output_dict: Dict[int, torch.Tensor] = {}  # Dict[int, torch.Tensor]
        self.hidden_status_from_head_dict: Dict[int, torch.Tensor] = {}
        self.activation_per_batch = 0
        self.do_ckpt_client_num = 0
        self.device_mem_capacity = torch.cuda.get_device_properties(self.server_device).total_memory  # bytes

        for client_id in range(self.num_clients):
            self.server_output_dict[client_id] = None
            self.hidden_status_from_head_dict[client_id] = None

        self.enforce_sync = enforce_sync
        if self.enforce_sync == False:
            self.num_accumulated_grads = 0
            self.grad_accum_threshold = self.num_clients
        self.num_updates = 0

    def _do_ckpt(self):
        if self.checkpoint_mode == 'no':
            return False
        elif self.checkpoint_mode == 'all':
            return True
        elif self.checkpoint_mode == 'selective':
            curr_mem_reserved = torch.cuda.memory_reserved()
            return curr_mem_reserved + self.activation_per_batch >= self.device_mem_capacity
        return False

    @see_mem()
    def _forward(self, client_id: int, activation: torch.Tensor, attention_mask: torch.LongTensor, position_embeddings: Tuple[torch.Tensor] = None):
        try:
            hidden_status_from_head = activation.to(self.server_device)
            hidden_status_from_head.requires_grad_(True)
            hidden_status_from_head.retain_grad()
            attention_mask = attention_mask.to(self.server_device) if attention_mask is not None else None
            pos_emb = tuple(torch.tensor(pos).to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
            if self._do_ckpt():
                print(
                    f'do ckpt for client {client_id},curr mem reserved {torch.cuda.memory_reserved() / 1024**3} GB, activation_per_batch {self.activation_per_batch / 1024**3} GB'
                )
                server_output = torch.utils.checkpoint.checkpoint(
                    self.trunk_model.__call__,
                    hidden_status_from_head,
                    attention_mask,
                    pos_emb,
                    use_reentrant=False,
                )
                self.do_ckpt_client_num += 1
            else:
                if self.activation_per_batch == 0:
                    torch.cuda.synchronize(self.server_device)
                    curr_cuda_mem_allocated = torch.cuda.memory_allocated()
                    server_output: torch.Tensor = self.trunk_model(hidden_status_from_head, attention_mask, pos_emb)
                    torch.cuda.synchronize(self.server_device)
                    self.activation_per_batch = torch.cuda.memory_allocated() - curr_cuda_mem_allocated
                    print('activation_per_batch', self.activation_per_batch / 1024**3, 'GB')
                else:
                    server_output: torch.Tensor = self.trunk_model(hidden_status_from_head, attention_mask, pos_emb)
            server_output = server_output.requires_grad_(True)
            self.server_output_dict[client_id] = server_output
            self.hidden_status_from_head_dict[client_id] = hidden_status_from_head
            activation_to_tail = server_output.cpu()
            torch.cuda.empty_cache()
            # if self.queue_order == "straggler_fo":
            self.client_fwd_progress[client_id] += 1
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Client {client_id}: Forward pass failed: {e}")
            raise e

    @see_mem()
    def _backward(self, client_id: int, server_grad: torch.Tensor, batch_idx: int):
        try:
            with self._profile_scope(client_id, batch_idx, "server_bwd_timestamp"):
                server_grad = server_grad.to(self.server_device)
                self.server_output_dict[client_id].backward(server_grad)

            grad_to_head = self.hidden_status_from_head_dict[client_id].grad.cpu()
            torch.cuda.empty_cache()

            # if self.queue_order == "straggler_fo":
            self.client_bwd_progress[client_id] += 1
            return grad_to_head
        except Exception as e:
            self.logger.error(f"Client {client_id}: Backward pass failed: {e}")
            raise e

    def compute_task(self):
        """根据配置选择同步或异步模式"""
        if self.enforce_sync:
            self._compute_task_sync()
        else:
            self._compute_task_async()

    def _compute_task_sync(self):
        unforwarded_clients = set(range(self.num_clients))
        unbackwarded_clients = set()
        temp_buffer = []

        while True:
            try:
                data = self._get_from_queue(self.activation_queue)

                if data["client_id"] not in unforwarded_clients:
                    temp_buffer.append(data)
                else:
                    with self._profile_scope(data["client_id"], data["batch_idx"], "server_fwd_timestamp"):
                        server_activation = self._forward(
                            data["client_id"],
                            data["activation"],
                            data["attention_mask"] if "attention_mask" in data else None,
                            data["position_embeddings"] if "position_embeddings" in data else None,
                        )
                        self._put_to_queue(
                            self.server_activation_queues[data["client_id"]],
                            {"client_id": data["client_id"], "server_activation": server_activation, "batch_idx": data["batch_idx"]},
                            phase="forward",
                        )

                    unforwarded_clients.remove(data["client_id"])
                    unbackwarded_clients.add(data["client_id"])

            except queue.Empty:
                pass
            try:
                data = self._get_from_queue(self.gradient_queue)
                d_activation_to_client = self._backward(data["client_id"], data["gradient"], data["batch_idx"])
                self._put_to_queue(
                    self.server_activation_queues[data["client_id"]],
                    {"client_id": data["client_id"], "server_gradient": d_activation_to_client, "batch_idx": data["batch_idx"]},
                    phase="backward",
                )
                unbackwarded_clients.remove(data["client_id"])
                if len(unbackwarded_clients) == 0 and len(unforwarded_clients) == 0:

                    all_clients = list(range(self.num_clients))
                    with self._profile_scope(all_clients, self.num_updates, "server_step_timestamp"):
                        self._update_server_model()

                    unforwarded_clients = list(range(self.num_clients))
                    for buffered_data in temp_buffer:
                        self._put_to_queue(self.activation_queue, buffered_data)

                    temp_buffer.clear()
                    self.num_updates += 1
                    self.do_ckpt_client_num = 0

            except queue.Empty:
                pass
            time.sleep(0.001)

    def _compute_task_async(self):
        while True:
            try:
                data = self._get_from_queue(self.activation_queue)

                with self._profile_scope(data["client_id"], data["batch_idx"], "server_fwd_timestamp"):
                    server_activation = self._forward(
                        data["client_id"],
                        data["activation"],
                        data["attention_mask"] if "attention_mask" in data else None,
                        data["position_embeddings"] if "position_embeddings" in data else None,
                    )
                    self._put_to_queue(
                        self.server_activation_queues[data["client_id"]],
                        {"client_id": data["client_id"], "server_activation": server_activation, "batch_idx": data["batch_idx"]},
                        phase="forward",
                    )

            except queue.Empty:
                pass
            try:
                data = self._get_from_queue(self.gradient_queue)
                d_activation_to_client = self._backward(data["client_id"], data["gradient"], data["batch_idx"])
                self._put_to_queue(
                    self.server_activation_queues[data["client_id"]],
                    {"client_id": data["client_id"], "server_gradient": d_activation_to_client, "batch_idx": data["batch_idx"]},
                    phase="backward",
                )
                self.num_accumulated_grads += 1

                if self.num_accumulated_grads == self.grad_accum_threshold:

                    all_clients = list(range(self.num_clients))
                    with self._profile_scope(all_clients, self.num_updates, "server_step_timestamp"):
                        self._update_server_model()

                    self.num_accumulated_grads = 0
                    self.num_updates += 1
                    self.do_ckpt_client_num = 0

            except queue.Empty:
                pass
            time.sleep(0.001)

    @torch.no_grad()
    def _update_server_model(self):
        # with self.client_lock:
        if self.server_args["use_avg"]:
            max_norm = 0.5 * self.num_clients
            grads = [p.grad for p in self.trunk_model.parameters() if p.grad is not None]
            for g in grads:
                g.data.div_(self.num_clients)
        else:
            max_norm = 0.5
        torch.nn.utils.clip_grad_norm_(self.trunk_model.parameters(), max_norm=max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize(self.server_device)

        self.logger.info("Server model updated")
