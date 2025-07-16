import socket
import threading
import queue
import logging
import time
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import copy

from usfl.server.server_base import ServerBase
from usfl.utils.exp import fed_avg_params


class ServerV1(ServerBase):

    def __init__(
        self,
        server_args: Dict[str, Any],
        server_model: nn.Module,
        server_device: torch.device,
        lr: float,
        num_clients: int,
        optimizer_clz: torch.optim.Optimizer.__class__ = torch.optim.AdamW,
        logger: logging.Logger = None,
        matrix_logger: logging.Logger = None,
    ):
        super().__init__(server_args=server_args, server_device=server_device, num_clients=num_clients, logger=logger)
        self.lr = lr
        self.matrix_logger = matrix_logger
        # special variables for V1 ------------------------------------
        self.trunk_models: Dict[int, nn.Module] = {}
        self.optimizers: Dict[int, torch.optim.Optimizer] = {}  # 用于保存服务端模型的优化器 V1
        # -------------------------------------------------------------
        self.locks: Dict[int, threading.Lock] = {client_id: threading.Lock() for client_id in range(num_clients)}
        self.server_hidden_states_output: Dict[int, torch.FloatTensor] = {}  # 用于保存客户端的输出激活量
        self.hidden_states_from_head: Dict[int, torch.FloatTensor] = {}  # 用于保存服务端的输入激活量
        for client_id in range(num_clients):
            self.trunk_models[client_id] = copy.deepcopy(server_model)
            self.trunk_models[client_id].to(self.server_device).train()
            self.optimizers[client_id] = optimizer_clz(self.trunk_models[client_id].parameters(), lr=self.lr)
            self.server_hidden_states_output[client_id] = None
            self.hidden_states_from_head[client_id] = None

    def handle_client_with_aggregation(self, client_conn: socket.socket, addr: Tuple) -> None:
        client_id: int = None
        try:
            client_conn.settimeout(10.0)
            while True:
                data: Dict = self.communicator.receive(client_conn)
                if data is None:
                    self.logger.info(f"Client {addr} disconnected")
                    break
                if "client_id" in data and client_id is None:
                    client_id = data["client_id"]
                    with self.client_lock:
                        for i, (c, a, cid) in enumerate(self.clients):
                            if c == client_conn and cid is None:
                                self.clients[i] = (c, a, client_id)
                                self.logger.info(f"Assigned client_id={client_id} to addr={addr}")
                                break
                if "activation" in data:
                    self.logger.info(f"Received activation from client_id={client_id} (addr={addr}): shape={data['activation'].shape}")
                    self.activation_queue.put(data)
                    response = self.server_activation_queues[client_id].get(timeout=10.0)
                    if response["client_id"] == client_id:
                        self._send_to_client(client_id, response)
                elif "gradient" in data:
                    self.logger.info(f"Received gradient from client_id={client_id} (addr={addr}): shape={data['gradient'].shape}")
                    self.gradient_queue.put(data)
                    response = self.server_activation_queues[client_id].get(timeout=10.0)
                    if response["client_id"] == client_id and "server_gradient" in response:
                        self._send_to_client(client_id, response)
                elif "aggregate" in data:
                    self.logger.info(f"Received aggregation request from client_id={client_id} (addr={addr})")
                    with self.client_lock:
                        self.aggregate_server_count += 1
                        self.logger.info(f"Server aggregate count: {self.aggregate_server_count}/{self.num_clients}")
                        if self.aggregate_server_count == self.num_clients:
                            response = self._trunk_models_aggregated()
                            self._reset_server_aggregate_server_count()
                            # log memory usage
                            self.matrix_logger.info(
                                f"{data['step']:^5}|"
                                f"{torch.cuda.max_memory_allocated(self.server_device)/1024**3:^15.3f}|"
                                f"{torch.cuda.max_memory_reserved(self.server_device)/1024**3:^18.3f}"
                            )
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats(self.server_device)
                            # Send acknowledgment to all clients
                            for cid in range(self.server_args["num_clients"]):
                                self._send_to_client(cid, {"status": "aggregate_complete"})
                        else:
                            self._send_to_client(client_id, {"status": "waiting_for_others"})
                elif 'aggregate_client' in data:
                    self.logger.info(f"Received client aggregation request from client_id={client_id} (addr={addr})")
                    self._handle_aggregate_client_models(data)

        except Exception as e:
            self.logger.error(f"Client {addr} (client_id={client_id}) error: {e}")
        finally:
            with self.client_lock:
                self.clients[:] = [(c, a, cid) for c, a, cid in self.clients if c != client_conn]
            client_conn.close()
            self.logger.info(f"Client {addr} (client_id={client_id}) closed")

    def compute_task(self) -> None:
        while True:
            try:
                data = self.activation_queue.get_nowait()
                self.logger.info(f"Processing activation for client_id={data['client_id']}, queue size={self.activation_queue.qsize()}")
                server_activation = self._forward(
                    data["client_id"],
                    data["activation"],
                    data["attention_mask"],
                    data["position_embeddings"],
                )
                self.server_activation_queues[data["client_id"]].put({"client_id": data["client_id"], "server_activation": server_activation})
                self.logger.info(f"Completed forward pass for client_id={data['client_id']}")
            except queue.Empty:
                pass
            try:
                data = self.gradient_queue.get_nowait()
                self.logger.info(f"Processing gradient for client_id={data['client_id']}, queue size={self.gradient_queue.qsize()}")
                d_activation_to_client = self._backward(data["client_id"], data["gradient"])
                self.server_activation_queues[data["client_id"]].put({"client_id": data["client_id"], "server_gradient": d_activation_to_client})
                self.logger.info(f"Completed backward pass for client_id={data['client_id']}")
            except queue.Empty:
                pass
            time.sleep(0.01)

    def _forward(
        self, client_id: int, activation: torch.Tensor, attention_mask: torch.LongTensor, position_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        try:
            with self.locks[client_id]:
                hidden_states_from_head = activation.to(self.server_device)
                hidden_states_from_head.requires_grad_(True)
                hidden_states_from_head.retain_grad()
                attention_mask = attention_mask.to(self.server_device)
                pos_emb = tuple(torch.tensor(pos).to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
                server_output: torch.Tensor = self.trunk_models[client_id](
                    hidden_states=hidden_states_from_head,
                    attention_mask=attention_mask,
                    position_embeddings=pos_emb,
                )
                server_output = server_output.requires_grad_(True)
                self.server_hidden_states_output[client_id] = server_output
                self.hidden_states_from_head[client_id] = hidden_states_from_head
                activation_to_tail = server_output.cpu()
                # self.logger.info(f"Client {client_id}: Releasing lock for forward")
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Client {client_id}: Forward pass failed: {e}")
            raise

    def _backward(self, client_id: int, server_grad: torch.FloatTensor) -> torch.FloatTensor:
        try:
            with self.locks[client_id]:
                self.optimizers[client_id].zero_grad()
                server_grad = server_grad.to(self.server_device)
                self.server_hidden_states_output[client_id].backward(server_grad)
                torch.nn.utils.clip_grad_norm_(self.trunk_models[client_id].parameters(), max_norm=0.5)
                self.optimizers[client_id].step()
                grad_to_head = self.hidden_states_from_head[client_id].grad.cpu()
                # torch.cuda.empty_cache()
            return grad_to_head
        except Exception as e:
            self.logger.error(f"Client {client_id}: Backward pass failed: {e}")
            raise

    def _trunk_models_aggregated(self) -> dict:
        try:
            self.logger.info("Acquiring aggregate lock")
            with self.aggregate_server_lock:
                self.logger.info("Aggregate lock acquired")
                w_trunk_model_params = []
                for client_id in range(self.num_clients):
                    w_trunk_model_params.append(list(filter(lambda p: p.requires_grad, self.trunk_models[client_id].parameters())))
                avg_trunk_model_params = fed_avg_params(w_trunk_model_params)
                for client_id in range(self.num_clients):
                    i = 0
                    for p in self.trunk_models[client_id].parameters():
                        if p.requires_grad:
                            p.data.copy_(avg_trunk_model_params[i], non_blocking=True)
                            i += 1
                torch.cuda.synchronize(device=self.server_device)
                torch.cuda.empty_cache()
                # w_trunk_models = []
                # for client_id in range(self.num_clients):
                #     self.trunk_models[client_id].cpu()
                #     w_trunk_models.append(self.trunk_models[client_id].state_dict())
                # w_glob_trunk_model = fed_average(w_trunk_models)
                # for client_id in range(self.num_clients):
                #     self.trunk_models[client_id].load_state_dict(w_glob_trunk_model)
                #     self.trunk_models[client_id].to(self.server_device).train()
                self.logger.info("Completed trunk models aggregation")
            self.aggregate_server_event.set()  # 通知聚合完成
            return {"status": "aggregation_completed"}
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            raise
