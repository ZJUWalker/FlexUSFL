from abc import abstractmethod
import socket
import threading
import queue
import logging
from queue import Queue
import time
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import copy

import torch.utils.checkpoint
from usfl.socket import SocketCommunicator
from usfl.utils.exp import fed_avg_params, fed_average


class ServerBase:

    def __init__(
        self,
        server_args: Dict[str, Any],
        server_device: torch.device,
        num_clients: int,
        logger: logging.Logger,
    ):
        super().__init__()
        self.server_args = server_args
        self.server_device = server_device
        self.num_clients = num_clients
        self.logger: logging.Logger = logger
        self.communicator = SocketCommunicator(is_server=True, port=server_args["port"], buffer_size=server_args["buffer_size"])
        self.aggregate_server_lock = threading.Lock()  # 用于同步聚合操作
        self.aggregate_server_count = 0  # 跟踪聚合信号的计数器
        self.aggregate_server_event = threading.Event()  # 用于通知聚合完成
        # --------------------------------------------------------------------------
        self.aggregate_client_lock = threading.Lock()  # 用于同步聚合操作
        self.aggregate_client_count = 0  # 跟踪聚合信号的计数器
        self.aggregate_client_event = threading.Event()  # 用于通知聚合完成
        self.client_head_model_params: Dict[int, List[nn.Parameter]] = {}  # 用于保存客户端模型的字典
        self.client_tail_model_params: Dict[int, List[nn.Parameter]] = {}  # 用于保存客户端模型的字典
        self.client_model_losses: Dict[int, float] = {}  # 用于保存客户端模型的损失
        # --------------------------------------------------------------------------
        self.activation_queue: Queue = Queue()  # 用于接收客户端的激活信号
        self.server_activation_queues: Dict[int, Queue] = {cid: queue.Queue() for cid in range(self.num_clients)}  # 用于发送给客户端的激活信号
        self.gradient_queue: Queue = Queue()  # 用于接收客户端的梯度信号
        self.clients: List[Tuple[socket.socket, tuple, int]] = []  # 客户端连接信息
        self.client_lock = threading.Lock()  # 用于同步客户端连接信息
        self.compute_task_thread: threading.Thread = None  # 用于计算任务的线程
        self.clients_comm_threads: List[threading.Thread] = []  # 用于处理客户端通信的线程
        pass

    @abstractmethod
    def _forward(
        self, client_id: int, activation: torch.Tensor, attention_mask: torch.Tensor, position_embeddings: Tuple[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError('Please implement "forward" method')

    @abstractmethod
    def _backward(self, client_id: int, server_grad: torch.Tensor):
        raise NotImplementedError('Please implement "backward" method')

    @abstractmethod
    def handle_client_with_aggregation(self, conn: socket.socket, addr: Tuple, *args, **kwargs) -> None:
        raise NotImplementedError('Please implement "handle_client_with_aggregation" method')

    @abstractmethod
    def compute_task(self, *args, **kwargs) -> None:
        raise NotImplementedError('Please implement "compute_task" method')

    # main function to start the server
    def run(self):
        # init compute task thread
        self.compute_task_thread = threading.Thread(target=self.compute_task, args=(), daemon=True)
        self.compute_task_thread.start()
        # init client communication threads
        self.logger.info("Waiting for clients to connect...")
        while True:
            conn, addr = self.communicator.accept_client()
            if conn:
                with self.client_lock:
                    self.clients.append((conn, addr, None))
                self.logger.info(f"Connected from {addr}, current client count: {len(self.clients)}")
                thread = threading.Thread(
                    target=self.handle_client_with_aggregation,
                    args=(conn, addr),
                )
                self.clients_comm_threads.append(thread)
                thread.start()
                if len(self.clients_comm_threads) == self.num_clients:
                    break
        self.logger.info(f"All {self.num_clients} clients connected")
        pass

    def _reset_server_aggregate_server_count(self) -> None:
        with self.aggregate_server_lock:
            self.aggregate_server_count = 0
            self.aggregate_server_event.clear()

    def _handle_aggregate_client_models(self, data: Dict) -> Dict[str, Any]:
        client_id = data['client_id']
        with self.client_lock:
            self.client_head_model_params[client_id] = data['head_params']
            self.client_tail_model_params[client_id] = data['tail_params']
            self.client_model_losses[client_id] = data['loss']
            self.aggregate_client_count += 1
            self.logger.info(f"Client aggregate count: {self.aggregate_client_count}/{self.num_clients}")
            if self.aggregate_client_count == self.num_clients:
                # aggregate the client models
                head_params_avg, tail_params_avg, loss = self._aggregate_client_models()
                # Send acknowledgment to all clients
                for cid in range(self.num_clients):
                    self._send_to_client(
                        cid, {"status": "aggregate_client_complete", "head_params": head_params_avg, "tail_params": tail_params_avg, "loss": loss}
                    )
            # else:
            #     self._send_to_client(client_id, {"status": "waiting_for_others"})

    def _aggregate_client_models(self) -> Dict:
        with self.aggregate_client_lock:
            head_params_avg = fed_avg_params(list(self.client_head_model_params.values()))
            tail_params_avg = fed_avg_params(list(self.client_tail_model_params.values()))
            avg_loss = sum(self.client_model_losses.values()) / len(self.client_model_losses)
            self.aggregate_client_count = 0
            self.aggregate_client_event.clear()
            self.client_head_model_params.clear()
            self.client_tail_model_params.clear()
            self.logger.info(f"Aggregated client models finished, avg loss: {avg_loss}")
            return head_params_avg, tail_params_avg, avg_loss

    def _send_to_client(self, client_id: int, response: Dict) -> bool:
        for conn, addr, cid in self.clients:
            if cid == client_id:
                try:
                    self.communicator.send(response, conn)
                    self.logger.info(f"Sent message to client_id={client_id} (addr={addr})")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to send message to client_id={client_id} (addr={addr}): {e}")
                    return False
        self.logger.warning(f"No connection found for client_id={client_id}")
        return False
