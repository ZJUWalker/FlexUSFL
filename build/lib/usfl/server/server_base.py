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
        self.aggregate_count = 0  # 跟踪聚合信号的计数器
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
        # ----------------------------------profile----------------------------------------
        self.compute_time = 0  # 用于记录计算时间
        self.aggregate_server_time = 0  # 用于记录聚合时间
        self.aggregate_client_time = 0  # 用于记录聚合时间
        self.idle_time = 0  # 用于记录空闲时间
        self.stop_count = 0  # 用于记录停止信号的数量
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
                # with self.client_lock:
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

    def _reset_aggregate_count(self) -> None:
        self.aggregate_count = 0

    def _aggregate_client_models(self) -> Dict:
        agg_start_time = time.time()
        head_params_avg = fed_avg_params(list(self.client_head_model_params.values()))
        tail_params_avg = fed_avg_params(list(self.client_tail_model_params.values()))
        avg_loss = sum(self.client_model_losses.values()) / len(self.client_model_losses)
        self.client_head_model_params.clear()
        self.client_tail_model_params.clear()
        agg_end_time = time.time()
        self.aggregate_client_time += agg_end_time - agg_start_time
        return head_params_avg, tail_params_avg, avg_loss

    def _send_to_client(self, client_id: int, response: Dict) -> bool:
        for conn, addr, cid in self.clients:
            if cid == client_id:
                try:
                    self.communicator.send(response, conn)
                    # print(f"Sent message to client_id={client_id} (addr={addr})")
                    # self.logger.info(f"Sent message to client_id={client_id} (addr={addr})")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to send message to client_id={client_id} (addr={addr}): {e}")
                    print(f"Failed to send message to client_id={client_id} (addr={addr}): {e}")
                    return False
        self.logger.warning(f"No connection found for client_id={client_id}")
        return False
