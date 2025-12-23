from abc import abstractmethod
from functools import partial, wraps
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
from dataclasses import dataclass, field
import torch.utils.checkpoint
from usfl.socket import SocketCommunicator
from usfl.utils.exp import fed_avg_params, fed_average
from usfl.utils.timestamp_recorder import GanttChartData
from contextlib import contextmanager
from typing import Union, List


@dataclass(order=True)
class PrioritizedItem:
    priority: tuple = field(compare=True)  # (client_progress, batch_idx)
    data: Any = field(compare=False)


def see_mem(func=None, *, version='v3', device='cuda:0'):
    # 场景 1: @see_mem(version='v2') -> func 是 None
    if func is None:
        # 返回一个固定了 version 的新函数，等待接收 func
        return partial(see_mem, version=version)

    # 场景 2: @see_mem -> func 是被装饰的函数
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize(device)
        mem_before = torch.cuda.memory_allocated(device) / 1024**3
        print(f"[{version}] Memory before {func.__name__}: {mem_before:.3f} GB")

        try:
            return func(*args, **kwargs)
        finally:
            torch.cuda.synchronize(device)
            mem_after = torch.cuda.memory_allocated(device) / 1024**3
            print(f"[{version}] Memory after {func.__name__}: {mem_after:.3f} GB")

    return wrapper


class ServerBase:
    def __init__(
        self,
        server_args: Dict[str, Any],
        server_device: torch.device,
        num_clients: int,
        logger: logging.Logger,
        matrix_logger: logging.Logger = None,  # 下沉到 Base
        enable_profiling: bool = True,
    ):
        self.server_args = server_args
        self.server_device = server_device
        self.num_clients = num_clients
        self.logger = logger
        self.matrix_logger = matrix_logger

        # --- 1. 通信与网络 ---
        torch.cuda.set_device(self.server_device)
        self.communicator = SocketCommunicator(is_server=True, port=server_args["port"], buffer_size=65536)
        self.clients: List[Tuple[socket.socket, tuple, int]] = []
        self.client_lock = threading.Lock()

        # --- 2. 线程管理 ---
        self.compute_task_thread: threading.Thread = None
        self.clients_comm_threads: List[threading.Thread] = []

        # --- 3. 队列与调度 (通用) ---
        # 解析队列顺序配置 (FIFO/LIFO)
        queue_order_raw = server_args.get("queue_order", "lifo")
        self.queue_order = str(queue_order_raw).lower()
        if self.logger:
            self.logger.info(f"[Server] Using {self.queue_order.upper()} queues.")

        # 统一使用 PriorityQueue (移除原代码中重复的 Queue 初始化)
        self.activation_queue = queue.PriorityQueue()
        self.gradient_queue = queue.PriorityQueue()
        # 发送给客户端的队列通常不需要优先级，保持普通 Queue 即可，视需求而定
        self.server_activation_queues: Dict[int, queue.Queue] = {cid: queue.Queue() for cid in range(self.num_clients)}

        # --- 4. 状态与进度追踪 ---
        self.aggregate_count = 0
        self.client_head_model_params: Dict[int, List[nn.Parameter]] = {}
        self.client_tail_model_params: Dict[int, List[nn.Parameter]] = {}
        self.client_model_losses: Dict[int, float] = {}

        # 进度追踪 (V1/V2/V3 通用)
        self.client_fwd_progress = [0] * num_clients
        self.client_bwd_progress = [0] * num_clients

        # --- 5. Profile / 统计信息 ---
        self.compute_time = 0
        self.aggregate_server_time = 0
        self.aggregate_client_time = 0
        self.idle_time = 0
        self.stop_count = 0

        # 初始化 Profile 数据容器 (具体填充由子类或辅助函数完成)
        self.enable_profiling = enable_profiling
        self.profile_datas: Dict[int, Any] = {}
        if self.enable_profiling:
            self._init_profile_data()
        else:
            print("Profiling disabled. ")

    def _init_profile_data(self):
        """辅助方法：统一初始化 Profile 数据，避免子类重复代码"""
        # 假设记录前 10 个 batch
        for client_id in range(self.num_clients):
            self.profile_datas[client_id] = [GanttChartData(batch_idx=i, client_id=client_id) for i in range(101)]

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

    def _put_to_queue(self, q, data: Any, phase: str = "forward"):
        c_id = data.get("client_id", 0)

        if phase == "forward":
            start_time = data.get("head_fwd_start_time", time.time())
            current_progress = self.client_fwd_progress[c_id]
        elif phase == "backward":
            start_time = data.get("tail_start_time", time.time())
            current_progress = self.client_bwd_progress[c_id]
        else:
            raise ValueError(f"Unknown phase: '{phase}'. Expected 'forward' or 'backward'.")

        if self.queue_order == "fifo":
            priority = (start_time, 0)
        elif self.queue_order == "lifo":
            priority = (-start_time, 0)
        elif self.queue_order == "straggler_fo":
            duration = time.time() - start_time
            priority = (current_progress, -duration)
        else:
            priority = (start_time, 0)

        q.put(PrioritizedItem(priority=priority, data=data))

    def _get_from_queue(self, q, timeout=None):
        if timeout is None:
            return q.get_nowait().data
        else:
            return q.get(timeout=timeout).data

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

    @contextmanager
    def _profile_scope(self, client_id_or_ids: Union[int, List[int]], batch_idx: int, attr_name: str):
        """
        用于计时的上下文管理器。
        自动处理：开关判断、CUDA同步、时间戳记录。
        """
        # 1. 如果没开 profiling，直接 yield 继续执行业务代码，不做任何操作
        if not self.enable_profiling:
            yield
            return

        # 2. 计时开始前操作
        if torch.device(self.server_device).type == "cuda":
            torch.cuda.synchronize(self.server_device)
        start_time = time.time()

        # 3. 执行业务代码
        yield

        # 4. 计时结束后操作
        if torch.device(self.server_device).type == "cuda":
            torch.cuda.synchronize(self.server_device)
        end_time = time.time()

        # 5. 写入数据
        try:
            # 假设你的 GanttChartData 对象有对应的属性 (如 server_fwd_timestamp)
            # 且该属性是一个列表或数组 [start, end]
            target_clients = client_id_or_ids if isinstance(client_id_or_ids, list) else [client_id_or_ids]

            for cid in target_clients:
                if cid in self.profile_datas and batch_idx < len(self.profile_datas[cid]):
                    record = self.profile_datas[cid][batch_idx]
                    timestamp_list = getattr(record, attr_name)
                    timestamp_list[0] = start_time
                    timestamp_list[1] = end_time
        except KeyError:
            # 防止 client_id 或 batch_idx 超出范围导致崩溃
            pass
        except AttributeError:
            self.logger.warning(f"Profile data missing attribute: {attr_name}")
