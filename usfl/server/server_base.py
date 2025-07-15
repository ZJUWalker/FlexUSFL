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
from usfl.utils.exp import fed_average

__all__ = ["ServerV1", "ServerV2", "ServerV3"]


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
        self.aggregate_lock = threading.Lock()  # 用于同步聚合操作
        self.aggregate_count = 0  # 跟踪聚合信号的计数器
        self.aggregate_event = threading.Event()  # 用于通知聚合完成
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

    def _reset_aggregate_count(self) -> None:
        with self.aggregate_lock:
            self.aggregate_count = 0
            self.aggregate_event.clear()

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
                        self.aggregate_count += 1
                        self.logger.info(f"Aggregate count: {self.aggregate_count}/{self.num_clients}")
                        if self.aggregate_count == self.num_clients:
                            response = self._trunk_models_aggregated()
                            self._reset_aggregate_count()
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
            with self.aggregate_lock:
                self.logger.info("Aggregate lock acquired")
                w_trunk_models = []
                for client_id in range(self.num_clients):
                    self.trunk_models[client_id].cpu()
                    w_trunk_models.append(self.trunk_models[client_id].state_dict())
                w_glob_trunk_model = fed_average(w_trunk_models)
                for client_id in range(self.num_clients):
                    self.trunk_models[client_id].load_state_dict(w_glob_trunk_model)
                    self.trunk_models[client_id].to(self.server_device).train()
                self.logger.info("Completed trunk models aggregation")
            self.aggregate_event.set()  # 通知聚合完成
            return {"status": "aggregation_completed"}
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            raise


class ServerV2(ServerBase):

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
        self.trunk_model = server_model.to(self.server_device).train()
        self.optimizer: torch.optim.Optimizer = optimizer_clz(self.trunk_model.parameters(), lr=self.lr)
        self.matrix_logger = matrix_logger
        self.server_output: torch.Tensor = None
        self.hidden_status_from_head: torch.Tensor = None

    def _forward(
        self, client_id: int, activation: torch.Tensor, attention_mask: torch.LongTensor, position_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        try:
            # self.logger.info(f"Client {client_id}: Acquiring lock for forward")
            with self.client_lock:
                # self.logger.info(f"Client {client_id}: Lock acquired for forward")
                hidden_status_from_head = activation.to(self.server_device)
                hidden_status_from_head.requires_grad_(True)
                hidden_status_from_head.retain_grad()
                attention_mask = attention_mask.to(self.server_device)
                pos_emb = tuple(torch.tensor(pos).to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
                server_output: torch.Tensor = self.trunk_model(
                    hidden_states=hidden_status_from_head,
                    attention_mask=attention_mask,
                    position_embeddings=pos_emb,
                )
                server_output = server_output.requires_grad_(True)
                self.server_output = server_output
                self.hidden_status_from_head = hidden_status_from_head
                activation_to_tail = server_output.cpu()
                # torch.cuda.empty_cache()
                # self.logger.info(f"Client {client_id}: Releasing lock for forward")
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Client {client_id}: Forward pass failed: {e}")
            raise e

    def _backward(self, client_id: int, server_grad: torch.Tensor) -> torch.Tensor:
        try:
            # self.logger.info(f"Client {client_id}: Acquiring lock for backward")
            with self.client_lock:
                # self.logger.info(f"Client {client_id}: Lock acquired for backward")
                self.optimizer.zero_grad()
                server_grad = server_grad.to(self.server_device)
                self.server_output.backward(server_grad)
                torch.nn.utils.clip_grad_norm_(self.trunk_model.parameters(), max_norm=0.5)
                self.optimizer.step()
                grad_to_head = self.hidden_status_from_head.grad.cpu()
                # torch.cuda.empty_cache()
                # self.logger.info(f"Client {client_id}: Releasing lock for backward")
            return grad_to_head
        except Exception as e:
            self.logger.error(f"Client {client_id}: Backward pass failed: {e}")
            raise e

    def handle_client_with_aggregation(self, conn: socket.socket, addr: Tuple, *args, **kwargs):
        # return super().handle_client_with_aggregation(conn, addr, *args, **kwargs)
        client_id = None
        try:
            conn.settimeout(60.0)
            while True:
                data = self.communicator.receive(conn)
                if data is None:
                    self.logger.info(f"Client {addr} disconnected")
                    break
                if "client_id" in data and client_id is None:
                    client_id = data["client_id"]
                    with self.client_lock:
                        for i, (c, a, cid) in enumerate(self.clients):
                            if c == conn and cid is None:
                                self.clients[i] = (c, a, client_id)
                                self.logger.info(f"Assigned client_id={client_id} to addr={addr}")
                                break
                if "activation" in data:
                    self.logger.info(f"Received activation from client_id={client_id} (addr={addr}): shape={data['activation'].shape}")
                    self.activation_queue.put(data)
                    response = self.server_activation_queues[client_id].get(timeout=180.0)
                    if response["client_id"] == client_id:
                        self._send_to_client(client_id, response)
                elif "gradient" in data:
                    self.logger.info(f"Received gradient from client_id={client_id} (addr={addr}): shape={data['gradient'].shape}")
                    self.gradient_queue.put(data)
                    response = self.server_activation_queues[client_id].get(timeout=180.0)
                    if response["client_id"] == client_id and "server_gradient" in response:
                        self._send_to_client(client_id, response)
                elif "aggregate" in data:
                    self.logger.info(f"Received aggregation request from client_id={client_id} (addr={addr})")
                    with self.aggregate_lock:
                        self.aggregate_count += 1
                        self.logger.info(f"Aggregate count: {self.aggregate_count}/{self.server_args['num_clients']}")
                        if self.aggregate_count == self.server_args["num_clients"]:
                            self._reset_aggregate_count()
                            # log memory usage
                            self.matrix_logger.info(
                                f"{data['step']:^5}|"
                                f"{torch.cuda.max_memory_allocated(self.server_device)/1024**3:^15.3f}|"
                                f"{torch.cuda.max_memory_reserved(self.server_device)/1024**3:^18.3f}"
                            )
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats(self.server_device)
                            # Send acknowledgment to all clients #TODO maybe not necessary
                            for cid in range(self.server_args["num_clients"]):
                                self._send_to_client(cid, {"status": "aggregate_complete"})
                        else:
                            self._send_to_client(client_id, {"status": "waiting_for_others"})
        except Exception as e:
            self.logger.error(f"Client {addr} (client_id={client_id}) error: {e}")
        finally:
            with self.client_lock:
                self.clients[:] = [(c, a, cid) for c, a, cid in self.clients if c != conn]
            conn.close()
            self.logger.info(f"Client {addr} (client_id={client_id}) closed")

    def compute_task(self):
        uncompleted_clients = list(range(self.num_clients))
        while True:
            try:
                data = self.activation_queue.get(timeout=30.0)
                if data["client_id"] not in uncompleted_clients:
                    self.activation_queue.put(data, timeout=30.0)
                    time.sleep(0.001)
                    continue
                self.logger.info(f"Processing activation for client_id={data['client_id']}, queue size={self.activation_queue.qsize()}")
                server_activation = self._forward(
                    data["client_id"],
                    data["activation"],
                    data["attention_mask"],
                    data["position_embeddings"],
                )
                self.server_activation_queues[data["client_id"]].put({"client_id": data["client_id"], "server_activation": server_activation})
                self.logger.info(f"Completed forward pass for client_id={data['client_id']}")

                if data["is_training"] == False:
                    continue

                while True:
                    try:
                        data = self.gradient_queue.get_nowait()
                        self.logger.info(f"Processing gradient for client_id={data['client_id']}, queue size={self.gradient_queue.qsize()}")
                        d_activation_to_client = self._backward(data["client_id"], data["gradient"])
                        self.server_activation_queues[data["client_id"]].put(
                            {"client_id": data["client_id"], "server_gradient": d_activation_to_client}
                        )
                        self.logger.info(f"Completed backward pass for client_id={data['client_id']}")

                        uncompleted_clients.remove(data["client_id"])
                        if len(uncompleted_clients) == 0:
                            uncompleted_clients = list(range(self.num_clients))
                        break
                    except queue.Empty:
                        pass

                    time.sleep(0.01)
            except queue.Empty:
                pass

            time.sleep(0.01)


class ServerV3(ServerV2):

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
        self.server_output_dict: Dict[int, torch.Tensor] = {}  # Dict[int, torch.Tensor]
        self.hidden_status_from_head_dict: Dict[int, torch.Tensor] = {}
        for client_id in range(self.num_clients):
            self.server_output_dict[client_id] = None
            self.hidden_status_from_head_dict[client_id] = None
        pass

    def _forward(self, client_id: int, activation: torch.Tensor, attention_mask: torch.LongTensor, position_embeddings: Tuple[torch.Tensor] = None):
        try:
            # self.logger.info(f"Client {client_id}: Acquiring lock for forward")
            with self.client_lock:
                # self.logger.info(f"Client {client_id}: Lock acquired for forward")
                hidden_status_from_head = activation.to(self.server_device)
                hidden_status_from_head.requires_grad_(True)
                hidden_status_from_head.retain_grad()
                attention_mask = attention_mask.to(self.server_device)
                pos_emb = tuple(torch.tensor(pos).to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
                if self.server_args["use_checkpoint"]:
                    server_output = torch.utils.checkpoint.checkpoint(
                        self.trunk_model.__call__,
                        hidden_status_from_head,
                        attention_mask,
                        pos_emb,
                        use_reentrant=False,
                    )
                else:
                    server_output: torch.Tensor = self.trunk_model(hidden_status_from_head, attention_mask, pos_emb)
                server_output = server_output.requires_grad_(True)
                self.server_output_dict[client_id] = server_output
                self.hidden_status_from_head_dict[client_id] = hidden_status_from_head
                activation_to_tail = server_output.cpu()
                # torch.cuda.empty_cache()
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Client {client_id}: Forward pass failed: {e}")
            raise e

    def _backward(self, client_id: int, server_grad: torch.Tensor):
        try:
            with self.client_lock:
                server_grad = server_grad.to(self.server_device)
                self.server_output_dict[client_id].backward(server_grad)
                torch.nn.utils.clip_grad_norm_(self.trunk_model.parameters(), max_norm=0.5)
                grad_to_head = self.hidden_status_from_head_dict[client_id].grad.cpu()
                torch.cuda.empty_cache()
                # self.logger.info(f"Client {client_id}: Releasing lock for backward")
            return grad_to_head
        except Exception as e:
            self.logger.error(f"Client {client_id}: Backward pass failed: {e}")
            raise e
        pass

    def compute_task(self):
        unforwarded_clients = list(range(self.num_clients))
        unbackwarded_clients = []

        while True:
            process_activation = True
            try:
                data = self.activation_queue.get_nowait()
                if data["client_id"] not in unforwarded_clients:
                    self.activation_queue.put(data)
                    time.sleep(0.001)
                    process_activation = False
                if process_activation:
                    self.logger.info(f"Processing activation for client_id={data['client_id']}, queue size={self.activation_queue.qsize()}")
                    server_activation = self._forward(
                        data["client_id"],
                        data["activation"],
                        data["attention_mask"],
                        data["position_embeddings"],
                    )
                    self.server_activation_queues[data["client_id"]].put({"client_id": data["client_id"], "server_activation": server_activation})
                    self.logger.info(f"Completed forward pass for client_id={data['client_id']}")
                    unforwarded_clients.remove(data["client_id"])
                    unbackwarded_clients.append(data["client_id"])

            except queue.Empty:
                pass

            try:
                data = self.gradient_queue.get_nowait()
                self.logger.info(f"Processing gradient for client_id={data['client_id']}, queue size={self.gradient_queue.qsize()}")
                d_activation_to_client = self._backward(data["client_id"], data["gradient"])
                self.server_activation_queues[data["client_id"]].put({"client_id": data["client_id"], "server_gradient": d_activation_to_client})
                self.logger.info(f"Completed backward pass for client_id={data['client_id']}")

                unbackwarded_clients.remove(data["client_id"])
                if len(unbackwarded_clients) == 0 and len(unforwarded_clients) == 0:
                    self._update_server_model()
                    unforwarded_clients = list(range(self.num_clients))

            except queue.Empty:
                pass
            time.sleep(0.01)

    def _update_server_model(self):
        with self.client_lock:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.logger.info("Server model updated")
