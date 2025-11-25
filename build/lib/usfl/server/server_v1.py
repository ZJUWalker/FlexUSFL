import socket
import threading
import queue
import logging
import time
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple
import json
from dataclasses import asdict
import os

from usfl.server.server_base import ServerBase
from usfl.utils.exp import fed_avg_params, fed_average
from usfl.utils.timestamp_recorder import GanttChartData


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
        self.profile_datas: Dict[int, GanttChartData] = {}  # 对应每一个客户端一个

        self.optimizers: Dict[int, torch.optim.Optimizer] = {}  # 用于保存服务端模型的优化器 V1
        self.aggregate_ready_clients: list[int] = []  # 用于保存准备好聚合的客户端id
        # -------------------------------------------------------------
        self.server_hidden_states_output: Dict[int, torch.FloatTensor] = {}  # 用于保存客户端的输出激活量
        self.hidden_states_from_head: Dict[int, torch.FloatTensor] = {}  # 用于保存服务端的输入激活量

        queue_order_raw = server_args.get("queue_order", "lifo")  # （fifo / lifo）
        self.queue_order = str(queue_order_raw).lower()

        if self.queue_order == "lifo":
            self.activation_queue = queue.LifoQueue()
            self.gradient_queue = queue.LifoQueue()
        else:
            self.activation_queue = queue.Queue()
            self.gradient_queue = queue.Queue()

        self.logger.info(f"[Server] Using {self.queue_order.upper()} queues for activation/gradient scheduling.")
        # <<< 新增结束

        for client_id in range(num_clients):
            self.trunk_models[client_id] = copy.deepcopy(server_model)
            self.trunk_models[client_id].to(self.server_device).train()
            self.optimizers[client_id] = optimizer_clz(self.trunk_models[client_id].parameters(), lr=self.lr)
            self.server_hidden_states_output[client_id] = None
            self.hidden_states_from_head[client_id] = None
            self.profile_datas[client_id] = [GanttChartData(batch_idx=i, client_id=client_id) for i in range(5)]
            # self.profile_data = [GanttChartData(batch_idx=i, client_id=client_id) for i in range(self.batch_num)]

    def handle_client_with_aggregation(self, client_conn: socket.socket, addr: Tuple) -> None:
        client_id: int = None

        try:
            client_conn.settimeout(60.0)
            while True:
                data: Dict = self.communicator.receive(client_conn)
                if data is None:
                    self.logger.info(f"Client {addr} disconnected")
                    break

                if "client_id" in data and client_id is None:
                    client_id = data["client_id"]
                    # if "activation" in data:
                    #     print(f"Server received activation from client_id={client_id}, batch_idx={data['batch_idx']}, timestamp={time.time()}")
                    with self.client_lock:
                        for i, (c, a, cid) in enumerate(self.clients):
                            if c == client_conn and cid is None:
                                self.clients[i] = (c, a, client_id)
                                self.logger.info(f"Assigned client_id={client_id} to addr={addr}")
                                break
                if "activation" in data:
                    # self.logger.info(f"Received activation from client_id={client_id} (addr={addr}): shape={data['activation'].shape}")
                    print(f"Server received activation from client_id={client_id}, batch_idx={data['batch_idx']}, timestamp={time.time()}")
                    # 这里保持不变，底层队列类型已经根据 queue_order 决定 FIFO / LIFO
                    self.activation_queue.put(data)

                    response = self.server_activation_queues[client_id].get(timeout=10.0)
                    if response["client_id"] == client_id:
                        self.profile_datas[response["client_id"]][response["batch_idx"]].server_fwd_send_timestamp[0] = time.time()
                        self._send_to_client(client_id, response)
                        self.profile_datas[response["client_id"]][response["batch_idx"]].server_fwd_send_timestamp[1] = time.time()
                elif "gradient" in data:
                    # self.logger.info(f"Received gradient from client_id={client_id} (addr={addr}): shape={data['gradient'].shape}")
                    # 同样，这里只调用 put，FIFO / LIFO 由队列类型决定
                    self.gradient_queue.put(data)
                    response = self.server_activation_queues[client_id].get(timeout=10.0)
                    if response["client_id"] == client_id and "server_gradient" in response:
                        self.profile_datas[response["client_id"]][response["batch_idx"]].server_bwd_send_timestamp[0] = time.time()
                        self._send_to_client(client_id, response)
                        self.profile_datas[response["client_id"]][response["batch_idx"]].server_bwd_send_timestamp[1] = time.time()
                elif "aggregate" in data:
                    # self.logger.info(f"Received aggregation request from client_id={client_id} (addr={addr})")
                    with self.client_lock:
                        try:
                            self.aggregate_count += 1
                            self.client_head_model_params[client_id] = data["head_params"]
                            self.client_tail_model_params[client_id] = data["tail_params"]
                            self.client_model_losses[client_id] = data["loss"]
                            self.aggregate_ready_clients.append(client_id)
                            if data["client_id"] == 0:
                                self.logger.info(f"Waiting for aggregation, step {data['step']}")
                            if self.aggregate_count == self.num_clients:
                                # aggregate server models
                                self._aggregate_trunk_models()
                                # aggregate client models
                                head_params_avg, tail_params_avg, avg_loss = self._aggregate_client_models()
                                self.aggregate_count = 0
                                # log memory usage
                                self.matrix_logger.info(
                                    f"{data['step']:^5}|"
                                    f"{torch.cuda.max_memory_allocated(self.server_device)/1024**3:^15.3f}|"
                                    f"{torch.cuda.max_memory_reserved(self.server_device)/1024**3:^18.3f}"
                                    f"{avg_loss:^10.4f}"
                                )
                                self.logger.info(
                                    f"Aggrageting server model finished, "
                                    f"total compute time: {self.compute_time:.2f}s, "
                                    f"total server aggregate time: {self.aggregate_server_time:.2f}s, "
                                    f"total clients aggregate time: {self.aggregate_client_time:.2f}s, "
                                    f"total idle time: {self.idle_time:.2f}s"
                                )
                                torch.cuda.reset_peak_memory_stats(self.server_device)
                                # Send acknowledgment to all clients
                                for cid in range(self.num_clients):
                                    self._send_to_client(
                                        client_id=cid,
                                        response={
                                            "status": "aggregate_complete",
                                            "head_params": head_params_avg,
                                            "tail_params": tail_params_avg,
                                            "loss": avg_loss,
                                        },
                                    )
                        except Exception as e:
                            self.logger.error(f"Aggregation failed: {e}")
                elif "stop" in data:
                    self.stop_count += 1
                    print(f"stop_count={self.stop_count}, num_clients={self.num_clients}")
                    if self.stop_count == self.num_clients:
                        print("All clients sent stop signal, server shutting down.")
                        # print(self.profile_datas)
                        serializable_dict = {key: [asdict(item) for item in lst[:-1]] for key, lst in self.profile_datas.items()}
                        save_dir = os.path.join(
                            "./vis",
                            f"version_{self.server_args['version']}",
                            f"model_{self.server_args['model'].split('/')[-1]}",
                            f"dataset_{self.server_args['dataset']}",
                            f"lag_{self.server_args['lag_ratio']}",
                            f"client_num_{self.server_args['num_clients']}",
                            f"order_{self.queue_order}"
                        )
                        os.makedirs(save_dir, exist_ok=True)

                        save_path = save_dir + f"/server_profile_data.json"
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
                        break
        except Exception as e:
            self.logger.error(f"Client {addr} (client_id={client_id}) error: {e}")
        finally:
            with self.client_lock:
                self.clients[:] = [(c, a, cid) for c, a, cid in self.clients if c != client_conn]
            client_conn.close()
            self.logger.info(f"Client {addr} (client_id={client_id}) closed")
            if self.clients == []:
                self.logger.info(
                    f"All clients disconnected, server shutting down ,"
                    f"total compute time: {self.compute_time:.2f} s, "
                    f"total server aggregate time: {self.aggregate_server_time:.2f} s, "
                    f"total clients aggregate time: {self.aggregate_client_time:.2f} s, "
                    f"total idle time: {self.idle_time:.2f} s"
                )

    def compute_task(self) -> None:
        while True:
            try:
                # 这里使用 get_nowait，底层队列是 Queue 或 LifoQueue，
                # 决定了是先到先服务还是后到先服务
                data = self.activation_queue.get_nowait()

                torch.cuda.current_stream().synchronize()
                self.profile_datas[data["client_id"]][data["batch_idx"]].server_fwd_timestamp[0] = time.time()

                # self.logger.info(f"Processing activation for client_id={data['client_id']}, queue size={self.activation_queue.qsize()}")
                server_activation = self._forward(
                    data["client_id"],
                    data["activation"],
                    data["attention_mask"] if "attention_mask" in data else None,
                    data["position_embeddings"] if "position_embeddings" in data else None,
                )
                self.server_activation_queues[data["client_id"]].put(
                    {"client_id": data["client_id"], "batch_idx": data["batch_idx"], "server_activation": server_activation}
                )
                torch.cuda.current_stream().synchronize()
                self.profile_datas[data["client_id"]][data["batch_idx"]].server_fwd_timestamp[1] = time.time()

            except queue.Empty:
                pass
            try:
                data = self.gradient_queue.get_nowait()
                # self.logger.info(f"Processing gradient for client_id={data['client_id']}, queue size={self.gradient_queue.qsize()}")

                d_activation_to_client = self._backward(data["client_id"], data["gradient"], data["batch_idx"])

                self.server_activation_queues[data["client_id"]].put(
                    {"client_id": data["client_id"], "batch_idx": data["batch_idx"], "server_gradient": d_activation_to_client}
                )
                # self.logger.info(f"Completed backward pass for client_id={data['client_id']}")
            except queue.Empty:
                pass
            time.sleep(0.01)

    def _forward(
        self, client_id: int, activation: torch.Tensor, attention_mask: torch.LongTensor, position_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        try:
            hidden_states_from_head = activation.to(self.server_device)
            hidden_states_from_head.requires_grad_(True)
            hidden_states_from_head.retain_grad()
            attention_mask = attention_mask.to(self.server_device) if attention_mask is not None else None
            pos_emb = tuple(torch.tensor(pos).to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
            # pos_emb = tuple(pos.clone().detach().to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
            server_output: torch.Tensor = self.trunk_models[client_id](
                hidden_states=hidden_states_from_head,
                attention_mask=attention_mask,
                position_embeddings=pos_emb,
            )
            server_output = server_output.requires_grad_(True)
            self.server_hidden_states_output[client_id] = server_output
            self.hidden_states_from_head[client_id] = hidden_states_from_head
            activation_to_tail = server_output.cpu()
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Client {client_id}: Forward pass failed: {e}")
            raise e

    def _backward(self, client_id: int, server_grad: torch.FloatTensor, batch_idx: int) -> torch.FloatTensor:
        try:
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_bwd_timestamp[0] = time.time()
            server_grad = server_grad.to(self.server_device)
            self.optimizers[client_id].zero_grad()
            self.server_hidden_states_output[client_id].backward(server_grad)
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_bwd_timestamp[1] = time.time()

            # torch.nn.utils.clip_grad_norm_(self.trunk_models[client_id].parameters(), max_norm=0.5)
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_step_timestamp[0] = time.time()
            self.optimizers[client_id].step()
            grad_to_head = self.hidden_states_from_head[client_id].grad.cpu()
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_step_timestamp[1] = time.time()

            return grad_to_head
        except Exception as e:
            self.logger.error(f"Client {client_id}: Backward pass failed: {e}")
            raise e

    def _reset_aggregate_count(self):
        # with self.aggregate_lock:
        super()._reset_aggregate_count()
        self.aggregate_ready_clients = []

    def _aggregate_trunk_models(self) -> dict:
        if len(self.trunk_models.keys()) == 1:
            return {"status": "aggregation_completed"}
        try:
            start_agg_time = time.time()
            w_trunk_model_params = []
            for client_id in range(self.num_clients):
                w_trunk_model_params.append([p for p in self.trunk_models[client_id].parameters() if p.requires_grad])
            avg_trunk_model_params = fed_avg_params(w_trunk_model_params)
            for client_id in range(self.num_clients):
                i = 0
                for p in self.trunk_models[client_id].parameters():
                    if p.requires_grad:
                        p.data.copy_(avg_trunk_model_params[i], non_blocking=True)
                        i += 1
            torch.cuda.synchronize(device=self.server_device)
            end_agg_time = time.time()
            self.aggregate_server_time += end_agg_time - start_agg_time  # 用于记录聚合时间
            torch.cuda.empty_cache()
            return {"status": "aggregation_completed"}
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            raise e
