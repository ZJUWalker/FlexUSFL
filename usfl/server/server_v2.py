import socket
import queue
import logging
import time
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import json
import os

from usfl.server.server_base import ServerBase
from usfl.utils.timestamp_recorder import GanttChartData


class ServerV2(ServerBase):
    """
    V2: 所有客户端共享同一个 Server Trunk Model (共享权重)。
    """

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
        enforce_sync: bool = False,
    ):
        super().__init__(
            server_args=server_args, server_device=server_device, num_clients=num_clients, logger=logger, matrix_logger=matrix_logger
        )
        self.lr = lr
        self.enforce_sync = enforce_sync

        # --- V2 特有逻辑：单模型 ---
        self.trunk_model = server_model.to(self.server_device).train()
        self.optimizer = optimizer_clz(self.trunk_model.parameters(), lr=self.lr)

        self.server_output: torch.Tensor = None
        self.hidden_status_from_head: torch.Tensor = None
        self.server_update_count = 0

    def _forward(
        self,
        client_id: int,
        activation: torch.Tensor,
        attention_mask: torch.LongTensor,
        position_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        try:
            hidden_status_from_head = activation.to(self.server_device)
            hidden_status_from_head.requires_grad_(True)
            hidden_status_from_head.retain_grad()
            attention_mask = attention_mask.to(self.server_device) if attention_mask is not None else None
            pos_emb = (
                tuple(torch.tensor(pos).to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
            )
            server_output: torch.Tensor = self.trunk_model(
                hidden_states=hidden_status_from_head,
                attention_mask=attention_mask,
                position_embeddings=pos_emb,
            )
            torch.cuda.synchronize(device=self.server_device)
            server_output = server_output.requires_grad_(True)
            self.server_output = server_output
            self.hidden_status_from_head = hidden_status_from_head
            activation_to_tail = server_output.cpu()
            torch.cuda.empty_cache()
            if self.queue_order == "straggler_fo":
                self.client_fwd_progress[client_id] += 1
                # print(f"Client {client_id} forward progress: {self.client_fwd_progress[client_id]}")
                # print(f"client_fwd_progress: {self.client_fwd_progress}")
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Client {client_id}: Forward pass failed: {e}")
            raise e

    def _backward(self, client_id: int, server_grad: torch.Tensor, batch_idx: int) -> torch.Tensor:
        try:
            with self._profile_scope(client_id, batch_idx, "server_bwd_timestamp"):
                server_grad = server_grad.to(self.server_device)
                self.optimizer.zero_grad()
                self.server_output.backward(server_grad)

            # torch.nn.utils.clip_grad_norm_(self.trunk_model.parameters(), max_norm=0.5)
            with self._profile_scope(client_id, batch_idx, "server_step_timestamp"):
                self.optimizer.step()
                grad_to_head = self.hidden_status_from_head.grad.cpu()

            torch.cuda.empty_cache()

            if self.queue_order == "straggler_fo":
                self.client_bwd_progress[client_id] += 1
                # print(f"Client {client_id} backward progress: {self.client_bwd_progress[client_id]}")
                # print(f"client_bwd_progress: {self.client_bwd_progress}")
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
                data: Dict = self.communicator.receive(conn)
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
                    # print(f"[Server] received activation from client {client_id} for batch {data['batch_idx']}, time: {time.time()}")
                    self._put_to_queue(self.activation_queue, data, "forward")
                    response = self._get_from_queue(self.server_activation_queues[client_id], timeout=120.0)
                    if response["client_id"] == client_id:
                        with self._profile_scope(response["client_id"], response["batch_idx"], "server_fwd_send_timestamp"):
                            self._send_to_client(client_id, response)
                elif "gradient" in data:
                    self._put_to_queue(self.gradient_queue, data, "backward")
                    response = self._get_from_queue(self.server_activation_queues[client_id], timeout=120.0)
                    if response["client_id"] == client_id and "server_gradient" in response:
                        try:
                            with self._profile_scope(response["client_id"], response["batch_idx"], "server_bwd_send_timestamp"):
                                self._send_to_client(client_id, response)
                        except Exception as e:
                            print(f"Error sending server gradient to client {addr} (client_id={client_id}): {e}")
                            raise e
                elif "aggregate" in data:
                    with self.client_lock:
                        self.client_head_model_params[client_id] = data["head_params"]
                        self.client_tail_model_params[client_id] = data["tail_params"]
                        self.client_model_losses[client_id] = data["loss"]
                        self.aggregate_count += 1
                        if data["client_id"] == 0:
                            self.logger.info(f"Waiting for aggregation, step {data['step']}")
                        if self.aggregate_count == self.num_clients:
                            # aggregate the client models
                            aggregate_client_models_start_time = time.time()
                            head_params_avg, tail_params_avg, loss = self._aggregate_client_models()
                            self.aggregate_count = 0
                            self.logger.info(f"Aggregated client models finished, avg loss: {loss}")
                            self.logger.info(
                                f"Aggrageting server model finished, "
                                f"total compute time: {self.compute_time:.2f}s, "
                                f"total server aggregate time: {self.aggregate_server_time:.2f}s, "
                                f"total clients aggregate time: {self.aggregate_client_time:.2f}s"
                            )
                            # Send acknowledgment to all clients
                            for cid in range(self.num_clients):
                                self._send_to_client(
                                    cid,
                                    {
                                        "status": "aggregate_client_complete",
                                        "head_params": head_params_avg,
                                        "tail_params": tail_params_avg,
                                        "loss": loss,
                                        "aggregate_start_time": aggregate_client_models_start_time,
                                    },
                                )
                            self.matrix_logger.info(
                                f"{data['step']:^5}|"
                                f"{torch.cuda.max_memory_allocated(self.server_device)/1024**3:^15.3f}|"
                                f"{torch.cuda.max_memory_reserved(self.server_device)/1024**3:^18.3f}|"
                                f"{loss:^10.4f}"
                            )
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats(self.server_device)
                        pass
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
                            # f"bps_{self.server_args['batch_per_sync']}",
                            f"order_{self.server_args['queue_order']}",
                        )
                        os.makedirs(save_dir, exist_ok=True)

                        save_path = save_dir + f"/server_profile_data.json"
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
                        break
        except Exception as e:
            # 注意：这里不需要把 e 放到 f-string 里，logger 会自动处理
            self.logger.exception(f"Client {addr} (client_id={client_id}) failed")
        finally:
            with self.client_lock:
                self.clients[:] = [(c, a, cid) for c, a, cid in self.clients if c != conn]
            conn.close()
            self.logger.info(f"Client {addr} (client_id={client_id}) closed")
            if self.clients == []:
                self.logger.info(
                    f"All clients disconnected, server shutting down ,"
                    f"total compute time: {self.compute_time:.2f} s, "
                    f"total server aggregate time: {self.aggregate_server_time:.2f} s, "
                    f"total clients aggregate time: {self.aggregate_client_time:.2f} s, "
                    f"total idle time: {self.idle_time:.2f} s"
                )

    def compute_task(self):
        if self.enforce_sync:
            self._compute_task_sync()
        else:
            self._compute_task_async()

    def _compute_task_sync(self):
        uncompleted_clients = list(range(self.num_clients))
        temp_buffer = []
        while True:
            try:
                # data = self.activation_queue.get_nowait()
                data = self._get_from_queue(self.activation_queue)

                if data["client_id"] not in uncompleted_clients:
                    temp_buffer.append(data)
                    continue

                with self._profile_scope(data["client_id"], data["batch_idx"], "server_fwd_timestamp"):
                    server_activation = self._forward(
                        data["client_id"],
                        data["activation"],
                        data["attention_mask"] if "attention_mask" in data else None,
                        data["position_embeddings"] if "position_embeddings" in data else None,
                    )
                    self._put_to_queue(
                        self.server_activation_queues[data["client_id"]],
                        {
                            "client_id": data["client_id"],
                            "server_activation": server_activation,
                            "batch_idx": data["batch_idx"],
                        },
                        phase="forward",
                    )

                while True:
                    try:
                        data = self._get_from_queue(self.gradient_queue)
                        d_activation_to_client = self._backward(data["client_id"], data["gradient"], data["batch_idx"])
                        self._put_to_queue(
                            self.server_activation_queues[data["client_id"]],
                            {
                                "client_id": data["client_id"],
                                "server_gradient": d_activation_to_client,
                                "batch_idx": data["batch_idx"],
                            },
                            phase="backward",
                        )

                        uncompleted_clients.remove(data["client_id"])
                        if len(uncompleted_clients) == 0:
                            uncompleted_clients = list(range(self.num_clients))
                            for buffer_data in temp_buffer:
                                self._put_to_queue(self.activation_queue, buffer_data, phase="forward")
                            temp_buffer = []
                        break
                    except queue.Empty:
                        time.sleep(0.001)

            except queue.Empty:
                time.sleep(0.001)

    def _compute_task_async(self):
        """非强同步版本：先到先服务，不等待其他客户端"""
        while True:
            try:
                # data = self.activation_queue.get_nowait()
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
                        {
                            "client_id": data["client_id"],
                            "server_activation": server_activation,
                            "batch_idx": data["batch_idx"],
                        },
                        phase="forward",
                    )

                while True:
                    try:
                        data = self._get_from_queue(self.gradient_queue)
                        d_activation_to_client = self._backward(data["client_id"], data["gradient"], data["batch_idx"])
                        # self.server_activation_queues[data["client_id"]].put(
                        #     {"client_id": data["client_id"], "server_gradient": d_activation_to_client, "batch_idx": data["batch_idx"]}
                        # )
                        self._put_to_queue(
                            self.server_activation_queues[data["client_id"]],
                            {
                                "client_id": data["client_id"],
                                "server_gradient": d_activation_to_client,
                                "batch_idx": data["batch_idx"],
                            },
                            phase="backward",
                        )
                        break
                    except queue.Empty:
                        time.sleep(0.001)
            except queue.Empty:
                time.sleep(0.001)
