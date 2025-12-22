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
from dataclasses import dataclass, field
from usfl.server.server_base import ServerBase
from usfl.utils.exp import fed_avg_params, fed_average
from usfl.utils.timestamp_recorder import GanttChartData


class MergeServer(ServerBase):
    """
    V1: 每个客户端拥有独立的 Server Trunk Model (不共享权重)。
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
        enforce_sync: bool = True,
    ):
        super().__init__(
            server_args=server_args, server_device=server_device, num_clients=num_clients, logger=logger, matrix_logger=matrix_logger
        )
        self.lr = lr
        self.enforce_sync = enforce_sync

        self.trunk_model = server_model.to(self.server_device).train()
        self.optimizer = optimizer_clz(self.trunk_model.parameters(), lr=self.lr)

        self.server_output: torch.Tensor = None
        self.hidden_status_from_head: torch.Tensor = None

        self.aggregate_ready_clients: list[int] = []
        self.num_updates = 0

    def handle_client_with_aggregation(self, client_conn: socket.socket, addr: Tuple) -> None:
        client_id: int = None

        try:
            client_conn.settimeout(60.0)
            while True:
                try:
                    data: Dict = self.communicator.receive(client_conn)
                except socket.timeout:
                    self.logger.error(f"Client {addr} (ID: {client_id}) timed out while waiting for data.")
                    break  # 或者 continue，视逻辑而定
                except Exception as e:
                    self.logger.error(f"Error receiving data from Client {addr} (ID: {client_id}): {e}")
                    break

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
                    self._put_to_queue(self.activation_queue, data, "forward")
                    try:
                        # 尝试获取结果，设置超时
                        response = self.server_activation_queues[client_id].get(timeout=60.0)

                        # 检查 response 是否有效
                        if response is None:
                            raise ValueError(f"Got None from server_activation_queue for Client {client_id}")

                        if response["client_id"] == client_id:
                            with self._profile_scope(response["client_id"], response["batch_idx"], "server_fwd_send_timestamp"):
                                self._send_to_client(client_id, response)
                        else:
                            self.logger.error(f"ID Mismatch! Expected {client_id}, got {response['client_id']}")

                    except queue.Empty:
                        self.logger.error(
                            f"Timeout waiting for server forward computation for Client {client_id} (Batch {data.get('batch_idx')})"
                        )
                    except Exception as e:
                        self.logger.error(f"Error processing forward response for Client {client_id}: {e}")

                elif "gradient" in data:
                    self._put_to_queue(self.gradient_queue, data, "backward")

                    try:
                        response = self.server_activation_queues[client_id].get(timeout=60.0)

                        if response is None:
                            raise ValueError(f"Got None from server_activation_queue (Backward) for Client {client_id}")

                        if response["client_id"] == client_id and "server_gradient" in response:
                            self.profile_datas[response["client_id"]][response["batch_idx"]].server_bwd_send_timestamp[0] = time.time()
                            self._send_to_client(client_id, response)
                            self.profile_datas[response["client_id"]][response["batch_idx"]].server_bwd_send_timestamp[1] = time.time()
                    except queue.Empty:
                        self.logger.error(f"Timeout waiting for server backward computation for Client {client_id}")
                    except Exception as e:
                        self.logger.error(f"Error processing backward response for Client {client_id}: {e}")
                elif "aggregate" in data:
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
                                # aggregate client models
                                aggregate_client_models_start_time = time.time()
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
                                            "aggregate_start_time": aggregate_client_models_start_time,
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
                            f"order_{self.queue_order}",
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

    def compute_task(self):
        """根据配置选择同步或异步模式"""
        if self.enforce_sync:
            self._compute_task_sync()
        else:
            self._compute_task_async()

    def _compute_task_sync(self) -> None:
        while True:
            try:
                if self.activation_queue.qsize() != self.num_clients:
                    continue

                torch.cuda.current_stream().synchronize()
                for i in range(self.num_clients):
                    self.profile_datas[i][self.num_updates].server_fwd_timestamp[0] = time.time()

                activations_list = []
                attention_masks_list = []
                pos_cos_list = []
                pos_sin_list = []
                client_ids_in_batch = []
                batch_idxs_list = []
                client_sample_counts = []

                while not self.activation_queue.empty():
                    data = self._get_from_queue(self.activation_queue)
                    print(f"Processing activation for client_id={data['client_id']}, batch index={data['batch_idx']}")

                    c_id = data.get("client_id")
                    b_idx = data.get("batch_idx")
                    client_ids_in_batch.append(c_id)
                    batch_idxs_list.append(b_idx)

                    act = data["activation"]
                    current_sample_count = act.size(0)
                    client_sample_counts.append(current_sample_count)
                    activations_list.append(act)

                    if "attention_mask" in data and data["attention_mask"] is not None:
                        attention_masks_list.append(data["attention_mask"])

                    # 收集 Position Embeddings (Tuple 处理)
                    if "position_embeddings" in data and data["position_embeddings"] is not None:
                        pos_emb = data["position_embeddings"]
                        cos = pos_emb[0]
                        sin = pos_emb[1]

                        # 关键修复：检查第一维是否与 activation 的 batch_size 一致
                        # 如果客户端传的是 [1, seq, dim]，通过 expand 扩展成 [4, seq, dim]
                        if cos.size(0) != current_sample_count:
                            # 使用 expand 不会增加显存占用，只是改变视图
                            cos = cos.expand(current_sample_count, *cos.shape[1:])
                            sin = sin.expand(current_sample_count, *sin.shape[1:])

                        pos_cos_list.append(cos)
                        pos_sin_list.append(sin)

                server_inputs = torch.cat(activations_list, dim=0)

                if attention_masks_list:
                    server_attention_mask = torch.cat(attention_masks_list, dim=0)
                else:
                    server_attention_mask = None

                if pos_cos_list and pos_sin_list:
                    batched_cos = torch.cat(pos_cos_list, dim=0)
                    batched_sin = torch.cat(pos_sin_list, dim=0)
                    server_position_embeddings = (batched_cos, batched_sin)
                else:
                    server_position_embeddings = None

                # torch.cuda.current_stream().synchronize()
                # self.profile_datas[data["client_id"]][data["batch_idx"]].server_fwd_timestamp[0] = time.time()

                # self.logger.info(f"Processing activation for client_id={data['client_id']}, queue size={self.activation_queue.qsize()}")
                server_activation = self._forward(
                    server_inputs,
                    server_attention_mask,
                    server_position_embeddings,
                )
                activation_slices = torch.split(server_activation, client_sample_counts, dim=0)

                for c_id, b_idx, act_slice in zip(client_ids_in_batch, batch_idxs_list, activation_slices):
                    self.server_activation_queues[c_id].put(
                        {
                            "client_id": c_id,
                            "batch_idx": b_idx,  # 把刚才记录的 0 传回去，告诉客户端这是第 0 轮的结果
                            "server_activation": act_slice,
                        }
                    )

                torch.cuda.current_stream().synchronize()
                for i in range(self.num_clients):
                    self.profile_datas[i][self.num_updates].server_fwd_timestamp[1] = time.time()

                while True:
                    try:
                        if self.gradient_queue.qsize() != self.num_clients:
                            continue

                        server_bwd_start_time = time.time()
                        for i in range(self.num_clients):
                            self.profile_datas[i][self.num_updates].server_bwd_timestamp[0] = server_bwd_start_time

                        grad_dict = {}

                        while not self.gradient_queue.empty():
                            data = self._get_from_queue(self.gradient_queue)
                            c_id = data["client_id"]
                            grad_dict[c_id] = data["gradient"]

                        ordered_grads = []
                        for c_id in client_ids_in_batch:
                            ordered_grads.append(grad_dict[c_id])

                        merged_gradient = torch.cat(ordered_grads, dim=0)
                        self.optimizer.zero_grad()
                        d_activation_to_client = self._backward(merged_gradient)

                        d_inputs_slices = torch.split(d_activation_to_client, client_sample_counts, dim=0)
                        for c_id, b_idx, d_in_slice in zip(client_ids_in_batch, batch_idxs_list, d_inputs_slices):
                            grad_to_send = d_in_slice  # 或者是 d_in_slice
                            self.server_activation_queues[c_id].put(
                                {
                                    "client_id": c_id,
                                    "batch_idx": b_idx,
                                    "server_gradient": grad_to_send,
                                }
                            )

                        server_bwd_end_time = time.time()
                        for i in range(self.num_clients):
                            self.profile_datas[i][self.num_updates].server_bwd_timestamp[1] = server_bwd_end_time

                        self.num_updates += 1
                        break

                    except queue.Empty:
                        time.sleep(0.001)
            except queue.Empty:
                time.sleep(0.001)

    def _compute_task_async(self) -> None:
        pass

    def _forward(
        self,
        activation: torch.Tensor,
        attention_mask: torch.LongTensor,
        position_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        try:
            hidden_states_from_head = activation.to(self.server_device)
            hidden_states_from_head.requires_grad_(True)
            hidden_states_from_head.retain_grad()
            attention_mask = attention_mask.to(self.server_device) if attention_mask is not None else None
            pos_emb = (
                tuple(torch.tensor(pos).to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
            )
            # pos_emb = tuple(pos.clone().detach().to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
            server_output: torch.Tensor = self.trunk_model(
                hidden_states=hidden_states_from_head,
                attention_mask=attention_mask,
                position_embeddings=pos_emb,
            )
            server_output = server_output.requires_grad_(True)
            self.server_output = server_output
            self.hidden_status_from_head = hidden_states_from_head
            activation_to_tail = server_output.cpu()
            # self.client_fwd_progress[client_id] += 1
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Server forward pass failed: {e}")
            raise e

    def _backward(self, server_grad: torch.FloatTensor) -> torch.FloatTensor:
        try:
            server_grad = server_grad.to(self.server_device)
            self.optimizer.zero_grad()
            self.server_output.backward(server_grad)
            self.optimizer.step()
            grad_to_head = self.hidden_status_from_head.grad.cpu()

            return grad_to_head
        except Exception as e:
            self.logger.error(f"Server backward pass failed: {e}")
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
