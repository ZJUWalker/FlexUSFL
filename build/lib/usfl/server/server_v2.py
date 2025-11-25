import socket
import queue
import logging
import time
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

from typing import Dict, List, Tuple
import json
from dataclasses import asdict
import os

from usfl.server.server_base import ServerBase
from usfl.utils.timestamp_recorder import GanttChartData

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
        self.profile_datas: Dict[int, GanttChartData] = {} # 对应每一个客户端一个
        self.optimizer: torch.optim.Optimizer = optimizer_clz(self.trunk_model.parameters(), lr=self.lr)
        self.matrix_logger = matrix_logger
        self.server_output: torch.Tensor = None
        self.hidden_status_from_head: torch.Tensor = None
        for client_id in range(self.num_clients):
            self.profile_datas[client_id] = [GanttChartData(batch_idx=i, client_id=client_id) for i in range(5)]

    def _forward(
        self, client_id: int, activation: torch.Tensor, attention_mask: torch.LongTensor, position_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        try:
            hidden_status_from_head = activation.to(self.server_device)
            hidden_status_from_head.requires_grad_(True)
            hidden_status_from_head.retain_grad()
            attention_mask = attention_mask.to(self.server_device) if attention_mask is not None else None
            pos_emb = tuple(torch.tensor(pos).to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
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
            # self.logger.info(f"Client {client_id}: Releasing lock for forward")
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Client {client_id}: Forward pass failed: {e}")
            raise e

    def _backward(self, client_id: int, server_grad: torch.Tensor, batch_idx: int) -> torch.Tensor:
        try:
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_bwd_timestamp[0] = time.time()
            server_grad = server_grad.to(self.server_device)
            self.optimizer.zero_grad()
            self.server_output.backward(server_grad)
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_bwd_timestamp[1] = time.time()
            
            # torch.nn.utils.clip_grad_norm_(self.trunk_model.parameters(), max_norm=0.5)
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_step_timestamp[0] = time.time()
            self.optimizer.step()
            grad_to_head = self.hidden_status_from_head.grad.cpu()
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_step_timestamp[1] = time.time()
            
            torch.cuda.empty_cache()
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
                    # self.logger.info(f"Received activation from client_id={client_id} (addr={addr}): shape={data['activation'].shape}")
                    self.activation_queue.put(data)
                    response = self.server_activation_queues[client_id].get(timeout=60.0)
                    if response["client_id"] == client_id:
                        self.profile_datas[response["client_id"]][response["batch_idx"]].server_fwd_send_timestamp[0] = time.time()
                        self._send_to_client(client_id, response)
                        self.profile_datas[response["client_id"]][response["batch_idx"]].server_fwd_send_timestamp[1] = time.time()
                elif "gradient" in data:
                    # self.logger.info(f"Received gradient from client_id={client_id} (addr={addr}): shape={data['gradient'].shape}")
                    self.gradient_queue.put(data)
                    response = self.server_activation_queues[client_id].get(timeout=60.0)
                    if response["client_id"] == client_id and "server_gradient" in response:
                        try:
                            self.profile_datas[response["client_id"]][response["batch_idx"]].server_bwd_send_timestamp[0] = time.time()
                            self._send_to_client(client_id, response)
                            self.profile_datas[response["client_id"]][response["batch_idx"]].server_bwd_send_timestamp[1] = time.time()
                        except Exception as e:
                            print(f"Error sending server gradient to client {addr} (client_id={client_id}): {e}")
                            raise e
                elif "aggregate" in data:
                    with self.client_lock:
                        self.client_head_model_params[client_id] = data["head_params"]
                        self.client_tail_model_params[client_id] = data["tail_params"]
                        self.client_model_losses[client_id] = data["loss"]
                        self.aggregate_count += 1
                        if data['client_id'] == 0:
                            self.logger.info(f"Waiting for aggregation, step {data['step']}")
                        if self.aggregate_count == self.num_clients:
                            # aggregate the client models
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
        uncompleted_clients = list(range(self.num_clients))
        while True:
            try:
                data = self.activation_queue.get_nowait()
                
                if data["client_id"] not in uncompleted_clients:
                    self.activation_queue.put(data, timeout=60.0)
                    time.sleep(0.001)
                    continue
                
                torch.cuda.current_stream().synchronize()
                self.profile_datas[data["client_id"]][data["batch_idx"]].server_fwd_timestamp[0] = time.time()
                # self.logger.info(f"Processing activation for client_id={data['client_id']}, queue size={self.activation_queue.qsize()}")
                server_activation = self._forward(
                    data["client_id"],
                    data["activation"],
                    data["attention_mask"] if "attention_mask" in data else None,
                    data["position_embeddings"] if "position_embeddings" in data else None,
                )
                self.server_activation_queues[data["client_id"]].put({"client_id": data["client_id"], "server_activation": server_activation, "batch_idx": data["batch_idx"]})
                torch.cuda.current_stream().synchronize()
                self.profile_datas[data["client_id"]][data["batch_idx"]].server_fwd_timestamp[1] = time.time()
                
                # self.logger.info(f"Completed forward pass for client_id={data['client_id']}")

                if data["is_training"] == False:
                    continue
                
                while True:
                    try:
                        data = self.gradient_queue.get_nowait()
                        # self.logger.info(f"Processing gradient for client_id={data['client_id']}, queue size={self.gradient_queue.qsize()}")
                        d_activation_to_client = self._backward(data["client_id"], data["gradient"], data["batch_idx"])
                        self.server_activation_queues[data["client_id"]].put(
                            {"client_id": data["client_id"], "server_gradient": d_activation_to_client, "batch_idx": data["batch_idx"]}
                        )
                        # self.logger.info(f"Completed backward pass for client_id={data['client_id']}")

                        uncompleted_clients.remove(data["client_id"])
                        if len(uncompleted_clients) == 0:
                            uncompleted_clients = list(range(self.num_clients))
                        break
                    except queue.Empty:
                        pass

                    time.sleep(0.001)
            except queue.Empty:
                pass

            time.sleep(0.001)
