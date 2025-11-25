import queue
import logging
import time
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

import torch.utils.checkpoint
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
        checkpoint_client_num:int = -1,
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
        self.checkpoint_client_num = 0 if checkpoint_client_num <= 0 else checkpoint_client_num
        # self.checkpoint_clients=[]
        self.server_output_dict: Dict[int, torch.Tensor] = {}  # Dict[int, torch.Tensor]
        self.hidden_status_from_head_dict: Dict[int, torch.Tensor] = {}
        self.profile_datas: Dict[int, GanttChartData] = {} # 对应每一个客户端一个
        
        for client_id in range(self.num_clients):
            self.server_output_dict[client_id] = None
            self.hidden_status_from_head_dict[client_id] = None
            self.profile_datas[client_id] = [GanttChartData(batch_idx=i, client_id=client_id) for i in range(5)]
        pass

    def _forward(self, client_id: int, activation: torch.Tensor, attention_mask: torch.LongTensor, position_embeddings: Tuple[torch.Tensor] = None):
        try:
            hidden_status_from_head = activation.to(self.server_device)
            hidden_status_from_head.requires_grad_(True)
            hidden_status_from_head.retain_grad()
            attention_mask = attention_mask.to(self.server_device) if attention_mask is not None else None
            pos_emb = tuple(torch.tensor(pos).to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None
            if client_id < self.checkpoint_client_num:
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
            torch.cuda.empty_cache()
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Client {client_id}: Forward pass failed: {e}")
            raise e

    def _backward(self, client_id: int, server_grad: torch.Tensor, batch_idx: int):
        try:
            # with self.client_lock:
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_bwd_timestamp[0] = time.time()
            server_grad = server_grad.to(self.server_device)
            self.server_output_dict[client_id].backward(server_grad)
            torch.cuda.synchronize(self.server_device)
            torch.cuda.current_stream().synchronize()
            self.profile_datas[client_id][batch_idx].server_bwd_timestamp[1] = time.time()

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
                    # self.logger.info(f"Processing activation for client_id={data['client_id']}, queue size={self.activation_queue.qsize()}")
                    torch.cuda.current_stream().synchronize()
                    self.profile_datas[data["client_id"]][data["batch_idx"]].server_fwd_timestamp[0] = time.time()
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
                    unforwarded_clients.remove(data["client_id"])
                    unbackwarded_clients.append(data["client_id"])
                
                # fwd_start_time = time.time()
            except queue.Empty:
                pass
            try:
                data = self.gradient_queue.get_nowait()

                d_activation_to_client = self._backward(data["client_id"], data["gradient"], data["batch_idx"])
                self.server_activation_queues[data["client_id"]].put({"client_id": data["client_id"], "server_gradient": d_activation_to_client, "batch_idx": data["batch_idx"]})
                # self.logger.info(f"Completed backward pass for client_id={data['client_id']}")

                unbackwarded_clients.remove(data["client_id"])
                if len(unbackwarded_clients) == 0 and len(unforwarded_clients) == 0:
                    torch.cuda.current_stream().synchronize()
                    step_start_time = time.time()
                    for i in range(self.num_clients):
                        self.profile_datas[i][data["batch_idx"]].server_step_timestamp[0] = step_start_time
                    self._update_server_model()
                    
                    torch.cuda.current_stream().synchronize()
                    step_end_time = time.time()
                    for i in range(self.num_clients):
                        self.profile_datas[i][data["batch_idx"]].server_step_timestamp[1] = step_end_time
                    unforwarded_clients = list(range(self.num_clients))

            except queue.Empty:
                pass
            time.sleep(0.001)
            
    # def handle_client_with_aggregation(self, conn, addr, *args, **kwargs):
    #     super().handle_client_with_aggregation(conn, addr, *args, **kwargs)
    #     if self.checkpoint_client_num >0:
    #         self.checkpoint_client_num-=1
    #         self.logger.info(f"Checkpoint client num decreased from {self.checkpoint_client_num+1} to {self.checkpoint_client_num}")

    @torch.no_grad()
    def _update_server_model(self):
        # with self.client_lock:
        if self.server_args["use_avg"]:
            max_norm = 0.5*self.num_clients
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
