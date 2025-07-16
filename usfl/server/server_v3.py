import queue
import logging
import time
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

import torch.utils.checkpoint
from usfl.server.server_v2 import ServerV2


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
