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


class FedServer:

    def __init__(
        self,
        num_clients: int,
        port: int,
        logger: logging.Logger,
    ):
        self.num_clients = num_clients
        self.port = port
        self.logger = logger
        self.clients: List[Tuple[socket.socket, Tuple[str, int], Optional[nn.Module]]] = []
        self.client_models_dict: Dict[int, Dict[str, Any]] = {}
        self.communicator = SocketCommunicator(port=port, is_server=True)
        self.clients_comm_threads: List[threading.Thread] = []
        self.client_lock = threading.Lock()
        self.aggregate_count = 0

    def handle_client_with_aggregation(self, client_conn: socket.socket, addr: Tuple) -> None:
        """
        Aggragate the client models to server model
        """
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
                elif "aggregate_client" in data:
                    self.logger.info(f"Received aggregation request from client_id={client_id} (addr={addr})")
                    with self.client_lock:
                        self.client_models_dict[client_id] = data
                        self.aggregate_count += 1
                        self.logger.info(f"Aggregate count: {self.aggregate_count}/{self.num_clients}")
                        if self.aggregate_count == self.num_clients:
                            # aggregate the client models
                            avg_dict = self._aggregate_client_models()
                            # Send acknowledgment to all clients
                            for cid in range(self.num_clients):
                                self._send_to_client(cid, {"status": "aggregate_complete", "state_dict": avg_dict})
                        else:
                            self._send_to_client(client_id, {"status": "waiting_for_others"})
        except Exception as e:
            self.logger.error(f"Client {addr} (client_id={client_id}) error: {e}")
        finally:
            with self.client_lock:
                self.clients[:] = [(c, a, cid) for c, a, cid in self.clients if c != client_conn]
            client_conn.close()
            self.logger.info(f"Client {addr} (client_id={client_id}) closed")

    def _aggregate_client_models(self) -> Dict[str, Any]:
        """
        Aggregate the client models to server model
        """
        w_avg = fed_average(list(self.client_models_dict.values()))
        self.client_models_dict.clear()
        self.aggregate_count = 0
        return w_avg

    def run(self):
        """
        Start the server
        """
        self.logger.info("Waiting for clients to connect...")
        while True:
            conn, addr = self.communicator.accept_client()
            if conn:
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
