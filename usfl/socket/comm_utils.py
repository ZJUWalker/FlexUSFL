import io
import socket
import pickle
import asyncio
import time
import numpy as np
import random


def calculate_network_delay(data_size_bytes, bandwidth_mbps=10, propagation_delay_ms=50, jitter_ms=10):
    """
    Calculate simulated network delay based on data size, bandwidth, propagation delay, and jitter.

    Args:
        data_size_bytes (int): Size of the data in bytes.
        bandwidth_mbps (float): Network bandwidth in Mbps (default: 10 Mbps).
        propagation_delay_ms (float): Base propagation delay in milliseconds (default: 50ms).
        jitter_ms (float): Jitter range in milliseconds (default: ±10ms).

    Returns:
        float: Simulated delay in seconds.
    """
    # Convert bandwidth to bytes per second
    bandwidth_bytes_per_sec = bandwidth_mbps * 125000

    # Calculate transmission delay in seconds
    transmission_delay = data_size_bytes / bandwidth_bytes_per_sec

    # Add propagation delay (convert ms to seconds)
    propagation_delay = propagation_delay_ms / 1000.0

    # Add random jitter (convert ms to seconds)
    jitter = random.uniform(-jitter_ms, jitter_ms) / 1000.0

    # Total delay in seconds
    total_delay = max(0, transmission_delay + propagation_delay + jitter)

    return total_delay


class SocketCommunicator(object):
    """
    _summary_ : Socket通信类工具

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        host="localhost",
        port=8888,
        is_server=False,
        buffer_size=4 * 4096,
        similuate_delay=True,
        **kwargs,
    ):
        self.host = host
        self.port = port
        self.is_server = is_server
        self.conn = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(kwargs.get("timeout", 600))  # 默认30秒超时
        self.buffer_size = buffer_size
        self.max_retry = kwargs.get("max_retry", 10)
        self.simuluate_delay = similuate_delay
        if self.is_server:
            self.clients = []  # 存储客户端连接
            self._init_server()
        else:
            self._init_client()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn and self.conn is not self.sock:
            self.conn.close()
        if self.is_server:
            for client, _ in self.clients:
                client.close()
        self.sock.close()

    def _init_server(self):
        try:
            self.sock.bind((self.host, self.port))
            self.sock.listen()
            print(f"[服务端] 正在监听 {self.host}:{self.port} ...")
        except socket.error as e:
            print(f"[服务端] 绑定失败: {e}")
            raise

    def _init_client(self):
        print(f"[客户端] 尝试连接 {self.host}:{self.port} ...")
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                self.sock.connect((self.host, self.port))
                print("[客户端] 已连接服务端")
                self.conn = self.sock
                break
            except socket.error as e:
                print(f"[客户端] 连接失败: {e}, 重试次数: {retry_count + 1}")
                retry_count += 1
                time.sleep(5)  # 等待5秒后重试
        if retry_count == self.max_retry:
            raise Exception("连接服务端失败")

    def accept_client(self):
        """服务器接受新客户端连接"""
        try:
            conn, addr = self.sock.accept()
            self.clients.append((conn, addr))
            print(f"[服务端] 已连接来自 {addr}")
            return conn, addr
        except socket.timeout:
            return None, None

    def send(self, obj, conn=None):
        """发送对象，带长度前缀"""
        if conn is None:
            conn = self.conn
        try:
            # 序列化对象并计算大小
            start_time = time.time()
            data = pickle.dumps(obj)
            # data_size = len(data)
            # if self.simuluate_delay:
            #     # 计算网络延迟（包括序列化和传输时间）
            #     delay = calculate_network_delay(
            #         data_size_bytes=data_size,
            #         bandwidth_mbps=230,  # 可调整带宽（Mbps）
            #         propagation_delay_ms=50,  # 可调整传播延迟（ms）
            #         jitter_ms=0,  # 可调整抖动范围（ms）
            #     )

            #     serialization_time = time.time() - start_time
            #     total_delay = max(0, delay + serialization_time)  # 包含序列化时间

            #     time.sleep(total_delay)

            # 发送数据
            length = len(data)
            conn.sendall(length.to_bytes(4, byteorder="big"))  # 发送4字节长度
            conn.sendall(data)
        except socket.error as e:
            print(f"[错误] 发送失败: {e}")
            raise

    def receive(self, conn=None):
        """接收对象，基于长度前缀"""
        if conn is None:
            conn = self.conn
        conn.settimeout(600)
        try:
            # 接收长度前缀
            start_time = time.time()
            length_bytes = conn.recv(4)
            if not length_bytes:
                return None
            length = int.from_bytes(length_bytes, byteorder="big")

            # 接收数据
            data = bytearray()
            while len(data) < length:
                packet = conn.recv(min(self.buffer_size, length - len(data)))
                if not packet:
                    return None
                data.extend(packet)

            # 反序列化数据
            obj = pickle.loads(data)
            return obj
        except socket.timeout:
            print(f"[错误] 接收超时，耗时: {time.time() - start_time:.3f}s")
            return None
        except pickle.UnpicklingError as e:
            print(f"[错误] 反序列化失败: {e}")
            return None
        except socket.error as e:
            print(f"[错误] 接收失败: {e}")
            return None
        finally:
            conn.settimeout(None)

    def handle_client(self, conn, addr):
        """处理单个客户端的通信"""
        try:
            while True:
                data = self.receive(conn)
                if data is None:
                    print(f"[服务端] 客户端 {addr} 断开连接")
                    break
                print(f"[服务端] 收到来自 {addr} 的数据: {data}")
                response = {"response": "收到你的消息！", "original": data}
                self.send(response, conn)
        except Exception as e:
            print(f"[服务端] 客户端 {addr} 错误: {e}")
        finally:
            conn.close()
            self.clients = [(c, a) for c, a in self.clients if c != conn]
            print(f"[服务端] 客户端 {addr} 已关闭")

    def close(self):
        """关闭所有连接"""
        if self.conn and self.conn is not self.sock:
            self.conn.close()
        if self.is_server:
            for client, _ in self.clients:
                client.close()
        self.sock.close()


class AsyncSocketCommunicator(SocketCommunicator):

    def __init__(self, host="localhost", port=8888, is_server=False, buffer_size=4 * 4096, **kwargs):
        super().__init__(host, port, is_server, buffer_size, **kwargs)
        self.loop = asyncio.get_event_loop()
        self.reader = None
        self.writer = None

    async def async_connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        print("[客户端] 已连接服务端")

    async def async_send(self, obj):
        data = pickle.dumps(obj) + b"END"
        self.writer.write(data)
        await self.writer.drain()

    async def async_receive(self):
        data = b""
        while True:
            packet = await self.reader.read(self.buffer_size)
            if not packet:
                break
            data += packet
            if data.endswith(b"END"):
                break
        return pickle.loads(data[:-3])

    async def async_close(self):
        self.writer.close()
        await self.writer.wait_closed()
