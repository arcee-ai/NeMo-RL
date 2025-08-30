"""
SPDX-License-Identifier: Apache-2.0

Minimal vendored StatelessProcessGroup from vLLM for metadata exchange.
Only the parts needed by PyNcclCommunicator are included.
"""
from __future__ import annotations

import dataclasses
import pickle
import socket
import time
from collections import deque
from datetime import timedelta
from typing import Any, Optional

import torch
from torch.distributed import TCPStore


@dataclasses.dataclass
class StatelessProcessGroup:
    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store
    socket: Optional[socket.socket]
    data_expiration_seconds: int = 3600

    send_dst_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: dict[int, int] = dataclasses.field(
        default_factory=dict
    )
    entries: deque[tuple[str, float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self) -> None:
        assert 0 <= self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {i: 0 for i in range(self.world_size)}

    def expire_data(self) -> None:
        now = time.time()
        while self.entries:
            key, ts = self.entries[0]
            if now - ts > self.data_expiration_seconds:
                try:
                    self.store.delete_key(key)
                except Exception:
                    pass
                self.entries.popleft()
            else:
                break

    def send_obj(self, obj: Any, dst: int) -> None:
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(obj))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.time()))

    def recv_obj(self, src: int) -> Any:
        key = f"send_to/{self.rank}/{self.recv_src_counter[src]}"
        obj = pickle.loads(self.store.get(key))
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Optional[Any], src: int) -> Any:
        if self.rank == src:
            self.expire_data()
            key = f"broadcast_from/{src}/{self.broadcast_send_counter}"
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.time()))
            return obj
        key = f"broadcast_from/{src}/{self.broadcast_recv_src_counter[src]}"
        recv_obj = pickle.loads(self.store.get(key))
        self.broadcast_recv_src_counter[src] += 1
        return recv_obj

    def barrier(self, timeout: float = 30.0) -> None:
        # Simple barrier: each rank sets a key; rank 0 then waits for all keys.
        barrier_id = f"barrier_{int(time.time()*1000)}"
        self.store.set(f"arrival_{barrier_id}_{self.rank}", b"1")
        if self.rank != 0:
            return
        deadline = time.time() + timeout
        for i in range(self.world_size):
            while True:
                try:
                    self.store.get(f"arrival_{barrier_id}_{i}")
                    break
                except Exception:
                    if time.time() > deadline:
                        raise RuntimeError("StatelessProcessGroup barrier timed out")
                    time.sleep(0.001)

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
        store_timeout: int = 300,
    ) -> "StatelessProcessGroup":
        launch_server = rank == 0
        if launch_server:
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind((host, port))
            listen_socket.listen()
            listen_fd = listen_socket.fileno()
        else:
            listen_socket = None
            listen_fd = None

        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=launch_server,
            timeout=timedelta(seconds=store_timeout),
            use_libuv=False,
            master_listen_fd=listen_fd,
        )

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            socket=listen_socket,
            data_expiration_seconds=data_expiration_seconds,
        )


