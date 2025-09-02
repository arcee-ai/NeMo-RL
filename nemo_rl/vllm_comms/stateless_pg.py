# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import os
import pickle
import socket
import sys
import time
import uuid
from collections import deque
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, Optional

import torch
from torch.distributed import ProcessGroup, TCPStore
from torch.distributed.distributed_c10d import (Backend, PrefixStore,
                                                _get_default_timeout,
                                                _unregister_process_group)
from torch.distributed.rendezvous import rendezvous

from packaging import version
from packaging.version import Version
import importlib
import ipaddress

import logging

# We prefer to use os.sched_yield as it results in tighter polling loops,
# measured to be around 3e-7 seconds. However on earlier versions of Python
# os.sched_yield() does not release the GIL, so we fall back to time.sleep(0)
USE_SCHED_YIELD = ((sys.version_info[:3] >= (3, 11, 1))
                   or (sys.version_info[:2] == (3, 10)
                       and sys.version_info[2] >= 8))


def sched_yield():
    if USE_SCHED_YIELD:
        os.sched_yield()
    else:
        time.sleep(0)

def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False

def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"

def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version('torch')) >= Version(target)


# Helper function used in testing.
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    torch_version = version.parse(torch_version)
    return torch_version >= version.parse(target)

@dataclasses.dataclass
class StatelessProcessGroup:
    """A dataclass to hold a metadata store, and the rank, world_size of the
    group. Only use it to communicate metadata between processes.
    For data-plane communication, create NCCL-related objects.
    """
    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store

    # stores a reference to the socket so that the file descriptor stays alive
    socket: Optional[socket.socket]

    data_expiration_seconds: int = 3600  # 1 hour

    # dst rank -> counter
    send_dst_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    # src rank -> counter
    recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: dict[int, int] = dataclasses.field(
        default_factory=dict)

    # A deque to store the data entries, with key and timestamp.
    entries: deque[tuple[str,
                         float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self):
        assert self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {
            i: 0
            for i in range(self.world_size)
        }

    def send_obj(self, obj: Any, dst: int):
        """Send an object to a destination rank."""
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(obj))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.time()))

    def expire_data(self):
        """Expire data that is older than `data_expiration_seconds` seconds."""
        while self.entries:
            # check the oldest entry
            key, timestamp = self.entries[0]
            if time.time() - timestamp > self.data_expiration_seconds:
                self.store.delete_key(key)
                self.entries.popleft()
            else:
                break

    def recv_obj(self, src: int) -> Any:
        """Receive an object from a source rank."""
        obj = pickle.loads(
            self.store.get(
                f"send_to/{self.rank}/{self.recv_src_counter[src]}"))
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Optional[Any], src: int) -> Any:
        """Broadcast an object from a source rank to all other ranks.
        It does not clean up after all ranks have received the object.
        Use it for limited times, e.g., for initialization.
        """
        if self.rank == src:
            self.expire_data()
            key = (f"broadcast_from/{src}/"
                   f"{self.broadcast_send_counter}")
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.time()))
            return obj
        else:
            key = (f"broadcast_from/{src}/"
                   f"{self.broadcast_recv_src_counter[src]}")
            recv_obj = pickle.loads(self.store.get(key))
            self.broadcast_recv_src_counter[src] += 1
            return recv_obj

    def all_gather_obj(self, obj: Any) -> list[Any]:
        """All gather an object from all ranks."""
        gathered_objs = []
        for i in range(self.world_size):
            if i == self.rank:
                gathered_objs.append(obj)
                self.broadcast_obj(obj, src=self.rank)
            else:
                recv_obj = self.broadcast_obj(None, src=i)
                gathered_objs.append(recv_obj)
        return gathered_objs

    def barrier(self, timeout: float = 30.0):
        """A robust barrier to synchronize all ranks.


        Uses a multi-phase approach to ensure all processes reach the barrier
        before proceeding:

        1. Each process signals it has reached the barrier

        2. Each process signals that it has confirmed the arrival of all other
        ranks.

        3. Rank 0 waits for all other ranks to signal their departure to ensure
        that all ranks have departed the barrier first.

        Args:
            timeout: Maximum time in seconds to wait for each phase (in seconds)


        Raises:
            RuntimeError: If coordination fails or times out
        """
        # Generate a barrier ID that is globally unique
        try:
            if self.rank == 0:
                barrier_id = f"barrier_{uuid.uuid4()}"
                self.broadcast_obj(barrier_id, src=0)
            else:
                barrier_id = self.broadcast_obj(None, src=0)
        except Exception as e:
            raise RuntimeError("Failed to broadcast barrier_id") from e

        # Phase 1: Signal arrival at barrier
        # Wait for all processes to arrive
        # We need all ranks to confirm the arrival of all other ranks.
        # This is the key synchronization point.
        arrival_key = f"arrival_{barrier_id}_{self.rank}"
        try:
            self.store.set(arrival_key, b"1")
        except Exception as e:
            raise RuntimeError("Failed to signal barrier arrival") from e

        start_time = time.time()
        processes_arrived: set[int] = set()

        while len(processes_arrived) < self.world_size:
            # Check for timeout
            cur_time = time.time()
            if cur_time - start_time > timeout:
                raise RuntimeError("Barrier timed out after %f seconds",
                                   timeout)

            # Check for each process
            for i in range(self.world_size):
                if i in processes_arrived:
                    continue

                key = f"arrival_{barrier_id}_{i}"
                try:
                    # Try to get the key - if it exists, we'll get a value
                    # If it doesn't exist, it will throw an exception
                    self.store.get(key)
                    processes_arrived.add(i)
                except KeyError:
                    # Key doesn't exist yet
                    pass
                except Exception as check_e:
                    logging.debug("Error checking key existence: %s", check_e)
                    sched_yield()

            # Short sleep to avoid tight polling
            if len(processes_arrived) < self.world_size:
                sched_yield()

        # Phase 2: Signal departure from barrier
        # We only care to block at this stage in rank 0, which runs the
        # server side of the TCPStore. We want to make sure that all
        # clients have departed the barrier before rank 0 in case the
        # next thing after the barrier is a shutdown, including tearing
        # down the TCPStore. Other ranks can exit the barrier immediately
        # after signaling their departure.
        departure_key = f"departure_{barrier_id}_{self.rank}"
        try:
            self.store.set(departure_key, b"1")
        except Exception as e:
            raise RuntimeError("Failed to signal barrier departure") from e

        if self.rank != 0:
            return

        # Make rank 0 wait for all processes to signal departure
        start_time = time.time()
        processes_departed: set[int] = set()

        while len(processes_departed) < self.world_size:
            # Check for timeout
            if time.time() - start_time > timeout:
                raise RuntimeError("Barrier departure timed out after %f s",
                                   timeout)

            # Check for each process
            for i in range(self.world_size):
                if i in processes_departed:
                    continue

                key = f"departure_{barrier_id}_{i}"
                try:
                    # Try to get the key - if it exists, we'll get a value
                    # If it doesn't exist, it will throw an exception
                    self.store.get(key)
                    processes_departed.add(i)
                except KeyError:
                    # Key doesn't exist yet
                    pass
                except Exception as check_e:
                    logging.debug("Error checking key existence: %s", check_e)
                    sched_yield()

            # Short sleep to avoid tight polling
            if len(processes_departed) < self.world_size:
                sched_yield()

        # Clean up keys to avoid leaking memory in the store
        for i in range(self.world_size):
            try:
                self.store.delete_key(f"arrival_{barrier_id}_{i}")
            except Exception:
                logging.debug("Error deleting key: %s",
                             f'arrival_{barrier_id}_{i}')

            try:
                self.store.delete_key(f"departure_{barrier_id}_{i}")
            except Exception:
                logging.debug("Error deleting key: %s",
                             f'departure_{barrier_id}_{i}')

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
        store_timeout: int = 300,
    ) -> "StatelessProcessGroup":
        """A replacement for `torch.distributed.init_process_group` that does not
        pollute the global state.

        If we have process A and process B called `torch.distributed.init_process_group`
        to form a group, and then we want to form another group with process A, B, C,
        D, it is not possible in PyTorch, because process A and process B have already
        formed a group, and process C and process D cannot join that group. This
        function is a workaround for this issue.

        `torch.distributed.init_process_group` is a global call, while this function
        is a stateless call. It will return a `StatelessProcessGroup` object that can be
        used for exchanging metadata. With this function, process A and process B
        can call `StatelessProcessGroup.create` to form a group, and then process A, B,
        C, and D can call `StatelessProcessGroup.create` to form another group.
        """ # noqa
        launch_server = rank == 0
        if launch_server:
            # listen on the specified interface (instead of 0.0.0.0)
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
            use_libuv=False,  # for now: github.com/pytorch/pytorch/pull/150215
            master_listen_fd=listen_fd,
        )

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            socket=listen_socket,
            data_expiration_seconds=data_expiration_seconds)


def init_gloo_process_group(backend: Backend, prefix_store: PrefixStore,
                            group_rank: int, group_size: int,
                            timeout: timedelta) -> ProcessGroup:
    """
    Stateless init ProcessGroup with gloo backend compatible with 
    different torch versions.
    """
    if is_torch_equal_or_newer("2.6"):
        pg = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
    else:
        options = ProcessGroup.Options(backend=backend)
        pg = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
            options,
        )
    from torch.distributed.distributed_c10d import ProcessGroupGloo
    backend_class = ProcessGroupGloo(prefix_store,
                                     group_rank,
                                     group_size,
                                     timeout=timeout)
    backend_type = ProcessGroup.BackendType.GLOO
    device = torch.device("cpu")
    if is_torch_equal_or_newer("2.6"):
        # _set_default_backend is supported in torch >= 2.6
        pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()

    pg._register_backend(device, backend_type, backend_class)
    return pg