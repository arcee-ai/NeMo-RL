"""
SPDX-License-Identifier: Apache-2.0

ctypes-based NCCL/RCCL wrapper (vendored from vLLM, lightly adapted).
"""
from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.distributed import ReduceOp

from .utils import find_nccl_library


ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p


ncclDataType_t = ctypes.c_int


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


ncclRedOp_t = ctypes.c_int


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.ncclSum
        if op == ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == ReduceOp.MAX:
            return cls.ncclMax
        if op == ReduceOp.MIN:
            return cls.ncclMin
        if op == ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class NCCLLibrary:
    exported_functions = [
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        Function("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        Function("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        Function(
            "ncclCommInitRank",
            ncclResult_t,
            [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int],
        ),
        Function(
            "ncclAllReduce",
            ncclResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t],
        ),
        Function(
            "ncclReduce",
            ncclResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t, ncclRedOp_t, ctypes.c_int, ncclComm_t, cudaStream_t],
        ),
        Function(
            "ncclAllGather",
            ncclResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t, ncclComm_t, cudaStream_t],
        ),
        Function(
            "ncclReduceScatter",
            ncclResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t],
        ),
        Function(
            "ncclSend",
            ncclResult_t,
            [buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int, ncclComm_t, cudaStream_t],
        ),
        Function(
            "ncclRecv",
            ncclResult_t,
            [buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int, ncclComm_t, cudaStream_t],
        ),
        Function(
            "ncclBroadcast",
            ncclResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int, ncclComm_t, cudaStream_t],
        ),
        Function("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
        Function("ncclGroupStart", ncclResult_t, []),
        Function("ncclGroupEnd", ncclResult_t, []),
    ]

    path_to_library_cache: dict[str, Any] = {}
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        so_file = so_file or find_nccl_library()
        try:
            if so_file not in NCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                NCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = NCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            # Mirror vLLM messaging but without logger dependency
            raise RuntimeError(
                f"Failed to load NCCL library from {so_file}. Platform={platform.platform()}"
            ) from e

        if so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs: dict[str, Any] = {}
            for func in NCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            NCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["ncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["ncclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId, rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(
            self._funcs["ncclCommInitRank"](ctypes.byref(comm), world_size, unique_id, rank)
        )
        return comm

    def ncclAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int, op: int, comm: ncclComm_t, stream: cudaStream_t) -> None:  # noqa: E501
        self.NCCL_CHECK(self._funcs["ncclAllReduce"](sendbuff, recvbuff, count, datatype, op, comm, stream))

    def ncclReduce(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int, op: int, root: int, comm: ncclComm_t, stream: cudaStream_t) -> None:  # noqa: E501
        self.NCCL_CHECK(self._funcs["ncclReduce"](sendbuff, recvbuff, count, datatype, op, root, comm, stream))

    def ncclReduceScatter(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int, op: int, comm: ncclComm_t, stream: cudaStream_t) -> None:  # noqa: E501
        self.NCCL_CHECK(self._funcs["ncclReduceScatter"](sendbuff, recvbuff, count, datatype, op, comm, stream))

    def ncclAllGather(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int, comm: ncclComm_t, stream: cudaStream_t) -> None:  # noqa: E501
        self.NCCL_CHECK(self._funcs["ncclAllGather"](sendbuff, recvbuff, count, datatype, comm, stream))

    def ncclSend(self, sendbuff: buffer_type, count: int, datatype: int, dest: int, comm: ncclComm_t, stream: cudaStream_t) -> None:  # noqa: E501
        self.NCCL_CHECK(self._funcs["ncclSend"](sendbuff, count, datatype, dest, comm, stream))

    def ncclRecv(self, recvbuff: buffer_type, count: int, datatype: int, src: int, comm: ncclComm_t, stream: cudaStream_t) -> None:  # noqa: E501
        self.NCCL_CHECK(self._funcs["ncclRecv"](recvbuff, count, datatype, src, comm, stream))

    def ncclBroadcast(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int, root: int, comm: ncclComm_t, stream: cudaStream_t) -> None:  # noqa: E501
        self.NCCL_CHECK(self._funcs["ncclBroadcast"](sendbuff, recvbuff, count, datatype, root, comm, stream))

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclCommDestroy"](comm))

    def ncclGroupStart(self) -> None:
        self.NCCL_CHECK(self._funcs["ncclGroupStart"]())

    def ncclGroupEnd(self) -> None:
        self.NCCL_CHECK(self._funcs["ncclGroupEnd"]())


__all__ = [
    "NCCLLibrary",
    "ncclDataTypeEnum",
    "ncclRedOpTypeEnum",
    "ncclUniqueId",
    "ncclComm_t",
    "cudaStream_t",
    "buffer_type",
]


