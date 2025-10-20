# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from typing import Callable, Union

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.uniproc_executor import (  # noqa
    ExecutorWithExternalLauncher as ExecutorWithExternalLauncherV0)
from vllm.executor.uniproc_executor import (  # noqa
    UniProcExecutor as UniProcExecutorV0)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput

FailureCallback = Callable[[], None]


# distributed_executor_backend 가 ray, mp, uni, external_launcher 인 경우 각각 해당 실행기 클래스 반환

# 측면                    멀티프로세싱 (MP)                      Ray
# 클라이언트 선택	         동일한 로직                           동일한 로직
# 프로세스 관리             Python subprocess	                 Ray Actor
# 통신 방식                ZMQ + 직접 IPC                      ZMQ + Ray 내부 통신
# Pipeline Parallelism   지원 안함 (max_concurrent_batches=1)	지원함
# 최적화                  기본 Python 멀티프로세싱                Compiled DAG, SPMD Worker
# 확장성                  단일 노드만                            다중 노드 지원
# 장애 복구               기본적                                Ray의 고급 장애 복구


#Client ←→ ZMQ Socket ←→ Python Subprocess(or Ray Actor) (EngineCore)
#                     ←→ Python Subprocess(or Ray Actor) (Worker 0)
#                     ←→ Python Subprocess(or Ray Actor) (Worker 1)
#                     ←→ Python Subprocess(or Ray Actor) (Worker 2)

# Ray를 사용하든 멀티프로세싱을 사용하든 make_async_mp_client의 클라이언트 선택 로직은 완전히 동일합니다. 차이점은:
# 클라이언트 계층: 동일한 AsyncMPClient, DPAsyncMPClient, DPLBAsyncMPClient 사용
# Executor 계층: RayDistributedExecutor vs MultiprocExecutor 사용
# 성능과 기능: Ray가 더 많은 최적화와 기능 제공

class Executor(ExecutorBase):
    """
    Abstract class for v1 executors, mainly define some methods for v1.
    For methods shared by v0 and v1, define them in ExecutorBase"""

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = (
            parallel_config.distributed_executor_backend)
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, ExecutorBase):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"ExecutorBase. Got {distributed_executor_backend}.")
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            from vllm.v1.executor.ray_distributed_executor import (  # noqa
                RayDistributedExecutor)
            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher
        else:
            raise ValueError("Unknown distributed executor backend: "
                             f"{distributed_executor_backend}")
        return executor_class

    def initialize_from_config(self,
                               kv_cache_configs: list[KVCacheConfig]) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_from_config",
                            args=(kv_cache_configs, ))
        self.collective_rpc("compile_or_warm_up_model")

    def register_failure_callback(self, callback: FailureCallback):
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass

    def determine_available_memory(self) -> list[int]:  # in bytes
        output = self.collective_rpc("determine_available_memory")
        return output

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        output = self.collective_rpc("get_kv_cache_spec")
        return output

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        output = self.collective_rpc("execute_model",
                                     args=(scheduler_output, ))
        return output[0]

    @property
    def max_concurrent_batches(self) -> int:
        return 1

    def profile(self, is_start: bool = True):
        self.collective_rpc("profile", args=(is_start, ))


class UniProcExecutor(UniProcExecutorV0, Executor):
    pass


class ExecutorWithExternalLauncher(ExecutorWithExternalLauncherV0, Executor):

    def determine_available_memory(self) -> list[int]:  # in bytes
        # same as determine_num_available_blocks in v0,
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group
        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]
