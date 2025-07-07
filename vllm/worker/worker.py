"""A GPU worker class."""
from typing import Dict, List, Tuple

import torch

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel, initialize_all_reduce_launcher)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata, SequenceOutputs
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import get_gpu_memory


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: int,
        distributed_init_method: str,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Initialize the distributed environment.
        _init_distributed_environment(parallel_config, rank,
                                      distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)

        # HBSEO 여기서 모델 로딩
        self.model = get_model(model_config)
        # 분산환경을 위해서 all_reduce를 launch하는듯?
        initialize_all_reduce_launcher(
            self.scheduler_config.max_num_batched_tokens,
            self.model_config.get_hidden_size(),
            self.model_config.dtype,
        )

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

    # vllm/engine/llm_engine.py의 _init_cache()에서 호출함.
    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seqs)

        # Execute the model.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache


    # HBSEO 여러 개의 seq_groups을 하나의 배치로 통합하여 모델 실행에 필요한 입력 데이터를 준비하는 역할
    # 아래의 모델에 전달될 항목에 대한 빈 리스트들을 초기화, 그리고 seq_group과 샘플링 파라미터를 저장할 seq_groups 리스트도 초기화해서 하나의 배치로 통합
    # - 입력 토큰(input_tokens), 
    # - 각 토큰의 위치(input_positions), 
    # - KV 캐시 슬롯 매핑 정보(slot_mapping), 
    # - 프롬프트 길이(prompt_lens), 
    # - 컨텍스트 길이(context_lens), 
    # - 블록 테이블(generation_block_tables) 등을

    # HBSEO 근데 매 step마다 이렇게 모두 초기화해서 계산한다. 즉, 기존 데이터에서 변경된 부분만 업데이트하는 증분(incremental)식이 아니다.
    # 이건 개선의 여지가 있을것 같다.
    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []

        # HBSEO 각 seq의 토큰 위치를 추적하는 리스트, 즉, 향후 positioning embedding을 위해 필요함.
        # 시퀀스 A: "Hello world"    → positions: [0, 1]
        # 시퀀스 B: "How are you?"   → positions: [0, 1, 2]
        # 배치 처리: [0, 1, 0, 1, 2] → 각 토큰이 해당 시퀀스 내 위치를 유지
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        # Add prompt tokens.
        # HBSEO Prompt 단계: "전체" prompt 토큰들을 수집
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            # HBSEO seq_group 에서 prefill 단계만 가져옴.
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            # HBSEO Prompt 토큰들을 입력 토큰 리스트에 추가
            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(prompt_tokens)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                # HBSEO prompt_len 만큼 0으로 채움
                slot_mapping.extend([0] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                # HBSEO 토큰 수를 block_size로 나누면 논리적인 block index가 나옴
                block_number = block_table[i // self.block_size]
                # HBSEO 거기서 offset 계산
                block_offset = i % self.block_size
                # 총 slot 갯수
                slot = block_number * self.block_size + block_offset
                # 최종적으로 해당 seq의 총 slot 개수를 append
                slot_mapping.append(slot)

        # Add generation tokens.
        # HBSEO decode 단계: 각 seq의 마지막 토큰만 처리
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            # HBSEO seq_group 에서 prefill이 아닌 decoding 단계만 가져옴.
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # decoding 단계이므로 prefill단계와 다르게 모든 seq_id를 확인함
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                # HBSEO 컨텍스트 길이에서 1을 뺀 값이 현재 토큰의 위치
                position = context_len - 1
                input_positions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        # HBSEO NVIDIA GPU의 Tensor Core는 행렬 곱셈과 같은 특정 연산을 고속으로 수행하는 데 특화된 유닛이다. 
        # Tensor Core는 특정 크기의 입력 데이터(일반적으로 8의 배수)에서 가장 효율적으로 작동하도록 설계되어 있다.
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        # HBSEO 입력 데이터를 GPU에서 처리할 수 있는 PyTorch 텐서 형식으로 변환하는 단계. 
        # 각 데이터의 특성에 맞는 적절한 데이터 타입(LongTensor, IntTensor)을 사용하고, 
        # 특히 block table은 효율적인 배치 처리를 위해 최대 길이에 맞춰 패딩된 후 텐서로 변환
        tokens_tensor = torch.cuda.LongTensor(input_tokens)
        positions_tensor = torch.cuda.LongTensor(input_positions)
        slot_mapping_tensor = torch.cuda.IntTensor(slot_mapping)
        context_lens_tensor = torch.cuda.IntTensor(context_lens)
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables
        ]
        block_tables_tensor = torch.cuda.IntTensor(padded_block_tables)

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
        )
        return tokens_tensor, positions_tensor, input_metadata


    # 스케줄러에 의해 어떤 블록이 Swap In/Out이 되고, 
    # 어떤 블록이 복제되어야 하는지 (blocks_to_copy), 
    # 모델에 포워딩되어야 하는 SequenceGroup이 무엇인지 결정되었지만,
    # 실제 GPU/CPU 메모리 간의 데이터 이동은 일어나지 않았음
    # CacheEngine 컴포넌트를 이용하여 실제로 데이터를 옮겨줌
    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> Dict[int, SequenceOutputs]:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        # HBSEO 모델에 전달될 항목에 대한 빈 리스트들을 초기화, seq_group과 샘플링 파라미터를 저장할 seq_groups 리스트도 초기화
        # HBSEO 근데 매 step마다 이렇게 모두 초기화해서 계산한다. 즉, 기존 데이터에서 변경된 부분만 업데이트하는 증분(incremental)식이 아니다.
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list)

        # Execute the model.
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: str,
) -> None:
    """Initialize the distributed environment."""
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=parallel_config.world_size,
        rank=rank,
        init_method=distributed_init_method,
    )
    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))
