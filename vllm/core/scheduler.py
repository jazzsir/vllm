import enum
import time
from typing import Dict, List, Optional, Tuple

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceOutputs,
                           SequenceStatus)

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class SchedulerOutputs:

    def __init__(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)

    def is_empty(self) -> bool:
        return (not self.blocks_to_swap_in and not self.blocks_to_swap_out
                and not self.blocks_to_copy)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        log_stats: bool,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.log_stats = log_stats

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
        )

        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

        self.last_logging_time: float = 0.0
        # List[timestamp, num_tokens]
        self.num_input_tokens: List[Tuple[float, int]] = []

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: str) -> None:
        for state_queue in [self.waiting, self.running, self.swapped]:
            for seq_group in state_queue:
                if seq_group.request_id == request_id:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.seqs:
                        if seq.is_finished():
                            continue
                        self.free_seq(seq, SequenceStatus.FINISHED_ABORTED)
                    return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule(
            self) -> Tuple[SchedulerOutputs, List[str], List[SequenceGroup]]:

        # Blocks that need to be swaped or copied before model execution.

        # HBSEO 실제 KV cache 이동은 여기서 안하고 관련 정보는 아래 Dict에 저장되고 LLMEngine에서 실제로 처리됨.
        # HBSEO Dict[gpu_block_id, cpu_block_id] 매핑정보를 담고 있음
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        ignored_seq_groups: List[SequenceGroup] = []

        # Fix the current time.
        now = time.time()

        # NOTE(woosuk): We prioritize the sequence groups in the RUNNING state
        # in order to minimize the preemption overheads.
        # Preemption happens only when there is no available slot to keep all
        # the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        # HBSEO 요청이 들어온 순서대로 정렬
        self.running = self.policy.sort_by_priority(now, self.running)


        # HBSEO seq_group내의 seqs의 상태가 어떤것은 waiting이고, 어떤것은 running일 수 없다.
        # seq_group은 논리적으로 동일한 요청에 속하는 하나 이상의 seq들을 묶어놓은 단위이다.
        # _preempt_by_swap()에 내가 작성한 주석 참고.

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            # HBSEO 이미 running 중인 seq_group인데 왜 다시 slot 을 append 하려고 하지?
            # 왜냐면, scheduler는 매 step마다(즉, batch에서 각 seq별 token을 생성할때마다) 호출되고,
            # 새로운 token을 생성하기 위해서 slot을 확보(append) 해야해기 때문임.

            # 여기서는 높은 우선순위의 seq_group이 slots을 할당 받을때(while) 까지 
            # 우선순위가 낮은 seq_group들을 preempt 해서 slots 확보
            while not self.block_manager.can_append_slot(seq_group):
                # Preempt the lowest-priority sequence groups.
                # HBSEO Preempt 전략은 아래와 같음.
                # 1. SequenceGroup 내에서 생성이 완료되지 않은 Sequence 개수를 구함
                # 2. 만약 이 값이 1이라면, 해당 시퀀스 그룹의 KVCache를 모두 Free 하고 RECOMPUTE 상태로 변경합니다.
                #    RECOMPUTE (속도 빠름): 다음에 해당 seq_group이 Running 상태로 변경될 때 KV 캐시가 재계산
                # 3. 만약 이 값이 1보다 크다면, 해당 시퀀스 그룹을 SWAP 상태로 변경.
                #    SWAP (속도 느림): 다음에 해당 seq_group이 Running 상태로 변경될 때 KV 캐시가 복원
                #    Swap Out이 예정된 그룹의 KV Cache 블록들은 blocks_to_swap_out(Dict[gpu_block_id, cpu_block_id]) 변수에 담김
                if self.running:
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            # HBSEO while의 조건으로 빠져나가면 else를 타고, break 로 빠져나가면 else를 안타게 됨.
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        # 이번 step에서 running 상태인 seq_group들로 업데이트 함
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        # HBSEO swap에 있는 seqs를 가능하다면 running 상태로 바꿔줌
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        # blocks_to_swap_out 조건은 방금 위에서 설정된 것이므로 "not blocks_to_swap_out" 조건을 추가함
        while self.swapped and not blocks_to_swap_out:
            # HBSEO swap에 있는 가장 높은 우선순위의 seq_group을 가져옴
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            # HBSEO 가장 높은 우선순위의 seq_group이 위에서 preempted 라면 더 이상 처리할 필요 없으니 break 함.
            if seq_group in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = sum(
                seq_group.num_seqs(status=SequenceStatus.RUNNING)
                for seq_group in self.running)
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break
            
            # HBSEO 위의 조건을 모두 통과했으면 swap in 이 가능하다는 것임
            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            self.running.append(seq_group)

        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        # Join waiting sequences if possible.
        # HBSEO waiting 상태(_preempt()에서 RECOMPUTE로 된 seq_group)의 seq_group을 처리하기 전에, 위에서 running 상태의 seq_group을 먼저 처리함
        # HBSEO 이유는 CPU 메모리 사용을 최대한 제한하기 위함
        # HBSEO waiting 큐에는 preempted seq_group이 앞에 추가되고, 새로운 seq_group이 뒤에 추가됨. 즉, 새로운 요청도 waiting 큐에 추가됨

        # HBSEO 새로 들어온 요청은 prefill을 해야 하고, 기존 요청이라면 decoding을 해야 함. 이걸 구분하기 위해서 기존에 있던 요청이라면 prompt_group_ids 리스트에 추가함.
        prompt_group_ids: List[str] = []

        # NOTE(woosuk): The sequence groups in the SWAPPED state are strictly
        # prioritized over the sequence groups in the WAITING state.
        # This is because we want to bound the amount of CPU memory taken by
        # the swapped sequence groups.
        if not self.swapped:
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.

            # HBSEO 여기서는 별도로 sort을 하지 않음.
            # HBSEO preempted seq_group은 waiting 큐 앞에 추가되고, 새로운 seq_group은 waiting 큐 뒤에 추가되므로 별도로 sorting하지 않음.
            while self.waiting:
                seq_group = self.waiting[0]
                # If the sequence group has been preempted in this step, stop.
                # HBSEO preempted 된 seq_group은 이번 step에서 바로 이전 코드에서 처리된것이므로 더 이상 확인할 필요 없음
                if seq_group in preempted:
                    break

                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens >= self.scheduler_config.max_seq_len:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        " and exceeds limit of "
                        f"{self.scheduler_config.max_seq_len}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    break

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                if (num_batched_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.num_seqs(
                    status=SequenceStatus.WAITING)
                num_curr_seqs = sum(
                    seq_group.num_seqs(status=SequenceStatus.RUNNING)
                    for seq_group in self.running)
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens

                # HBSEO 새로운 요청이라면 prefill해야 함으로, 여기에 추가함. 이건 scheduler() 에서 사용 예정
                # 다시 셜명하면, 여기는 RECOMPUTE이거나 새로운 요청이므로 prompt group 즉, prefill 단계임을 나타낸다.
                # 위에서 기존 running 상태의 seq_group은 decoding을 해야 하므로 prompt_group_ids에 추가하지 않음
                prompt_group_ids.append(seq_group.request_id)

        # HBSEO _scheduler()에서는 이렇게 mapping 정보만 만들고, KV Cache를 실제 메모리에 적용하는 부분은 LLMEngine에서 처리하는것 같다. 
        # HBSEO 이건 확인 필요함.
        scheduler_outputs = SchedulerOutputs(
            # HBSEO Swapped 상태인 그룹이 다시 Running 상태로 돌아와 프로세스를 진행해야 할 경우, Swap-in 시키기 위해 CPU↔GPU 간 블록 매핑 테이블
            blocks_to_swap_in=blocks_to_swap_in,

            # HBSEO Running 상태였던 그룹이 Swapped 상태로 변경되어 Swap-out 시키기 위한 GPU↔CPU 간 블록 테이블
            blocks_to_swap_out=blocks_to_swap_out,

            # HBSEO GPU 내에 있는 블록을 다른 위치로 복사할 블록 Pair 리스트.
            blocks_to_copy=blocks_to_copy,
        )

        # HBSEO 이후는 통계정보를 위한 코드임, 만약 log_stats 옵션이 설정안되어 있다면 Skip
        if not self.log_stats:
            return scheduler_outputs, prompt_group_ids, ignored_seq_groups

        # TODO(woosuk): Move the below code to the engine.
        now = time.time()
        if num_batched_tokens > 0:
            self.num_input_tokens.append((now, num_batched_tokens))
        elapsed_time = now - self.last_logging_time
        # HBSEO 경과된 시간이 미리 설정된 로깅 간격(_LOGGING_INTERVAL_SEC, 기본값 5초)보다 
        # 큰 경우에만 로깅을 수행. 이는 너무 빈번한 로깅으로 인한 오버헤드를 줄이기 위함.
        if elapsed_time > _LOGGING_INTERVAL_SEC:
            self.last_logging_time = now
            # HBSEO 현재 시간으로부터 _LOGGING_INTERVAL_SEC 이내에 기록된 데이터만 남기고, 
            # 오래된 데이터는 제거. 이는 이동 평균(moving average) 방식으로 처리량을 계산하기 위함.
            self.num_input_tokens = [(t, n) for t, n in self.num_input_tokens
                                     if now - t < _LOGGING_INTERVAL_SEC]
            # HBSEO 로깅을 위한 데이터 포인트가 최소 2개 이상인 경우에만 처리량 계산
            if len(self.num_input_tokens) > 1:
                # 리스트의 첫 번째 요소를 제외한 나머지 요소들의 토큰 개수를 모두 더함. 
                # 첫 번째 요소는 구간 시작점을 나타내므로 계산에서 제외
                total_num_tokens = sum(n
                                       for _, n in self.num_input_tokens[:-1])

                # 로깅 구간의 시작 시간(self.num_input_tokens의 첫 번째 요소의 시간)부터 현재 시간(now)까지의 
                # 시간을 계산하여 로깅 윈도우 크기를 구함.
                window = now - self.num_input_tokens[0][0]

                # 전체 처리된 토큰 개수를 로깅 윈도우 크기로 나누어 평균 처리량(tokens per second)을 계산.
                avg_throughput = total_num_tokens / window
            else:
                avg_throughput = 0.0

            # HBSEO GPU KV 캐시 사용량을 계산
            total_num_gpu_blocks = self.cache_config.num_gpu_blocks
            num_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
            num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
            gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

            # HBSEO CPU KV 캐시 사용량을 계산
            # CPU blocks은 config에서 0으로 주면 CPU KV 캐시를 사용하지 않을 것 같음.
            total_num_cpu_blocks = self.cache_config.num_cpu_blocks
            if total_num_cpu_blocks > 0:
                num_free_cpu_blocks = (
                    self.block_manager.get_num_free_cpu_blocks())
                num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
                cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
            else:
                cpu_cache_usage = 0.0

            logger.info(f"Throughput: {avg_throughput:.1f} tokens/s, "
                        f"Running: {len(self.running)} reqs, "
                        f"Swapped: {len(self.swapped)} reqs, "
                        f"Pending: {len(self.waiting)} reqs, "
                        f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                        f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        return scheduler_outputs, prompt_group_ids, ignored_seq_groups

    def schedule(
        self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs,
               List[SequenceGroup]]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        (scheduler_outputs, prompt_group_ids,
         ignored_seq_groups) = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []

        # HBSEO 이번 step에서 running 상태인 seq_group들에 대한 metadata를 생성함
        for seq_group in self.running:
            is_prompt = seq_group.request_id in prompt_group_ids

            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            # HBSEO sequenceGroup의 seqs 중에 running 상태인 seqs 만 가져옴
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # HBSEO logical block(logical_token_blocks)은 seq에 있음
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                # 각 프롬프트 별로 한 개의 고유한 아이디를 부여받는 request_id
                request_id=seq_group.request_id,

                # Waiting → Running으로 바뀐 그룹인지에 대한 여부. KV Cache가 없을 때는 프롬프트를 이용하여 KV Cache를 계산해야 하는데, 
                # 이 값을 이용하여 모델에 프롬프트 토큰까지 넣어줄지 말지를 결정
                # 즉, prefill(prompt)을 해야 하는지 아니면 decoding을 해야 하는지에 대한 여부
                is_prompt=is_prompt,

                # 그룹 내에 있는 모든 시퀀스들에 대하여, ID를 key로 하고 ID에 대응되는 토큰들을 value로 하는 매핑 테이블
                seq_data=seq_data,

                # 샘플링에 사용될 파라미터. top_k, top_p 등이 포함되어 있음.
                sampling_params=seq_group.sampling_params,

                # 해당 시퀀스 그룹 내 각 시퀀스들에 해당하는 Physical Block 리스트를 매핑 한 테이블.
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs, ignored_seq_groups

    def update(
        self,
        seq_outputs: Dict[int, SequenceOutputs],
    ) -> List[SequenceGroup]:
        # Update the running sequences and free blocks.
        for seq_group in self.running:
            # Process beam search results before processing the new tokens.
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                output = seq_outputs[seq.seq_id]
                if seq.seq_id != output.parent_seq_id:
                    # The sequence is a fork of the parent sequence (beam
                    # search). Free the current sequence.
                    self.block_manager.free(seq)
                    # Fork the parent sequence.
                    parent_seq = seq_group.find(output.parent_seq_id)
                    parent_seq.fork(seq)
                    self.block_manager.fork(parent_seq, seq)

            # Process the new tokens.
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Append a new token to the sequence.
                output = seq_outputs[seq.seq_id]
                seq.append_token_id(output.output_token, output.logprobs)
        # Return a shallow copy of the running queue to prevent the queue
        # from being modified by the caller.
        return self.running.copy()

    def free_seq(self, seq: Sequence, finish_status: SequenceStatus) -> None:
        seq.status = finish_status
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not supported. In such a case,
        # we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            # HBSEO RECOMPUTE가 SWAP보다 비용이 저렴하다.
            # seqs가 한개이면 RECOMPUTE하고 그 이상이면 SWAP
            if len(seqs) == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, "Invalid preemption mode."

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            # seq의 KVCache를 모두 Free 하고 Waiting 상태로 변경
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    # HBSEO seq_group내의 seqs의 상태가 어떤것은 waiting이고, 어떤것은 running일 수 없다!!
    # HBSEO seq_group은 논리적으로 동일한 요청에 속하는 하나 이상의 seq들을 묶어놓은 단위이다.
    # 아래 코드에서 seq_group.get_seqs(status=SequenceStatus.RUNNING)는 해당 seq_group 내에서 현재 RUNNING 상태인 모든 Sequence를 가져오고,
    # for seq in seqs: 루프를 돌면서 각 seq의 상태를 SequenceStatus.SWAPPED로 변경한다.
    # 이는 seq_group이 선점될 때, 해당 그룹에 속한 모든 RUNNING 상태의 seq가 함께 SWAPPED 상태로 전환됨을 의미.
    # _allocate() 에서도 seq_group내의 모든 seq들을 한꺼번에 처리함
    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        # CPU 메모리에 대한 free block size 확인
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        # swap out = GPU block -> CPU block
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
