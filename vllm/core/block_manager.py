"""A block manager that manages token blocks."""
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

# 특정 장치 (GPU 또는 CPU)에 대한 물리적 토큰 블록의 할당 및 해제를 관리
class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size)
            self.free_blocks.append(block)

    # TODO 실제 CUDA 레벨에서 추가된것은 아닌것 같다. 향후 어디서 추가되는지 확인 필요
    def allocate(self) -> PhysicalTokenBlock:
        # free block이 없으면 오류
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        # free block에서 하나 꺼내서 ref_count를 1로 설정
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]

# 논리적 토큰 블록과 물리적 토큰 블록 간의 매핑을 관리. seq의 토큰을 저장하기 위한 물리적 블록을 할당/해제하며, 스왑 인/아웃 기능을 처리
class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,

        # HBSEO watermark는 GPU 메모리 사용 상항선을 정해 놓고, 그 이상의 요청은 받지 않도록 하는것.
        # 기존 시퀀스의 길이가 길어지면 GPU 메모리 사용량이 늘어나고, 이보다 우선순위가 낮은 seq는 swap-out이 일어난다.
        # 메모리를 너무 타이트 하게 사용하면 이런 swap-out이 빈번하게 일어 날 수 있으므로,
        # 메모리 사용 상한선을 정해 놓는것임.
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        # 여기서 BlockAllocator 로 CPU, GPU에 대한 물리적 블록을 관리.
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        # HBSEO seq 별로 block_table을 관리하는것 같다.
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs()[0]
        num_required_blocks = len(seq.logical_token_blocks)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        # Use watermark to avoid frequent cache eviction.
        return (num_free_gpu_blocks - num_required_blocks >=
                self.watermark_blocks)

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs()[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        block_table: BlockTable = []
        for _ in range(len(seq.logical_token_blocks)):
            block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.

            # HBSEO seq_group에서 같은 block을 공유하는 seqs도 있고 아닌 seqs도 있을것인데 
            # 왜 ref_count를 전체 num_seqs의 갯수로 주는지 이해가 안됐다.
            # 이유는, scheduler에서 self.waiting 큐에 있는 요청을 처음 할당할때 이 함수가 호출된다.
            # 즉, 처음 tokens을 생성할때는 block을 모두 공유할 수 있기 때문에 ref_count를 num_seqs로 주는것 같다
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.

        # HBSEO 정확히 하려면 seq_group의 seq들이 여러 block을 사용할 수 있을것이고, 이 여러 block에서 할당할 수 있는 slot이 남아 있는지 확인해야 한다.
        # 이렇게 안하고 seq개수만큼 새로운 free blocks 있는지만 확인하는 이유 최악의 시나리오 가정해서 빠르게 확인하기 위함인것 같다인것 같다
        # 실제로 slopt을 append할때는 append_slot()에서 확인한다.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    # 시퀀스가 새로운 논리적 블록을 필요로 하면 새로운 물리적 블록을 할당하고 블록 테이블에 추가
    # 마지막 물리적 블록이 다른 시퀀스와 공유되지 않으면 해당 블록에 토큰을 추가할 수 있음.
    # 마지막 물리적 블록이 공유되면 Copy-on-Write 메커니즘을 사용하여 새로운 블록을 할당하고 
    # 기존 블록의 내용을 복사한 후 기존 블록을 해제. 새로운 블록의 번호를 반환.
    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        # HBSEO 아래와 같은 구조로 되어 있다. 여기서 physical blocks에 넣기 위한 앞의 두가지를 가져오는것.
        # "Logical KV cache blocks" <-> "Block tabels" <-> "Physical KV cache blocks"
        logical_blocks = seq.logical_token_blocks
        # block_table은 seq 별로 관리하는구나!!
        block_table = self.block_tables[seq.seq_id]

        # 새로운 logical block이 추가되면 pyhsical block도 추가해야 한다.
        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            # TODO 실제 CUDA 레벨에서 추가된것은 아닌것 같다. 향후 어디서 추가되는지 확인 필요
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # HBSEO 여기서는 block이 같은 seq_group 내의 seq들에 의해서만 공유된다.
        # HBSEO 다른 seq_group과는 block을 공유하지 않는다.
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                blocks.add(block)
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            # HBSEO TODO seq_id별로 block_tables이 존재하는것 같다. 향후 block_table 분석할 때 확인 필요.
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                # HBSEO 같은 seq_group 내에서는 seq들의 출력 토큰에 대한 block이 같을 경우 해당 블럭을 공유한다.
                # HBSEO 그래서 같은 block이면 아래처럼 allocate()하지 않고 ref_count만 증가시킨다.
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    # ref_count: 해당 block을 해제할때 ref_count가 0이 돼야 free 가능하다.
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    # HBSEO TODO 현재 cpu_block에 저정된 KV Cache를 gpu_block에 복사하는 로직은 LLMEngine에서 할것 같은데, 확인 필요.
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    # 여기엔 CPU 에 대한 free blocks 개수 리턴
    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:

                # HBSEO 같은 seq_group 내에서는 seq들의 출력 토큰에 대한 block이 같을 경우 해당 블럭을 공유한다.
                # HBSEO 그래서 같은 block이면 아래처럼 allocate()하지 않고 ref_count만 증가시킨다.
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    # ref_count: 해당 block을 해제할때 ref_count가 0이 돼야 free 가능하다.
                    cpu_block.ref_count += 1

                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                # HBSEO 여기서 GPU KV Cache free 하네
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in block_table:
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()
