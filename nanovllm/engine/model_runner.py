import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        default_dtype = torch.get_default_dtype()
        # -----------------------------------------
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()  # 会跑一次最大形状的前向，让 cuDNN/CUDA 等完成选择/初始化。
        self.allocate_kv_cache()  # 根据显存余量计算能放下多少 KV blocks，并把缓存切片直接挂到每层模块的 k_cache/v_cache 上。
        if not self.enforce_eager:
            self.capture_cudagraph()  # 预先用不同 batch size 桶捕获多份 decode 前向的 CUDA Graph，后续重复回放。
        # -----------------------------------------
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2 ** 20)
                dist.barrier()  # 主进程创建共享内存后等待
            else:
                dist.barrier()  # 等待主进程创建好共享内存
                self.shm = SharedMemory(name="nanovllm")  # 子进程可以通过shm.name重新连接
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()  # 断开进程与共享内存的连接
            dist.barrier()  # 等待dist中进程组中的所有进程完成
            if self.rank == 0:
                self.shm.unlink()  # 真正释放共享内存，所有进程断开后才能回收
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        # 用 Python pickle + 共享内存 + 事件唤醒，开销极低，也不占用 NCCL 通道
        # 在多 GPU 分布式训练/推理时，通常会用 NCCL 来做张量的同步（如 all-reduce、broadcast 等）。
        # 这些操作走的通道本质上是 GPU-GPU 之间的高带宽通信路径。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # shm来当rpc用
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):  # 重置峰值统计，便于后面做 KV Cache 预算。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free  # cuda ctx + torch + 其他程序
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # PyTorch 分配器 从进程启动到目前为止的 峰值 使用量
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # PyTorch 当前真正在用 的显存（所有活着的 tensor + 少量簿记开销）

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 2 * L * block_size * num_kv_heads * head_dim * dtyp_size  k和v
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        # (total * a - used - (peak - current)) // block_size
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # [2, L, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size,
                                    num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # [num_blocks, block_size, num_kv_heads, head_dim]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 对齐成相同长度的二维表（缺省填 -1），转到 GPU
        # [B, max_blocks]
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)  # 前缀和
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue

            # slot_mapping: 对“尚未缓存”的块，按 block_table[i]*block_size + offset 计算精确写入地址。
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            # 当有prefix cache时候，新的q会短很多，而kv必须覆盖 旧+新，block table提供了旧的kv在哪个block的信息
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # prefill, eager或者 batch> 512(未capture) 的时候，普通前向
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 在model内部的layer前向过程中用到了get_context
            return self.model.compute_logits(self.model(input_ids, positions))
        else:  # 走Cuda graph replay
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]  # 找到刚好大于等于bs的graph_bs
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            #
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 仅rank0 sample
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # Capture不同batch size的cuda graph，推理时候直接用graph，减少kernel launch overhead
    @torch.inference_mode()
    def capture_cudagraph(self):
        # https://zhuanlan.zhihu.com/p/715863693
        # https://zhuanlan.zhihu.com/p/700224642
        # prefill 由于形状高度可变、不易稳定捕获，直接 eager 前向。
        # 1. capture之前，先让warmup跑一次，触发库资源的初始化
        # 2. capture的过程中，模型不会真正执行，而是生成一个静态图
        # 3. 计算图不支持动态shape
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # CUDA Graph 回放（replay）成功的两个硬性前提：捕获期与回放期在内存地址和张量形状/布局上必须一致。
        # 所以要预分配这些buffer
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))  # 预先创建一套桶化的 batch size
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):  # 先用大的batch捕获，分配最大的pool，之后的小batch可以共享这块pool
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            # graph.enable_debug_mode()
            with torch.cuda.graph(graph, self.graph_pool):  # 捕获前向
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            # graph.debug_dump("graph.dot")
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
