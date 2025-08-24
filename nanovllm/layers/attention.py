import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# program_id：类似 CUDA 的 blockIdx，决定当前 kernel 负责的数据范围。
# arange：生成一个向量，表示当前线程块内的 index。
# load/store：读写 GPU 内存。
@triton.jit
def store_kvcache_kernel(
        key_ptr,
        key_stride,  # 因为 kernel 每个 program 处理一个 token，它需要知道 “到下一个 token 的内存偏移是多少”
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,  # 每个 token 的向量总长度
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    slot = tl.load(slot_mapping_ptr + idx)  # 每个 token 对应的 cache slot
    cache_offsets = slot * D + tl.arange(0, D)  # 一个 slot 容量是 D 个数（对应一个 token 的完整 key/value 向量）

    # 把新 token 的 k/v 向量存到 cache 对应的 slot
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor,
                  value: torch.Tensor,
                  k_cache: torch.Tensor,
                  v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # stride 检查保证张量内存布局符合预期（方便 Triton kernel 按连续内存读写），保证张量的内存布局和 Triton kernel 的访问模式匹配
    # stride[d] = 沿着维度 d 走一步时，在内存中要跨过多少个元素

    # 确保 head_dim 维度内存连续，能一口气读取
    assert key.stride(-1) == 1 and value.stride(-1) == 1  # 最后一维连续存放，如果不连续，Triton kernel tl.arange(0, D) 这种连续 load 就不成立
    # 在第1维（head）上走一步，要跨过 head_dim 个元素
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # k_cache 的形状应该是 (num_slots, D)
    # 要求在第1维走一步（下一个 slot），必须跨过 D 个元素
    # 每个 slot 就是一个完整的 [num_heads × head_dim] 连续块。
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 并行处理N个token
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
            self,
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            # 如果已经有缓存，新的 k/v 会被写到 cache 中
            # slot_mapping 决定写入的位置
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache

            # 高效计算 变长序列注意力，支持不同长度的 batch。
            # 传入的 cu_seqlens_q/k 是前缀和数组，用来告诉 FlashAttention 每个序列的范围。
            # causal=True 确保是因果注意力（只能看见前面的 token）。
            # block_table 用来做 block-sparse 注意力（高效内存利用）。
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:  # decode
            # 专门优化过的注意力实现，利用 KV 缓存避免重复计算。
            # 只需和历史 KV 做 attention，而不是重复算整个序列。
            # q: (batch_size, seqlen, num_heads, head_dim)
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
