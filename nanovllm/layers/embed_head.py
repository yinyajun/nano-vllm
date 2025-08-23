import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context

# 沿vocab维做tensor parallel
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0

        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0) # num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)  # 在第 0 维（行）上截取 [start_idx : start_idx + shard_size]
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
            # 对于属于本 rank 的 token：得到局部索引 x - start
            # 对于不属于本 rank 的 token：mask=False，结果强制成 0
        y = F.embedding(x, self.weight) # 即使 token 不属于本 rank，也会查表，因为它们在上一步被改成了索引 0

        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            # 属于本 rank 的 token → 保留
            # 不属于本 rank 的 token → 乘 0，变成全零向量
            dist.all_reduce(y) # 各个 rank 把自己的结果向量按位置相加。
            # 由于同一个 token 只会在“正确的 rank”留下非零，其它 rank 都是零，所以相加后得到的就是全局正确 embedding。
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # 例：cu=[0,4,7,10] → cu[1:]-1=[3,6,9]，正好是每条样本末 token 的位置。
            last_indices = context.cu_seqlens_q[1:] - 1 # refill 阶段一般只需要拿到每个序列最后位置的 logits。
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias) # [B, d] * [d, V_shard] = [B, V_shard]
        if self.tp_size > 1:
            # 只有 rank 0 预先分配一个 list 来接收来自所有 rank的分片 logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # 使用分布式 gather 把每个 rank 的 logits 发送到 dst=0（rank 0）
            # rank 0：all_logits[i] 会被填入来自 rank i 的分片 [*, vocab_shard]
            # 其他rank：作为发送端，不返回聚合结果
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
            # rank 0：把 tp_size 份 [*, vocab_shard] 沿最后一维拼起来，得到完整 logits：
            # 形状变为 [*, vocab_shard * tp_size]，也就是 [*, num_embeddings]
            # 非 0 号 rank：返回 None（因为手里没有全量 logits）
        return logits
