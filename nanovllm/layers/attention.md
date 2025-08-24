## 1. 背景：为什么需要 `block_table`

在大模型的推理（特别是 **长上下文 / 多 batch 拼接** 的时候），直接存放 KV cache 的方式会有几个问题：

- **内存浪费**：如果每个序列长度不一样，那要 padding 到相同长度，很多位置其实没用到。
- **访存效率低**：不同样本的 KV cache 在内存里可能是碎片化的，做 attention 时很难高效地读。
- **扩展困难**：推理过程中，batch 会动态变化（有的序列结束了，有的新序列进来），必须有一种灵活的“映射表”来管理。

于是引入了 **block-sparse KV cache** 的概念：

- 把 KV cache 组织成 **固定大小的 block（比如 16 或 32 个 token 一块）**。
- 每个序列对应哪些 block，不用连续存，而是通过一个表来查。
- 这样就能像查“索引表”一样，快速定位到对应的 KV 内存块。

------

## 2. `block_table` 的含义

在这个实现里，`block_table` 就是那个 **索引表**：

- 它是一个 **二维矩阵**，形状大概是：

  ```
  (batch_size, max_blocks_per_sequence)
  ```

- **每一行对应一个序列**，存放它的 KV cache 使用了哪些 block。

- 值是 **block 在全局 KV 缓存里的索引**。

  - 比如 `block_table[i, j] = b` 就表示：
     序列 i 的第 j 个 block 存在全局 KV cache 的第 b 号位置。

这样 Attention 的时候，不是顺序扫一整个 KV cache，而是先查 `block_table` 再去对应 block 里取数据。

------

## 3. 在 Prefill 阶段

```
if context.block_tables is not None:  # prefix cache
    k, v = k_cache, v_cache
```

- 当有 **prefix cache**（提示词已经预先存过 KV），就直接把 `k, v` 指向全局缓存。
- 后续调用 `flash_attn_varlen_func` 时，传入 `block_table`，它会知道“每个 query token 应该对哪些 KV block 计算注意力”。

------

## 4. 在 Decode 阶段

```
o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                            cache_seqlens=context.context_lens,
                            block_table=context.block_tables,
                            softmax_scale=self.scale, causal=True)
```

- 这时每次只解一个新 token。
- 它会根据 `block_table` 去找 **该序列对应的历史 KV block**，而不是遍历整个缓存。
- 从而：
  - 减少不必要的访存（只访问该序列的历史 block）。
  - 多 batch 拼接时，每个样本只用自己的 `block_table` 就能正确检索。

------

## 5. 总结成一句

**`block_table` 就是“序列 → KV 缓存 block 的索引映射表”。**
 它让 FlashAttention 能在 **block-sparse KV cache** 的场景下快速、灵活地找到每个序列的历史 KV，从而支持：

- 高效内存复用
- 动态 batch 管理
- 长上下文扩展