##  模型结构

- **architectures**: `"Qwen3ForCausalLM"`
   表明这是一个用于自回归语言建模（Causal LM）的 Qwen3 模型。
- **model_type**: `"qwen3"`
   指定模型类别，和 Hugging Face Transformers 框架兼容。
- **hidden_size**: `1024`
   每层 Transformer 的隐藏维度大小。
- **intermediate_size**: `3072`
   前馈网络（FFN）的中间层维度，通常是 `hidden_size` 的 3 倍。
- **num_hidden_layers**: `28`
   模型有 28 层 Transformer block。
- **num_attention_heads**: `16`
   每层有 16 个注意力头。
- **head_dim**: `128`
   每个注意力头的维度，所以注意力总维度 = 16 × 128 = 2048（这里通常会等于 hidden_size，但 Qwen3 可能用了投影）。
- **num_key_value_heads**: `8`
   采用了 **分组 KV Cache**（Grouped Query Attention, GQA），即 16 query 头共享 8 个 key/value 头，降低推理显存占用。

------

## 🧠 注意力与位置编码

- **attention_bias**: `false`
   不使用 attention bias。
- **attention_dropout**: `0.0`
   注意力层不做 dropout，保证推理稳定性。
- **max_position_embeddings**: `40960`
   支持最长 40,960 token 的序列。
- **rope_theta**: `1000000`
   旋转位置编码（RoPE）的基数，值越大代表更长上下文建模能力。
- **rope_scaling**: `null`
   没有额外的 RoPE scaling，直接使用上面的 θ。
- **sliding_window**: `null`
   **use_sliding_window**: `false`
   没有采用滑动窗口注意力（即全上下文可见）。
- **max_window_layers**: `28`
   表明所有 28 层都支持长上下文（可能用于配置分层窗口注意力时生效）。

------

## 🔢 归一化与激活

- **hidden_act**: `"silu"`
   前馈网络激活函数为 SiLU (Swish)。
- **rms_norm_eps**: `1e-06`
   RMSNorm 的数值稳定性参数 ε。

------

## 🧩 词表与嵌入

- **vocab_size**: `151936`
   词表大小，接近 15 万。
- **tie_word_embeddings**: `true`
   输入和输出词嵌入权重共享，减少参数量。

------

## ⚙️ 训练与推理相关

- **initializer_range**: `0.02`
   参数初始化范围（正态分布）。
- **torch_dtype**: `"bfloat16"`
   训练/推理时使用 bfloat16 精度。
- **use_cache**: `true`
   生成时使用 KV 缓存，加速推理。
- **transformers_version**: `"4.51.0"`

[//]: # (   表明这是在 Hugging Face Transformers 4.51.0 版本下导出的配置。)

-------

```shell
Qwen3Model(
  (embed_tokens): Embedding(151936, 1024)
  (layers): ModuleList(
    (0-27): 28 x Qwen3DecoderLayer(
      (self_attn): Qwen3Attention(
        (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
        (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
        (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
        (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
        (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
        (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
      )
      (mlp): Qwen3MLP(
        (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
        (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
        (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
    )
  )
  (norm): Qwen3RMSNorm((1024,), eps=1e-06)
  (rotary_emb): Qwen3RotaryEmbedding()
)
```
---
## 🔎 模块解读

1. **嵌入层**

   ```
   (embed_tokens): Embedding(151936, 1024)
   ```

   - 词表大小 = 151,936
   - 维度 = 1024
   - 参数量 = `151936 × 1024 ≈ 155M`

2. **Decoder 层（共 28 层）**
    每一层包含：

   - **Self-Attention**
     - q_proj: `1024 → 2048`
     - k_proj: `1024 → 1024`
     - v_proj: `1024 → 1024`
     - o_proj: `2048 → 1024`
        ⇒ 参数量大约 `2.048M + 1.048M + 1.048M + 2.097M ≈ 6.2M`
   - **MLP**
     - gate_proj: `1024 → 3072`
     - up_proj: `1024 → 3072`
     - down_proj: `3072 → 1024`
        ⇒ 参数量大约 `3.145M + 3.145M + 3.145M ≈ 9.4M`
   - **RMSNorm**
      基本上是 `O(hidden_size)` 的参数量（1k 量级，可以忽略不计）。

   🔹 所以单层参数 ≈ `6.2M + 9.4M ≈ 15.6M`

   🔹 28 层总参数 ≈ `28 × 15.6M ≈ 436M`

3. **最终 LayerNorm**

   ```
   (norm): Qwen3RMSNorm((1024,))
   ```

   - 只有 1024 个可训练参数，可忽略。

------

## 📊 总参数估算

- **Embedding**: ~155M
- **28 层 Decoder**: ~436M
- **输出层**（tie embedding → 共享输入嵌入权重，不单独增加参数）

👉 总计大约 **591M 参数**，也就是 **6 亿规模模型**。

------

## ✅ 总结

这份 Qwen3 模型大致规模：

- **28 层，Hidden size = 1024，FFN = 3072，16 头注意力（GQA）**
- **总参数量 ≈ 6 亿**

---

## 🔹 基本设定 (Qwen3 配置)

- `hidden_size = 1024`
- `head_dim = 128`
- `num_attention_heads = 16` （Q 头数）
- `num_key_value_heads = 8` （KV 头数）

所以：

- Q: `[B, L, 2048] → reshape → [B, L, 16, 128]`
- K: `[B, L, 1024] → reshape → [B, L, 8, 128]`
- V: `[B, L, 1024] → reshape → [B, L, 8, 128]`

------

## 🔹 Attention 矩阵计算

1. **标准多头注意力 (对称情况)**
   $$
   \text{scores} = Q \cdot K^T
   $$

   - Q: `[B, 16, L, 128]`
   - K: `[B, 16, L, 128]`
   - scores: `[B, 16, L, L]`

2. **GQA 的情况 (Q 多，KV 少)**

   - Q: `[B, 16, L, 128]`
   - K: `[B, 8, L, 128]`
   - **映射规则**：16 个 Q 头映射到 8 个 KV 头（两个 Q 头共享一个 KV 组）
     - 例如 Q[0], Q[1] 都用 K[0]
     - Q[2], Q[3] 都用 K[1]
     - …以此类推
   - scores: `[B, 16, L, L]` （因为每个 Q head 都能和对应的 K 组做乘积）

3. **Attention 应用到 V**

   - Softmax(scores): `[B, 16, L, L]`
   - V: `[B, 8, L, 128]`
   - 映射同样共享规则（2 个 Q 头对应同一个 V 头）
   - 输出: `[B, 16, L, 128]`

4. **拼接 + 输出投影**

   - 拼接 → `[B, L, 16 × 128] = [B, L, 2048]`
   - o_proj → `[B, L, 1024]`

------

## ✅ 总结：Attention 矩阵维度

- **scores (QK^T)**: `[B, 16, L, L]`
- **attention × V**: `[B, 16, L, 128]`
- **拼接后**: `[B, L, 2048]`
- **投影回 hidden_size**: `[B, L, 1024]`