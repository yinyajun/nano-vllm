# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Manual Download

If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)


## `example.py`

Generating: 100%|████████████████████████████████████████████████████████████████████████████| 4/4 [00:15<00:00,  3.76s/it, Prefill=67tok/s, Decode=42tok/s]


Prompt: '<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n'
Completion: "<think>\nOkay, the user wants me to introduce myself. Let me start by recalling my capabilities. I can help with a wide range of tasks, from simple queries to more complex problems. I'm trained to provide accurate and helpful answers.\n\nI should mention my ability to understand different languages, as the user might need assistance in multiple languages. Also, I can assist with various types of tasks like writing, math, coding, and more. I should keep the tone friendly and approachable, making sure the introduction is clear and comprehensive.\n\nI need to make sure I cover the key points without being too verbose. Let me check if there's anything else I should include, like my availability or any specific skills. Since the user hasn't specified, I'll stick to the standard introduction. Alright, time to put it all together in a natural and friendly way.\n</think>\n\nHello! I'm a language model designed to help with a wide range of tasks. I can assist with writing, math, coding, and more. If you have any questions or need help with something, feel free to ask! 😊<|im_end|>"


Prompt: '<|im_start|>user\nlist all prime numbers within 20<|im_end|>\n<|im_start|>assistant\n'
Completion: "<think>\nOkay, so I need to list all the prime numbers between 20. Let me think. First, I remember that a prime number is a number greater than 1 that has no positive divisors other than 1 and itself. So, I need to check which numbers from 20 onwards are prime.\n\nStarting with 20. Let's see, 20 divided by 2 is 10, so 20 is not prime because it's even. Then 21. Hmm, 21 divided by 3 is 7, so 21 is 3 times 7, which means it's not prime. Next, 22. 22 is even, so it's not prime. 23... wait, what's 23? Let me check. Divided by 2, 3, 5, 7... 23 divided by 2 is 11.5, so no. 3? 23 divided by 3 is about 7.66. 5? 23 divided by 5 is 4.6. 7? 23 divided by 7 is around 3.28. So, 23 is prime. Okay, that's the first one.\n\nNext, 24. Divided by 2 is 12, so even. Not prime. 25. Ends with 5, so divisible by 5. Not prime. 26. Even again. Not prime. 27. Divided by 3 is 9. Not prime. 28. Even. Not prime. 29... Let me check. Divided by 2? No. 3? 29 divided by 3 is about 9.66. 5? No. 7? 29 divided by 7 is about 4.14. 11? 29 divided by 11 is about 2.63. So, 29 is prime.\n\nNext, 30. Even. Not prime. 31. Let me check. Divided by 2? No. 3? 31 divided by 3 is about 10.33. 5? No. 7? 31 divided by 7 is around 4.43. So, 31 is prime. Then 32."


Prompt: '<|im_start|>user\n为什么天空是蓝色的<|im_end|>\n<|im_start|>assistant\n'
Completion: '<think>\n好的，用户问为什么天空是蓝色的。首先，我需要回忆一下相关的科学知识。天空呈现蓝色主要是因为地球大气层中的分子和原子对可见光的吸收和散射。具体来说，大气中的氮气和氧气分子会吸收不同波长的光，而蓝色光因为波长较短，被散射得更充分，所以看起来是蓝色。\n\n接下来，用户可能对这个现象不太了解，或者需要更详细的解释。我应该从物理角度解释，比如光的散射现象，以及颜色选择的原理。同时，可能需要提到其他因素，比如太阳辐射的波长，以及大气中的其他成分对颜色的影响。\n\n还要考虑用户的潜在需求，他们可能对环境科学或天文学感兴趣，或者只是好奇。所以回答时要保持简洁明了，同时确保信息准确无误。需要确认科学原理是否正确，是否有其他解释方式，比如彩虹的形成，但这里重点在散射，所以可以适当提到但不要展开。\n\n最后，检查是否有遗漏的信息，比如是否需要提到云层的作用，但通常云层对颜色影响不大，而是散射导致。确保回答结构清晰，先总述，再分点解释，最后总结。\n</think>\n\n天空呈现蓝色的主要原因是**大气中分子和原子对可见光的散射现象**。具体来说，地球大气层中的氮气和氧气分子会在**可见光谱范围内**对不同波长的光（如红、橙、黄、绿等）进行**散射**。由于蓝光波长较短，其在大气中被散射得更充分，因此我们看到的天空颜色主要是**蓝色**。此外，太阳光在穿过大气层时，也会在不同角度散射出各种颜色，但整体上天空的蓝色是主要贡献。<|im_end|>'


Prompt: '<|im_start|>user\n10年前，我比我哥大2岁，十年后应该大几岁<|im_end|>\n<|im_start|>assistant\n'
Completion: '<think>\n嗯，用户问的是十年前他比他的哥哥大2岁，十年后应该比哥哥大几岁。首先，我需要确认时间线。十年前，他比哥哥大2岁，那么现在的时间是十年前加上10年，也就是十年后。所以，从十年前到现在的年龄差是10年，而他比哥哥大2岁。现在要算十年后的情况。\n\n用户可能想知道年龄差的变化。因为年龄差是一个恒定的数值，不管过了多少年，只要时间相同，年龄差就不会改变。所以十年后，年龄差还是2岁。不过，用户可能希望更详细的解释，或者有没有其他因素需要考虑，比如是否还有其他变化，但根据问题，应该不需要。\n\n可能用户没有意识到年龄差是恒定的，或者想确认这个逻辑。需要确保回答准确，同时用简单易懂的语言解释，避免复杂术语。可能需要举例说明，比如假设现在哥哥是30岁，十年后哥哥是40岁，而用户可能是18岁，所以十年后用户应该是58岁，年龄差还是2岁。这样用户就能明白道理了。\n</think>\n\n十年后，年龄差仍为2岁。  \n因为年龄差是恒定的，无论过了多少年，只要时间相同，差值不变。  \n\n**示例**：  \n- 去年（10年前）：你比哥哥大2岁  \n- 现在（10年后）：你比哥哥大2岁  \n- 未来（再过10年）：你仍比哥哥大2岁  \n\n因此，十年后你比哥哥大**2岁**。<|im_end|>'

## `bench.py`

Generating: 100%|████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.05s/it, Prefill=4tok/s, Decode=432tok/s]
Total: 133966tok, Time: 21.84s, Throughput: 6134.44tok/s
