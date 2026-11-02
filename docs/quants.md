# Choosing a quant

[Back to the docs](README.md)

Which checkpoint of a model to run comes down to three questions, in this
order:

1. **Does it fit** your Mac's memory with the context you need?
2. **How close to bf16** does it stay?
3. **How fast** is it?

Each model below has two parts: closeness to bf16, one table per model, and
speed, one section per Mac that measured it.

Measured so far: [Qwen3.8-27B](#qwen38-27b). For other models, the
[model table](../README.md#models) lists the usual checkpoints.

## 1. Does it fit

Splash sizes memory and context when it starts: the weights first, then as
much context as its Metal budget holds beside them, up to the model's native
window ([memory and context](server.md#memory-and-context)). `--max-context`
asks for a size, and a size the budget cannot hold stops startup with the most
it allows, so a start with the context you need answers the question.
`--max-memory` lowers the budget to leave room for other applications,
`--language-only` leaves out the vision tower and frees memory for context, and
`--max-cache-disk 0` gives back the SSD cache's staging. For 24 GB Macs, see
[smaller GGUFs](performance.md#smaller-ggufs-on-24-gb-macs).

## 2. How close to bf16

Two numbers, both against the bf16 model on the same text:

- **KL**: KL(bf16 || quant) per position, averaged. Lower is closer.
- **EAR@10**: the overlap of the two distributions over bf16's 10 most likely
  tokens, both renormalized to those 10. 100 means identical there.

## 3. How fast

Speed is reported per Mac, with that Mac's method, against the model's MLX
4-bit on the same Mac where it was measured.

## Qwen3.8-27B

### Which one

- **Default: [`sady4850/Qwen3.8-27B-4bit-GPTQ`](https://huggingface.co/sady4850/Qwen3.8-27B-4bit-GPTQ).**
  Same layout, size and speed as `mlx-community/Qwen3.8-27B-4bit`, only
  GPTQ-rounded: KL to bf16 29% lower on wiki, 47% lower on chat.
- **More quality for 1.3 GB more: `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`.**
  Closest of the 4-bit files. Decode about 11% slower on the M1 Pro, 5-7%
  fewer speculative steps per second on the M2 Max.
- **Closest overall: `unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_S`** (or the SLQ EAR98
  file, about equal at about the same size). About 3.5 GB more than the MLX
  4-bit and about 20% slower on the M1 Pro.
- **Short on memory: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S`.** 3.4 GB
  smaller, about as close to bf16 as the MLX 4-bit, but slower on both Macs.

On the M1 Pro (32 GB) the MLX 4-bit serves with
`--language-only --max-context 64K --max-memory 21G`.

```bash
splash serve --model sady4850/Qwen3.8-27B-4bit-GPTQ
splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

### Closeness to bf16

Measured on an M1 Pro (32 GB) with 1.4.0-m1, text only. Size is the text
weights in GB (10^9 bytes).

| Quant | Size | KL wiki | KL chat | EAR@10 wiki | EAR@10 chat |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/Qwen3.8-27B-4bit` | 15.13 | 0.0562 | 0.1906 | 92.69 | 88.66 |
| `sady4850/Qwen3.8-27B-4bit-GPTQ` | 15.13 | 0.0399 | 0.1002 | 94.36 | 92.22 |
| `unsloth/Qwen3.8-27B-GGUF:Q4_0` | 16.06 | 0.0325 | 0.1412 | 94.64 | 91.26 |
| `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | 16.46 | 0.0144 | 0.0831 | 97.18 | 93.90 |
| `unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_S` | 18.67 | 0.0065 | 0.0574 | 97.92 | 95.24 |
| `kaitchup/Qwen3.8-27B-SLQ-GGUF:EAR98-Q5_K_S` | 18.88 | 0.0104 | 0.0486 | 97.97 | 95.57 |
| `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S` | 11.77 | 0.0573 | 0.2121 | 92.54 | 87.64 |

- Every quant but the GSQ IQ3_S differs from the MLX 4-bit in KL on both texts
  (95% paired bootstrap over windows). The GSQ IQ3_S does not, on either.
- The wiki mean depends on one token in one window. Without that window:
  MLX 4-bit 0.0436, GPTQ 0.0280, Q4_0 0.0264, UD-Q4_K_M 0.0075, UD-Q5_K_S
  0.0036, SLQ EAR98 0.0036, GSQ IQ3_S 0.0500.
- The MLX 4-bit and the GSQ IQ3_S score below bf16's chat perplexity.
- On 1.5.0-m1 the GSQ IQ3_S's top-512 log-probabilities came out bit-identical
  to 1.4.0-m1 on 6 windows; the MLX 4-bit's KL to bf16 matched to 4 decimals.
- Independent agent check: Alexander Fedora replayed 78 recorded agent turns
  against Q8_0 on an M2 Max; the share of turns with the same tool call was
  MLX 4-bit 31%, GPTQ 44%, Q4_0 44%, UD-Q4_K_M 51%, UD-Q5_K_M 56%, GSQ IQ3_S
  44% ([discussion #34](https://github.com/paperniuk/splash/discussions/34#discussioncomment-19226461)).

<details>
<summary>How it was measured</summary>

- **Texts.** wiki: wikitext-2 test. chat: the first 300 conversations of
  UltraChat test as plain text. Each is tokenized with the model's tokenizer
  and cut into 30 chunks of 2,049 token ids.
- **Quant side.** `make teacher-logits` ([local benchmarks](validate.md#local-benchmarks))
  on each chunk with `--window 2048:1016`: a cold prefill of the first 1,032
  tokens, then 1,016 teacher-forced positions through the production verify
  path, INT8 KV cache.
- **Reference.** bf16 `Qwen/Qwen3.8-27B` in PyTorch on a GPU, the same token
  ids, top-512 log-probabilities at 128 positions per window. KL is exact over
  the top 512 and bounds the rest by the tail mass; against the full vocabulary
  this stays within 2% of the mean.
- **Intervals.** Paired bootstrap over the 30 windows of each text, against the
  MLX 4-bit.

</details>

Limits:

- The texts are prompt text, not the model's own answers or agent sessions.
  KL to bf16 measures closeness, not task success.
- `sady4850/Qwen3.8-27B-4bit-GPTQ` is the author's checkpoint. Part of its
  calibration is UltraChat train; the chat text is UltraChat test.

### Speed: M1 Pro, 32 GB

Decode time of the teacher-forced windows above against the MLX 4-bit,
positive is slower. Median seconds per window over 60 windows, one request at
short context, 1.4.0-m1, a separate session per quant.

| Quant | Decode time |
| --- | ---: |
| MLX 4-bit | base |
| GPTQ | -0.2% |
| Q4_0 | +5.9% |
| UD-Q4_K_M | +11.1% |
| UD-Q5_K_S | +20.2% |
| SLQ EAR98 | +19.7% |
| GSQ IQ3_S | +18.2% |

- The GSQ IQ3_S and MLX 4-bit were re-measured on 1.5.0-m1 in one session,
  alternating, 6 windows: the GSQ decoded 8% faster than on 1.4.0-m1 (+27.5%
  there), the MLX 4-bit stayed within noise. The other rows are 1.4.0-m1.

### Speed: M2 Max, 64 GB

Measured by Alexander Fedora on an M2 Max (38-core GPU, 64 GB) with 1.5.0-m1,
on his own requests grouped by input length (32K: up to 32,768 tokens).
Requests per group: 5 at 32K, 74 at 64K, 38 at 96K, 40 at 128K.

MLX 4-bit: decode 48.5 tok/s and prefill 213 tok/s at 32K, 39.4 and 100.5 at
128K.

Decode tok/s depends on how many draft tokens the text keeps per speculative
step. Speculative steps per second, on requests with at least 50 output tokens,
compare the engines' raw decode speed, against the MLX 4-bit:

| Quant | 32K | 64K | 96K | 128K |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | -0.7% | -2.5% | -2.8% | -2.0% |
| UD-Q4_K_M | -5.3% | -6.7% | -7.0% | -7.0% |
| UD-Q5_K_M | -12.8% | -12.9% | -12.6% | -10.6% |
| GSQ IQ3_S | -17.7% | -15.9% | -15.8% | -13.2% |
| PQ2_0 | -7.2% | -8.1% | -7.9% | -6.5% |

<details>
<summary>Decode and prefill tok/s by context</summary>

Decode tok/s, output-weighted:

| Quant | 32K | 64K | 96K | 128K |
| --- | ---: | ---: | ---: | ---: |
| MLX 4-bit | 48.5 | 43.4 | 42.1 | 39.4 |
| Q4_0 | 46.9 | 44.4 | 42.7 | 39.3 |
| UD-Q4_K_M | 40.4 | 42.7 | 39.7 | 37.5 |
| UD-Q5_K_M | 37.6 | 40.5 | 35.8 | 35.0 |
| GSQ IQ3_S | 38.8 | 38.0 | 37.4 | 34.7 |
| PQ2_0 | 41.7 | 34.3 | 35.0 | 31.7 |

Prefill tok/s, new prompt tokens over time to first token, cache hits
included:

| Quant | 32K | 64K | 96K | 128K |
| --- | ---: | ---: | ---: | ---: |
| MLX 4-bit | 213.2 | 143.6 | 119.8 | 100.5 |
| Q4_0 | 212.9 | 143.5 | 120.1 | 100.5 |
| UD-Q4_K_M | 205.8 | 139.5 | 117.1 | 98.6 |
| UD-Q5_K_M | 201.3 | 136.8 | 115.4 | 97.5 |
| GSQ IQ3_S | 206.9 | 130.9 | 105.3 | 87.0 |
| PQ2_0 | 169.0 | 125.1 | 105.7 | 90.4 |

</details>

- The GSQ IQ3_S rows come from an earlier run on 1.4.0-m1 with the
  performance changes that 1.5.0-m1 includes.
- UD-Q5_K_M and PQ2_0 have no closeness measurement here.
- [Source](https://github.com/paperniuk/splash/pull/43#discussion_r4445066753).
