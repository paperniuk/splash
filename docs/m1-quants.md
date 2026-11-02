# Qwen3.8-27B quants on M1

[Back to the docs](README.md)

Which 4- and 5-bit Qwen3.8-27B checkpoint to run on an M1 Pro with 32 GB, and
how close each one stays to the bf16 model when Splash runs it. All numbers
come from this fork's runtime on one M1 Pro (32 GB), text only.

## Which one

- **Default: [`sady4850/Qwen3.8-27B-4bit-GPTQ`](https://huggingface.co/sady4850/Qwen3.8-27B-4bit-GPTQ).**
  Same layout, size and speed as `mlx-community/Qwen3.8-27B-4bit`, only
  GPTQ-rounded: KL to bf16 29% lower on wiki, 47% lower on chat.
- **More quality for 1.3 GB more: `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`.**
  Closest of the 4-bit files, decode about 11% slower.
- **Closest overall: `unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_S`** (or the SLQ EAR98
  file, about equal at about the same size). About 3.5 GB more than the MLX
  4-bit and 20% slower.
- **Short on memory: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S`.** 3.4 GB
  smaller, about as close to bf16 as the MLX 4-bit, but slower.

```bash
splash serve --model sady4850/Qwen3.8-27B-4bit-GPTQ
splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

## Results

KL is KL(bf16 || quant) per position, averaged. EAR@10 is the overlap of the
two distributions over bf16's 10 most likely tokens, both renormalized to
those 10 (100 means identical there). Size is the text weights in GB
(10^9 bytes). Decode is the decode time of the same teacher-forced windows
against the MLX 4-bit; positive is slower.

| Quant | Size | KL wiki | KL chat | EAR@10 wiki | EAR@10 chat | Decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/Qwen3.8-27B-4bit` | 15.13 | 0.0562 | 0.1906 | 92.69 | 88.66 | base |
| `sady4850/Qwen3.8-27B-4bit-GPTQ` | 15.13 | 0.0399 | 0.1002 | 94.36 | 92.22 | -0.2% |
| `unsloth/Qwen3.8-27B-GGUF:Q4_0` | 16.06 | 0.0325 | 0.1412 | 94.64 | 91.26 | +5.9% |
| `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | 16.46 | 0.0144 | 0.0831 | 97.18 | 93.90 | +11.1% |
| `unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_S` | 18.67 | 0.0065 | 0.0574 | 97.92 | 95.24 | +20.2% |
| `kaitchup/Qwen3.8-27B-SLQ-GGUF:EAR98-Q5_K_S` | 18.88 | 0.0104 | 0.0486 | 97.97 | 95.57 | +19.7% |
| `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S` | 11.77 | 0.0573 | 0.2121 | 92.54 | 87.64 | +18.2% |

- Every quant but the GSQ IQ3_S differs from the MLX 4-bit in KL on both texts
  (95% paired bootstrap over windows). The GSQ IQ3_S does not, on either.
- The wiki mean depends on one token in one window. Without that window:
  MLX 4-bit 0.0436, GPTQ 0.0280, Q4_0 0.0264, UD-Q4_K_M 0.0075, UD-Q5_K_S
  0.0036, SLQ EAR98 0.0036, GSQ IQ3_S 0.0500.
- Perplexity ranks these quants wrongly: the MLX 4-bit and the GSQ IQ3_S score
  below bf16's chat perplexity. Compare quants by KL.

## How it was measured

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
- **Speed.** Median decode seconds per window over 60 windows. KL, EAR and
  speed were measured on 1.4.0-m1, speed in separate sessions per quant. The
  GSQ IQ3_S and MLX 4-bit were re-measured on 1.5.0-m1 in one session,
  alternating, 6 windows: the GSQ's top-512 log-probabilities came out
  bit-identical and its decode 8% faster (+18.2% here, +27.5% on 1.4.0-m1).
  The MLX 4-bit stayed within noise.

## Limits

- The texts are prompt text, not the model's own answers or agent sessions.
  KL to bf16 measures closeness, not task success.
- `sady4850/Qwen3.8-27B-4bit-GPTQ` is the author's checkpoint. Part of its
  calibration is UltraChat train, so its chat number may be optimistic; wiki is
  the cleaner comparison.
- Speed is one request at short context. At long context the gaps get
  smaller.
- One model, one Mac.
- Independent agent check: Alexander Fedora replayed 78 recorded agent turns
  against Q8_0 on an M2 Max; the share of turns with the same tool call was
  MLX 4-bit 31%, GPTQ 44%, Q4_0 44%, UD-Q4_K_M 51%, UD-Q5_K_M 56%, GSQ IQ3_S
  44% ([discussion #34](https://github.com/paperniuk/splash/discussions/34#discussioncomment-19226461)).
