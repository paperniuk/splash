# Splash for M1 and M2 Macs

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/paperniuk/splash)](https://github.com/paperniuk/splash/releases/latest)

An unofficial port of [Inco's Splash](https://github.com/incoai/splash) to
Macs with an M1 or M2 chip. Splash is a local inference engine that serves
Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next and Flint to coding agents and to
any OpenAI or Anthropic compatible client. Official Splash needs an M3 or
newer; this fork runs the same engine, version 1.6.0, on M1 and M2.

**On an M3 or newer, use official Splash:** `brew install incoai/tap/splash`.
Nothing here is faster there.

The engine, the models and the draft models are Inco's work. What this fork
adds:

- **It starts on M1/M2.** Upstream refuses GPUs older than the M3 family.
- **Metal kernels written for these GPUs.** Upstream's kernels use Apple's
  matrix library, which runs far below these GPUs' simdgroup matrix rate and
  hangs the GPU on some GGUF shapes. The kernels here cover the MLX 4-bit
  checkpoints, GGUF files, attention, the vision encoder and Flash-Next's
  own kernels.
- **Neural Engine prefill on M1/M2.** The dense 27B splits each FFN layer of
  long prompts between the GPU and the Neural Engine, about 1.5x faster
  prefill on an M1 Max.
- **Long prompts in short GPU commands.** macOS aborts a GPU command that
  holds the GPU for too long, and on M1/M2 a full prefill chunk does. Prefill
  is split, so agent sessions do not end with an engine failure. See
  [Long prompts](#long-prompts).
- **A prebuilt release.** One line installs it, with no Xcode, Homebrew or pip.

Measured on an M1 Max with 64 GB. M1 Pro, M1 Ultra and M2 run the same code,
and owners of an M1 Ultra and an M2 Max have reported it working.

## Install

Needs macOS 26.4 or later.

```bash
curl -fsSL https://github.com/paperniuk/splash/releases/latest/download/install-m1.sh | sh
splash-m1 serve --model mlx-community/Qwen3.8-27B-4bit
```

This installs the command `splash-m1` and leaves an official `splash` alone.
The first run downloads the model and its draft and prepares the weights;
later starts reuse them. Once it prints `Ready`, open
<http://127.0.0.1:8000>, or start an installed agent from another terminal:

```bash
splash-m1 opencode    # or: splash-m1 claude / codex / hermes / pi
```

Running the same `curl` line again upgrades. Models live in the Hugging Face
cache and survive upgrades.

## Which model for your Mac

| Memory | Model | `--model` | Weights |
|---|---|---|---|
| 64 GB | Qwen3.8-Flash-Next, MoE | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0` | 66 GB |
| 64 GB | Flint | `unsloth/flint-GGUF:UD-Q4_K_XL` | 25 GB |
| 36 GB or more | Qwen3.8-27B, dense | `mlx-community/Qwen3.8-27B-4bit` | 15 GB |
| 36 GB or more | Qwen3.6-35B-A3B, MoE, fastest | `mlx-community/Qwen3.6-35B-A3B-4bit` | 20 GB |
| 36 GB or more | Qwen3.8-27B as GGUF, other quants | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | 17 GB |
| 24 GB | a smaller GGUF quant | `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` | |
| 16 GB | Ternary Bonsai 2, a 27B in 7 GB | `prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0` | 7 GB |

The old `incoai/...-Splash` packages no longer load since Splash 1.3; serve
the MLX checkpoint of the same model instead.

Everything here was measured on a 64 GB Mac. The memory column for the 27B
and 35B rows is upstream's requirement. Flash-Next fits a 64 GB Mac because
its 27 GB per-layer embedding table stays on the SSD and is read as needed;
the experts stay in memory. Bonsai was run with memory capped to
11 GB, where it starts with a 28K context; a stock 16 GB Mac gives the GPU
about 10.7 GB, slightly too little, so raise the limit first:

```bash
sudo sysctl iogpu.wired_limit_mb=12288   # resets at reboot
splash-m1 serve --model prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0 --language-only
```

`--language-only` skips vision and saves its memory. Reports from 16, 24 and
32 GB Macs are welcome.

## Speed on an M1 Max

M1 Max, 32-core GPU, 64 GB, release 1.6.0-m1, temperature 0, 400 tokens
on a short prompt. Decode in tok/s, one request / two at once:

| Model | 1 request | 2 requests |
|---|---:|---:|
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0` | 79 | 98 |
| `unsloth/flint-GGUF:UD-Q4_K_XL` | 77 | 94 |
| `mlx-community/Qwen3.8-27B-4bit` | 55 | 60 |
| `unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_M` | 47 | 54 |

Prose is slower than code because the draft model guesses fewer tokens.
Flint was measured on 1.5.0-m1 with its fork pull request.

The 27B reads a cold 16K prompt at about 197 tok/s with the Neural Engine.
In a replayed agent session that grows to 128K of context, prompt reading
averaged 175 tok/s up to 32K, 119 at 64K, 91 at 96K and 73 at 128K, and
decode 25, 26, 21 and 17 tok/s. Splash keeps the prompt in a prefix cache,
so an agent pays for each part of it once.

On an M2 Max, [a user measured](https://github.com/paperniuk/splash/issues/5)
the same kind of session at about 204 tok/s of prompt reading up to 32K and 79 at
128K.

Method and the other numbers are in the
[release notes](https://github.com/paperniuk/splash/releases).

## Flags that behave differently on M1/M2

Upstream's README describes every flag. These matter more here.

### Long prompts

`--prefill-mode bounded` is the default: every GPU command buffer of a
prefill stays within 2 seconds (`--prefill-bound-ms`). When a command holds
the GPU for too long while the display needs it, macOS aborts it
(`ImpactingInteractivity`) and the engine restarts with the bound halved,
down to 500 ms.

Do not pass `--prefill-mode full` on M1/M2. It restores upstream's behaviour,
which is what gets aborted. Keep `--prefill-bound-scope` at its default,
`buffer`: `command` bounds whole commands instead, and on an M2 Max
`buffer` read prompts 9 to 36% faster. The engine switches to `command` by
itself after an abort, only when it has to.

### Neural Engine

The dense 27B prefills with the Neural Engine by default. The first start
measures it against the GPU and remembers the split; MoE models run their
prefill on the GPU only, as upstream. `--disable-ane` keeps everything on
the GPU.

### `--kv-format bf16`

Keep the default INT8 cache unless you are comparing quality. BF16 takes
twice the memory, and in a 128K agent session on an M1 Max it read prompts
2 to 4% slower and decoded up to 8% slower.

## API and settings

The API is upstream's: OpenAI Chat Completions, OpenAI Responses and
Anthropic Messages on `127.0.0.1:8000`, with streaming, tool calls, images
and PDFs. `splash-m1 serve --help` lists the options. For the rest, read
[the README of the 1.6 branch](https://github.com/paperniuk/splash/blob/apple7-m1-1.6/README.md)
with `splash-m1` in place of `splash`. Since 1.4, `--max-context` can go past
a model's native window with YaRN (`--yarn-factor`).

Reasoning is turned off per request with `"reasoning_effort": "none"`.

## Build from source

Needs Xcode with the Metal Toolchain.

```bash
git clone -b apple7-m1-1.6 https://github.com/paperniuk/splash.git
cd splash
make -j4
./splash serve --model mlx-community/Qwen3.8-27B-4bit
```

Branch `apple7-m1-1.6` is upstream 1.6.0 plus the M1/M2 commits. `main`
follows upstream and does not run on M1/M2.

## Status

- Stable: [1.6.0-m1](https://github.com/paperniuk/splash/releases/latest),
  upstream 1.6.0 plus the M1/M2 work.
- Known gaps: the 35B MoE's expert kernels are not tuned for M1/M2 yet;
  Macs below 64 GB are covered by the kernel tests, not run end to end on
  this release.

Problems on M1/M2 belong in [this fork's issues](https://github.com/paperniuk/splash/issues),
not upstream's.

## Credits

Splash is built by [Inco](https://github.com/incoai/splash); the
[launch post](https://inco.ai/blog/splash/) explains its design. If this port
is useful to you, please star the original too. The installer is based on
npanj's `install-q8.sh`. [@ghisguth](https://github.com/ghisguth) tested
the port on an M2 Max and contributed most of its decode speedups: GGUF Q4_0,
the K-quant and MLX decode kernels and fewer decode passes.
[@sady4850](https://github.com/sady4850) contributed the faster long-context
attention and Neural Engine prefill from 128 rows,
[@abssoul](https://github.com/abssoul) Flint and the small MoE tiles, and
[@airiclenz](https://github.com/airiclenz) and
[@UtmostMaker](https://github.com/UtmostMaker) scheduler and upstream port work.

Apache-2.0, see [LICENSE](LICENSE); the GGUF kernels include MIT-licensed
material from llama.cpp, see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
Model weights keep their own licenses.
