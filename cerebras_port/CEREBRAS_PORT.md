# Porting `gpt-oss-120b` to an ALCF CS-3

## Executive recommendation

A port is technically plausible, but it is not a checkpoint-conversion exercise. The Cerebras Model Zoo release 2.10.0 has generic PyTorch and weight-streaming support, and its documented MoE support makes the model family a reasonable target. However, I found no evidence in the public release-2.10.0 Model Zoo of a ready-made `gpt-oss` model, an MXFP4 kernel, or an HF-to-CS converter for `GptOssForCausalLM`.

The recommended first implementation is therefore:

1. Use the official `openai/gpt-oss-120b` checkpoint and tokenizer as the reference.
2. Implement the exact architecture as a custom Cerebras PyTorch/Model Zoo model.
3. Initially expand the MXFP4 MoE weights to BF16 during offline conversion. Keep the router, attention, embeddings, and output head in BF16.
4. Run fixed-shape autoregressive inference through the Model Zoo weight-streaming path on one CS-3 / one CSX.
5. Only after numerical correctness is demonstrated, investigate a native MXFP4 Cerebras kernel or a later Cerebras release with explicit support.

This is likely to fit the ALCF appliance: ALCF documents a large-model MemoryX group of twelve 1128-GiB nodes, while the official BF16-expanded checkpoint is approximately 233.7 GiB before runtime artifacts. The model should not be expected to fit in the WSE's 44 GiB SRAM; weight streaming is essential.

The highest-risk items are exact MXFP4 decoding, the unconventional SwiGLU, alternating sliding/full attention, YaRN RoPE, attention denominator bias, autoregressive implicit loops, and vocabulary/logit memory at 201,088 tokens.

## Hardware and software facts

### ALCF CS-3

The ALCF CS-3 installation is a four-CS-3 Cerebras Wafer-Scale Cluster. Each CS-3 uses WSE-3 with:

- 900,000 AI-optimized cores
- 44 GB on-chip SRAM
- 48 KB dedicated SRAM per core
- approximately 21 PB/s on-chip memory bandwidth
- approximately 214 Pb/s fabric bandwidth

The cluster separates compute and model storage. MemoryX stores weights and streams them to the WSE; SwarmX broadcasts and aggregates across CS-X systems; input servers handle preprocessing and data delivery. For this inference target, use one CSX initially and the large MemoryX group.

ALCF jobs are submitted from a Cerebras user node through the appliance framework, not directly with a conventional batch scheduler. The ALCF instructions use `cszoo`, `csctl`, `/software/cerebras`, and a Python virtual environment built around the `Release_2.10.0` Model Zoo tag. User nodes require the ALCF proxy for external downloads:

```bash
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
```

The 2.10.0 documentation describes:

- Cerebras PyTorch 2.0 APIs
- Model Zoo `Trainer` and `cszoo` workflows
- PyTorch custom models using `cstorch.compile`
- CSX backend and `ClusterConfig`
- weight streaming as the normal large-model execution mode
- `--validate_only` and `--compile_only` workflows
- autoregressive inference settings such as `start_token`, `stop_sequences`, `max_tokens`, and `loop_dim`
- checkpoint conversion tools, but only for model implementations with registered converters

The ALCF environment setup is approximately:

```bash
mkdir -p ~/R_2.10.0
cd ~/R_2.10.0
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
git clone https://github.com/Cerebras/modelzoo.git
cd modelzoo
git checkout Release_2.10.0
cd ..
/usr/bin/python3.11 -m venv venv_cerebras_pt
source venv_cerebras_pt/bin/activate
pip install --upgrade pip
pip install -e modelzoo
```

Before implementing, record the actual appliance versions and available resources from the user node:

```bash
python --version
python -c 'import torch, cerebras.pytorch as cstorch; print(torch.__version__); print(cstorch.__file__)'
cszoo --help
cszoo checkpoint list-converters
csctl get jobs
csctl cluster --help
```

Do not assume the public `main` Model Zoo branch is compatible with this site. Use the checked-out `Release_2.10.0` source and the installed Cerebras packages.

## Target model

The official configuration identifies the target as `GptOssForCausalLM` with:

| Property | Value |
|---|---:|
| Layers | 36 |
| Hidden size | 2880 |
| Attention heads | 64 |
| Key/value heads | 8 |
| Head dimension | 64 |
| Experts | 128 per layer |
| Experts selected per token | 4 |
| Vocabulary | 201,088 |
| Maximum context | 131,072 tokens |
| Sliding window | 128 tokens on alternating layers |
| Dense attention | Alternates with sliding attention |
| RoPE | YaRN, theta 150000, factor 32, original context 4096 |
| Normalization | Pre-LN RMSNorm, epsilon 1e-5 |
| Activation | Modified gated SwiGLU, clamp limit 7 |
| Attention | GQA, rotary embeddings, learned denominator bias |
| Parameters | 116.83B total, 5.13B active/token |
| Official checkpoint | 60.8 GiB, native MXFP4 MoE weights plus BF16 non-MoE weights |
| License | Apache 2.0, subject to the model repository's terms |

The model card describes 128 experts with top-4 routing and softmax over only the selected experts. It uses alternating banded and fully dense attention, with 128-token bandwidth. The full layers use 131,072-token YaRN context. The official implementation also contains an unconventional clamped/residual SwiGLU, so substituting a standard Model Zoo SwiGLU is not acceptable without a numerical comparison.

The MXFP4 checkpoint stores each quantized tensor as packed FP4 values (`tensor.blocks`) plus block scales (`tensor.scales`); the scale is applied over the last dimension. MoE projection weights are the quantized portion. Do not reinterpret these tensors as ordinary int4 or replace them with generic `bitsandbytes` quantization.

## Why not directly use vLLM or the OpenAI reference implementation?

The OpenAI repository provides an excellent correctness oracle, including a simple BF16 PyTorch implementation and an optimized Triton/MXFP4 implementation. Neither is a Cerebras execution backend. vLLM is similarly GPU-oriented and is not the execution layer documented for the ALCF CS-3 appliance.

Use the OpenAI reference on CPU or a GPU host to establish expected logits and generated token sequences. Use Cerebras PyTorch/Model Zoo for the CS-3 implementation. This separation avoids trying to make Triton, CUDA, vLLM, or GPU-specific MXFP4 code run inside the Cerebras appliance.

## Proposed implementation

### Phase 0: freeze the reference

On a host with sufficient RAM and storage:

```bash
pip install 'torch>=2.0' transformers safetensors huggingface_hub
huggingface-cli download openai/gpt-oss-120b --local-dir ./gpt-oss-120b
```

Use the exact repository revision recorded by `huggingface-cli`; save its `config.json`, tokenizer files, and file hashes. Run the OpenAI reference implementation in BF16 and record:

- logits for one short prompt at each layer boundary if hooks are available
- next-token logits for prompt lengths such as 1, 128, 129, 4096, and 8192
- greedy token IDs for a small prompt suite
- sampled outputs only after greedy correctness works
- peak host memory and load time

Start with short contexts. The 131k context should be a later acceptance test, not a first compile target.

### Phase 1: make an exact BF16 custom model

Add a small private Model Zoo extension rather than modifying a generic GPT-2 model in place. The extension should contain:

- `GptOssModelConfig`
- `GptOssForCausalLM` or the Model Zoo equivalent expected by `Trainer`
- model blocks with RMSNorm, GQA, RoPE/YaRN, sliding/full masks, learned attention denominator bias, router, top-4 MoE, and the exact SwiGLU
- an inference forward path using the Cerebras implicit-loop API
- a tokenizer/data adapter that emits `torch.int32` IDs
- a checkpoint-loading/conversion utility

The public Model Zoo inference model pattern is approximately:

```python
import cerebras.pytorch as cstorch
import torch

class GptOssForCausalLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GptOssTransformer(config)
        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.start_token = config.start_token
        self.stop_sequences = config.stop_sequences
        self.max_tokens = config.max_tokens

    def forward(self, data, autoregressive=False):
        if autoregressive:
            return self.inference_step(data)
        return self.forward_logits(data)

    def inference_step(self, data):
        input_ids = data["input_ids"]
        loop_index = cstorch.experimental.start_implicit_loop(
            input_ids, loop_dim=1
        )
        hidden = self.transformer(
            input_ids,
            inference_loop_index=loop_index,
        )
        logits = self.lm_head(hidden)
        next_token = logits.argmax(dim=-1).to(torch.int32)
        return cstorch.experimental.update_implicit_loop(
            input_tensor=input_ids,
            index_tensor=loop_index,
            update_tensor=next_token,
            stop_sequences_tensor=data["stop_sequences"],
            start_token=self.start_token,
            max_tokens=self.max_tokens,
        )
```

This is a structural skeleton, not a drop-in implementation. Match the exact 2.10.0 signatures in the installed Model Zoo. In particular, determine whether the standard sampling helper and implicit-loop update support this model's output shape and whether logits must be projected or sharded before the vocabulary operation.

For the first Cerebras compile, use:

- batch size 1 or the smallest supported batch
- fixed sequence length, initially 128 or 256
- greedy decoding
- `max_tokens` 8 or 16
- BF16 activations and weights
- no dynamic shapes
- no sampling, tool calling, or chat-template logic inside the model

### Phase 2: convert the official checkpoint

First inspect the actual tensors and the release-2.10.0 converter registry:

```bash
cszoo checkpoint list-converters
cszoo checkpoint info ./gpt-oss-120b/model-*.safetensors
```

If no `gpt_oss` converter exists, write a streaming converter using `safetensors.safe_open`. It must:

1. Load each non-MoE tensor without an unnecessary full-model copy.
2. Decode packed MXFP4 values exactly according to the OpenAI implementation.
3. Apply each block scale in the correct last-dimension grouping.
4. Materialize decoded MoE projections as BF16 for the first port.
5. Preserve tensor orientation and map names into the custom model's state dict.
6. Write a Cerebras-compatible HDF5/`.mdl` checkpoint using the release-2.10.0 checkpoint API or a state-dict conversion supported by `cszoo`.
7. Emit a manifest containing source revision, tensor names, shapes, dtypes, hashes, and conversion statistics.

An intentionally explicit offline decode test should compare a handful of decoded projection rows against the OpenAI BF16 reference. Do not start a CS-3 compile until this test passes.

Expected storage planning:

- official MXFP4 source: approximately 60.8 GiB
- BF16 expansion of all 116.8B parameters: approximately 233.7 GiB, plus checkpoint/container overhead
- the documented large-model MemoryX group should have ample capacity, subject to project allocation and appliance selection
- local user-node scratch must still accommodate downloads, temporary conversion files, and output checkpoints

If the release-2.10.0 checkpoint writer cannot accept a checkpoint of this size or a custom state dict, contact ALCF/Cerebras support before attempting ad-hoc splitting. The appliance's MemoryX path is the supported solution; manually sharding weights across user processes is not.

### Phase 3: compile and execute a small target

Create a Model Zoo YAML based on the exact 2.10.0 Trainer schema. The following is a configuration sketch; field names must be confirmed against the installed release:

```yaml
trainer:
  init:
    backend:
      backend_type: CSX
      cluster_config:
        num_csx: 1
        job_time_sec: 86400
    model:
      name: gpt_oss
      hidden_size: 2880
      num_hidden_layers: 36
      num_attention_heads: 64
      num_key_value_heads: 8
      head_dim: 64
      num_local_experts: 128
      num_experts_per_tok: 4
      vocab_size: 201088
      max_position_embeddings: 131072
      sliding_window: 128
      max_tokens: 16
      start_token: 201088
      stop_sequences: []
      loop_dim: 1
    precision:
      enabled: true
      fp16_type: bfloat16
    checkpoint:
      load_checkpoint_states: model
  fit:
    train_dataloader: null
```

Inference may instead require the `cszoo lm_eval` or a custom evaluation callback workflow. Use the Model Zoo's existing generative inference examples as the template, not the training `fit` path if the release requires a separate inference callback.

Use validation and compile-only first:

```bash
cszoo validate ./gpt_oss_120b.yaml
cszoo lm_eval ./gpt_oss_120b.yaml \
  --target_device CSX \
  --checkpoint_path ./gpt-oss-120b-cs/checkpoint.mdl \
  --compile_only \
  --mount_dirs "$PWD" \
  --python_paths "$PWD"
```

For a full custom workflow, the equivalent launch should use the release's `run.py`/`cszoo` entry point and include absolute `--mount_dirs` and `--python_paths`. Put source code, tokenizer, checkpoint, and any generated inputs under mounted paths. Use `screen` or `tmux` on the user node for long compile/execute jobs.

After compile-only succeeds, run one short greedy request and inspect `csctl` logs. Reuse the same compile directory for repeated requests where the input shapes and inference settings are unchanged.

### Phase 4: numerical validation

Compare Cerebras and reference outputs at progressively larger scope:

1. Configuration and parameter-count checks.
2. Tensor-name, shape, and dtype checks after conversion.
3. MXFP4 decode checks against the official reference.
4. Embedding output for fixed token IDs.
5. Attention and MoE outputs for one layer on CPU.
6. Full BF16 logits for a short fixed prompt.
7. Greedy token IDs for 8, 16, and 64 generated tokens.
8. Sliding-window boundary cases around 128 tokens.
9. Alternating dense-layer and YaRN positions.
10. Long-context prompts at 4096, 8192, 32768, and finally 131072 tokens.

Use tolerances appropriate to BF16 and the decoded MXFP4 weights. Exact floating-point equality is not a valid criterion; greedy token agreement and bounded logit error are the primary criteria. If token agreement fails, use per-layer probes to isolate attention, routing, SwiGLU, or conversion errors.

A minimal result record should contain:

```json
{
  "model_revision": "<HF revision>",
  "cerebras_release": "Release_2.10.0",
  "checkpoint_format": "bf16-expanded-mxfp4",
  "prompt_sha256": "<hash>",
  "input_tokens": 128,
  "generated_tokens": 16,
  "greedy_token_match": true,
  "max_abs_logit_error": "<measured>",
  "first_token_latency_sec": "<measured>",
  "decode_tokens_per_sec": "<measured>"
}
```

### Phase 5: performance and production service

Once correctness is established:

- benchmark prompt lengths and generation lengths separately
- measure compile time, weight-transfer time, first-token latency, steady-state tokens/s, and host-side overhead
- test batch sizes 1, 2, 4, and the largest stable fixed batch
- test one versus multiple CSX systems only if the workload needs it
- measure MemoryX utilization and whether the 1128-GiB group is selected
- reuse compiled artifacts and avoid unnecessary checkpoint saves
- keep the model server outside the appliance if a persistent HTTP/OpenAI-compatible endpoint is needed; have it submit requests to a long-running Cerebras job or use the site's supported serving mechanism

Do not promise Cerebras Cloud's advertised approximately 3000 tokens/s for this first private port. That number is for Cerebras' production implementation, likely with proprietary kernel and serving optimizations. Treat it as an external reference, not an acceptance target.

## Native MXFP4 follow-up

BF16 expansion is the practical bring-up path, but it gives up much of the model's storage advantage and may reduce throughput. After the BF16 model works, investigate these options in order:

1. Ask the ALCF/Cerebras administrators whether the installed 2.10.0 appliance already has an internal `gpt-oss` implementation or MXFP4 kernel that is not in the public Model Zoo.
2. Check newer compatible Cerebras releases for explicit GPT-OSS/MXFP4 support before writing a kernel.
3. Determine whether Cerebras supported operations expose block-FP4 matmul or a custom-kernel interface in the site installation.
4. Implement a Cerebras-native block-MXFP4 expert linear layer if the API and compiler support it.
5. Keep router and non-MoE layers in BF16 unless the compiler documents a better supported format.

Do not silently dequantize and re-quantize with a different format while claiming equivalence. The official checkpoint's MXFP4 layout and scale grouping must be preserved for a native path.

## Risks and decision gates

| Gate | Success criterion | If it fails |
|---|---|---|
| Environment | 2.10.0 Model Zoo and CSX sample run | Resolve site environment before model work |
| Converter | Decoded tensors match official reference | Fix packing/scales/name mapping |
| Custom model | `cszoo validate` passes | Reduce to a single block and add ops incrementally |
| Small compile | Fixed-shape inference compiles | Replace unsupported op or request Cerebras guidance |
| Full BF16 load | Checkpoint loads and runs on the large MemoryX group | Verify MemoryX selection/capacity and checkpoint format |
| Numerical parity | Greedy tokens agree on short prompts | Probe per-layer outputs |
| Long context | 131k compile/execute is stable | Document a smaller operational context |
| Performance | Useful latency/throughput under ALCF limits | Optimize kernels, batching, and compile reuse |
| MXFP4 | Native format is supported without accuracy loss | Keep BF16-expanded fallback or stop native work |

The project should be stopped or re-scoped if any of these are true:

- release 2.10.0 cannot compile the required dynamic/implicit autoregressive loop
- the appliance does not expose enough MemoryX capacity for the chosen checkpoint representation
- the model's learned attention denominator bias or custom SwiGLU cannot be represented without changing model behavior
- custom MoE routing is unsupported or causes unacceptable compilation/runtime cost
- ALCF policy does not permit persistent service-like jobs or the required external data transfer

## First hardware session checklist

When access is available, run a small sample before downloading 60.8 GiB:

```bash
source ~/R_2.10.0/venv_cerebras_pt/bin/activate
cd ~/R_2.10.0/modelzoo
python -c 'import torch, cerebras.pytorch as cstorch; print(torch.__version__); print(cstorch.__file__)'
cszoo checkpoint list-converters
csctl get jobs
```

Then compile the supplied 111M GPT-3 sample from the ALCF guide. Record the exact command, generated job IDs, compile duration, execute duration, and logs. Next compile a tiny custom model containing only one GPT-OSS-style attention block and one two-expert MoE block. This tests the difficult operations without consuming a full-model compile allocation.

Before the full run, ask the system administrators these concrete questions:

1. Is the documented large-model MemoryX group available to this project and automatically selected for a 234-GiB BF16 checkpoint?
2. Is GPT-OSS already supported internally on this CS-3 installation?
3. Does the 2.10.0 appliance support MXFP4/block-FP4 matrix operations for user models?
4. Which Model Zoo model APIs and custom-kernel APIs are approved on the appliance?
5. What are the per-job time, storage, and concurrent-job limits for a full-model compile?
6. Is a long-running inference job or local request server permitted under the testbed usage policy?

## Sources

- [ALCF CS-3 system overview](https://docs.alcf.anl.gov/ai-testbed/cerebras/)
- [ALCF running a model/program](https://docs.alcf.anl.gov/ai-testbed/cerebras/running-a-model-or-program/)
- [ALCF environment customization](https://docs.alcf.anl.gov/ai-testbed/cerebras/customizing-environment/)
- [Cerebras 2.10.0 getting started](https://training-docs.cerebras.ai/rel-2.10.0/getting-started/overview)
- [Cerebras porting PyTorch models](https://training-docs.cerebras.ai/rel-2.10.0/model-zoo/migration/porting-pytorch-models-to-cerebras)
- [Cerebras weight streaming](https://training-docs.cerebras.ai/rel-2.10.0/concepts/weight-streaming-execution)
- [Cerebras checkpoint conversion](https://training-docs.cerebras.ai/rel-2.10.0/model-zoo/migration/convert-checkpoints-and-model-configs/convert-checkpoints-and-model-configs)
- [Cerebras generative evaluation on CSX](https://training-docs.cerebras.ai/rel-2.10.0/model-zoo/core-workflows/downstream-validation-using-eleuther-eval-harness)
- [OpenAI GPT-OSS repository](https://github.com/openai/gpt-oss)
- [GPT-OSS-120B Hugging Face model](https://huggingface.co/openai/gpt-oss-120b)
- [GPT-OSS model card](https://arxiv.org/html/2508.10925)
- [Cerebras WSE-3 specifications](https://www.cerebras.ai/cs3)
