# DeepSeek V2 Implementation From Scratch

A PyTorch implementation of DeepSeek-V2, featuring Multi-head Latent Attention (MLA) and Mixture of Experts (MoE) architecture, with support for both pre-training and supervised fine-tuning (SFT).

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Training Modes](#training-modes)
- [Code Issues & Fixes](#code-issues--fixes)
- [Usage](#usage)
- [Dependencies](#dependencies)

## 🎯 Overview

This implementation recreates the DeepSeek-V2 model architecture with:
- **Multi-head Latent Attention (MLA)**: Efficient attention mechanism with low-rank compression
- **Mixture of Experts (MoE)**: Sparse expert routing for efficient scaling
- **RoPE with Extensions**: YaRN, Dynamic NTK, Position Interpolation
- **KV Cache**: Optimized inference with key-value caching
- **8-bit Optimization**: Memory-efficient training with BitsAndBytes

## 🏗️ Architecture

### Key Components
1. **MLA (Multi-head Latent Attention)**: Reduces KV cache memory by compressing keys and values
2. **MoE (Mixture of Experts)**: Routes tokens to top-k experts out of 64 total experts
3. **Shared Experts**: Always-active experts combined with routed experts
4. **RoPE Variants**: Multiple rotary position embedding strategies for context extension

## 📁 File Structure

### Core Model Files

#### `config.py` & `config.json`
**Purpose**: Model configuration and hyperparameters
- Defines `DeepSeekConfig` dataclass
- Contains all architectural parameters (d_model, heads, layers, MoE settings)
- **What it's for**: Central configuration for model architecture

**Key Parameters**:
```python
d_model: 1024                    # Hidden dimension
nheads: 16                       # Number of attention heads
num_layers: 4                    # Number of transformer blocks
num_routed_experts: 64          # Total MoE experts
topk: 6                         # Active experts per token
kv_lora_rank: 512              # KV compression rank
```

---

#### `mla.py`
**Purpose**: Multi-head Latent Attention implementation
- Implements efficient attention with KV compression
- Includes multiple RoPE variants (YaRN, Dynamic NTK, Position Interpolation)
- Provides KV cache for fast inference
- **What it's for**: Core attention mechanism of DeepSeek-V2

**Key Classes**:
- `MultiHeadLatentAttention`: Main MLA implementation
- `RotaryEmbedding`: Base RoPE implementation
- `YarnRotaryEmbedding`: YaRN scaling for long contexts
- `KVCache`: Key-value caching for efficient generation

---

#### `moe.py`
**Purpose**: Mixture of Experts implementation
- Routes tokens to top-k experts
- Includes shared experts (always active)
- Implements load balancing loss
- **What it's for**: Sparse expert routing for model capacity scaling

**Key Classes**:
- `MoE`: Main MoE layer with routing logic
- `FeedForward`: Individual expert network (SwiGLU variant)
- `Distributor`: Batches inputs by expert assignment
- `AddAuxiliaryLoss`: Adds load balancing loss during backprop

---

#### `deepseek.py`
**Purpose**: Complete DeepSeek model implementation
- Assembles MLA + MoE into transformer blocks
- Implements model forward pass and generation
- **What it's for**: Main model class used for training and inference

**Key Classes**:
- `DeepSeekModelForCausalLM`: Complete language model with LM head
- `DeepSeekModel`: Base transformer without LM head
- `Block`: Single transformer layer (MLA + MoE + LayerNorm)

---

### Training Files

#### `train.py`
**Purpose**: Pre-training script for causal language modeling
- **What it's for**: Pre-training the model on raw text (TinyShakespeare or FineWebEdu)
- Implements training loop with gradient accumulation
- Supports cosine learning rate schedule with warmup
- Includes 8-bit optimizer support
- **Usage**: `python train.py --dataset fineweb_edu --batch-size 8`

**Key Features**:
- Mixed precision training (bfloat16/float16)
- Gradient clipping
- Checkpoint saving/resuming
- Weights & Biases logging

---

#### `sft.py`
**Purpose**: Supervised Fine-Tuning script
- **What it's for**: Fine-tuning the pre-trained model on instruction-following datasets
- Uses HuggingFace datasets (e.g., ultrachat_200k)
- Applies ChatML format for conversations
- **Usage**: `python sft.py` (config loaded from `sft.json`)

**Training Flow**:
1. Loads pre-trained checkpoint (if available)
2. Formats data with ChatML template
3. Masks non-assistant tokens in loss
4. Fine-tunes with conversation data

---

#### `trainer.py`
**Purpose**: Generic trainer class for SFT
- **What it's for**: Reusable training logic extracted from `sft.py`
- Handles training loop, evaluation, checkpointing
- Works with PyTorch DataLoader
- Supports gradient accumulation and mixed precision

---

#### `training_config.py`
**Purpose**: Training hyperparameters configuration
- **What it's for**: Centralized training settings (learning rate, batch size, etc.)
- Used by both `train.py` and `sft.py`

---

### Data Processing Files

#### `dataloader.py`
**Purpose**: Data loading for pre-training
- **What it's for**: Loading tokenized pre-training data (TinyShakespeare, FineWebEdu)
- Loads sharded numpy arrays
- Provides next batch with automatic shard rotation
- **Used by**: `train.py`

**Key Classes**:
- `DataLoader`: Base dataloader for sharded data
- `FineWebEduDataLoader`: FineWeb-Edu dataset loader
- `TinyShakespeareDataLoader`: Shakespeare dataset loader

---

#### `datacollator.py`
**Purpose**: Data collation for supervised fine-tuning
- **What it's for**: Formatting conversational data into ChatML format
- Pads sequences to batch max length
- Masks non-assistant tokens (loss only on assistant responses)
- **Used by**: `sft.py`

**Key Classes**:
- `DataCollatorForChatMl`: Processes chat conversations
- `ChatMlSpecialTokens`: Defines special tokens (`<|im_start|>`, `<|im_end|>`)

**ChatML Format**:
```
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm fine, thank you!<|im_end|>
```

---

#### `tokenizer.py`
**Purpose**: Tokenizer wrapper
- **What it's for**: Text tokenization using TikToken (cl100k_base)
- Adds special tokens for ChatML
- **Used by**: `datacollator.py`, `sft.py`

---

#### `sampling.py`
**Purpose**: Sampling strategies for generation
- **What it's for**: Nucleus (top-p) and top-k sampling during inference
- **Functions**:
  - `sample_top_p`: Nucleus sampling
  - `sample_top_k`: Top-k sampling

---

### Configuration Files

#### `sft.json`
**Purpose**: SFT training configuration
- **What it's for**: Hyperparameters for supervised fine-tuning
- Specifies dataset, learning rate, batch size, etc.

---

## 🔄 Training Modes

### 1. Pre-Training (`train.py`)
**Purpose**: Train the model from scratch on raw text

```bash
python train.py \
  --dataset fineweb_edu \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-4 \
  --max-train-steps 30000
```

**What it does**:
- Trains on next-token prediction task
- Uses raw text data (tokenized into shards)
- Learns general language understanding

**Datasets**:
- `tinyshakespeare`: Small Shakespeare corpus for testing
- `fineweb_edu`: Educational web content (FineWeb-Edu)

---

### 2. Supervised Fine-Tuning (`sft.py`)
**Purpose**: Fine-tune on instruction-following conversations

```bash
python sft.py
```

**What it does**:
- Loads pre-trained checkpoint (optional)
- Trains on conversational data (user-assistant exchanges)
- Only computes loss on assistant responses
- Makes model follow instructions

**Dataset**:
- HuggingFace: `HuggingFaceH4/ultrachat_200k`
- Format: Conversation turns with roles (user/assistant)

---

## 🐛 Code Issues & Fixes

### Critical Issues Found

#### 1. ❌ **Missing Attention Mask Processing in `deepseek.py`**
**Location**: `deepseek.py:43` (Block.forward)

**Issue**: The attention mask is passed but never converted to the proper format for PyTorch's scaled_dot_product_attention.

**Current Code**:
```python
x, past_key_value = self.self_attn(
    self.input_layernorm(x), past_key_value, attention_mask
)
```

**Problem**: `attention_mask` with shape `(B, T)` needs to be converted to `(B, 1, T, T)` causal mask.

**Fix**: In `mla.py`, ensure `get_attention_mask()` is called properly:
```python
# In mla.py around line 541-542
if q_len == kv_seq_len:
    attn_mask = get_attention_mask(attn_mask)  # This line exists but get_attention_mask is not defined!
```

**MISSING FUNCTION**: `get_attention_mask()` is called but never defined!

**Add this function to `mla.py`**:
```python
def get_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Convert attention mask from (B, T) to causal mask (B, 1, T, T)
    
    Args:
        attention_mask: (B, T) where 1 = attend, 0 = mask
    
    Returns:
        Causal attention mask (B, 1, T, T) with -inf for masked positions
    """
    if attention_mask is None:
        return None
    
    B, T = attention_mask.shape
    # Create causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones((T, T), device=attention_mask.device))
    causal_mask = causal_mask.view(1, 1, T, T)
    
    # Combine with padding mask
    attention_mask = attention_mask.view(B, 1, 1, T)
    combined_mask = causal_mask * attention_mask
    
    # Convert to additive mask (0 -> -inf, 1 -> 0)
    combined_mask = (1.0 - combined_mask) * torch.finfo(attention_mask.dtype).min
    
    return combined_mask
```

---

#### 2. ⚠️ **Debug Print Statement in Production Code**
**Location**: `deepseek.py:46`

```python
if past_key_value is not None:
    print(f"past_key_value shape: {past_key_value.key_cache[0].shape}")
```

**Issue**: This will spam console during training/inference with KV cache.

**Fix**: Remove or wrap in debug flag:
```python
# Remove this line entirely, or:
if past_key_value is not None and os.environ.get('DEBUG', False):
    print(f"past_key_value shape: {past_key_value.key_cache[0].shape}")
```

---

#### 3. ⚠️ **Incorrect Logic in MoE Parameter Counting**
**Location**: `deepseek.py:100-105`

```python
if not name.find(routed_moe_module_name) == -1 or is_activated_routed_moe_module(name):
    activated_params += param.numel()
```

**Issue**: The logic is inverted. `str.find()` returns `-1` when NOT found, so `not ... == -1` means "found". But then it's OR'ed with another check that also means "found", which is redundant.

**Should be**:
```python
if name.find(routed_moe_module_name) == -1 or is_activated_routed_moe_module(name):
    activated_params += param.numel()
```

**Better fix** (clearer logic):
```python
# Count all params except inactive routed experts
if routed_moe_module_name not in name or is_activated_routed_moe_module(name):
    activated_params += param.numel()
```

---

#### 4. ⚠️ **Type Hint Inconsistency**
**Location**: Multiple files

```python
# In deepseek.py
def forward(self, x: torch.tensor, ...) -> torch.tensor:
```

**Issue**: Should be `torch.Tensor` (capital T), not `torch.tensor` (lowercase - that's a function).

**Fix**: Change all occurrences:
```python
def forward(self, x: torch.Tensor, ...) -> torch.Tensor:
```

---

#### 5. ⚠️ **Typo in Comment**
**Location**: `trainer.py:23` and `train.py:75`

```python
# create optim groups. Any parameters that is 2D will be weight decayed, otherwis    no.
```

**Fix**: `otherwis` → `otherwise`, extra spaces

---

#### 6. ⚠️ **Missing Import in `mla.py`**
**Location**: `mla.py:542`

**Issue**: `get_attention_mask()` is called but never imported or defined.

**Fix**: Add the function definition (see Fix #1 above).

---

#### 7. ⚠️ **Inconsistent Device Handling**
**Location**: `dataloader.py:97-104`

**Issue**: `TinyShakespeareDataLoader` has special device handling with `pin_memory()`, but `FineWebEduDataLoader` doesn't.

**Recommendation**: Standardize device handling or document why they differ.

---

#### 8. ⚠️ **Unused `ctx` Parameter in `train.py`**
**Location**: `train.py:113-117`

```python
def forward_and_backward(model, x, y, optimizer, ctx: torch.autocast = nullcontext()):
    with ctx:
        _, loss, _ = model(x, y)
```

**Issue**: Function is defined but never called in the code!

**Fix**: Either remove the function or use it in `train()`:
```python
# In train() function, replace:
with ctx:
    _, train_loss, _ = model(x, y)
train_loss = train_loss / args.gradient_accumulation_steps
train_loss.backward()

# With:
forward_and_backward(model, x, y, optimizer, ctx)
```

---

#### 9. ✅ **Commented Out Code**
**Location**: `sft.py:43-55`

```python
# max_len = 0
# train_batch_size = 0
# ...
```

**Recommendation**: Remove commented debugging code before production.

---

### Summary of Required Fixes

| Priority | Issue | File | Line | Fix Required |
|----------|-------|------|------|--------------|
| 🔴 CRITICAL | Missing `get_attention_mask()` function | `mla.py` | 542 | Add function definition |
| 🟡 MEDIUM | Debug print in production | `deepseek.py` | 46 | Remove or guard with flag |
| 🟡 MEDIUM | Inverted MoE param counting | `deepseek.py` | 100-105 | Fix boolean logic |
| 🟢 LOW | Type hints `torch.tensor` → `torch.Tensor` | Multiple | Various | Update type hints |
| 🟢 LOW | Typo in comment | `trainer.py`, `train.py` | 23, 75 | Fix spelling |
| 🟢 LOW | Unused function | `train.py` | 113 | Remove or use |

---

## 🚀 Usage

### Installation

```bash
pip install torch tiktoken datasets transformers bitsandbytes wandb python-dotenv
```

### Pre-Training

```bash
# Train on TinyShakespeare (small test)
python train.py --dataset tinyshakespeare --batch-size 4 --max-train-steps 1000

# Train on FineWebEdu (full pre-training)
python train.py --dataset fineweb_edu --batch-size 8 --gradient-accumulation-steps 8
```

### Supervised Fine-Tuning

```bash
# Edit sft.json to configure training
python sft.py
```

### Generation

```python
import torch
from deepseek import DeepSeekModelForCausalLM, DeepSeekConfig
import json

# Load config
with open("config.json", "r") as f:
    config = json.load(f)
config = DeepSeekConfig(**config)

# Load model
model = DeepSeekModelForCausalLM(config)
model.load_state_dict(torch.load("checkpoint.pt")["model"])
model.eval()

# Generate
input_ids = torch.tensor([[1, 2, 3, 4]]).to(config.device)
output = model.generate(input_ids, max_length=50, temperature=0.7)
print(output)
```

---

## 📦 Dependencies

```
torch>=2.0.0
tiktoken
datasets
transformers
bitsandbytes
wandb
python-dotenv
numpy
jinja2
```

---

## 📊 Model Statistics

From `config.json`:
- **Parameters**: ~2.3B (estimated with d_model=1024, 4 layers, 64 experts)
- **Active Parameters**: ~400M per token (6 experts + shared experts)
- **Context Length**: 4096 tokens (extendable to 16K+ with YaRN)
- **Vocabulary**: 101,024 tokens

---

## 🎓 Key Concepts

### What is Pre-Training?
- Training from scratch on raw text
- Task: predict next token
- Learns general language patterns
- **File**: `train.py`

### What is SFT (Supervised Fine-Tuning)?
- Fine-tuning pre-trained model on instructions
- Task: follow user instructions
- Learns to be helpful, harmless, honest
- **File**: `sft.py`

### What is MLA (Multi-head Latent Attention)?
- Compresses KV cache using low-rank projection
- Reduces memory by ~5x compared to standard attention
- **File**: `mla.py`

### What is MoE (Mixture of Experts)?
- Uses sparse expert routing
- Only activates top-k experts per token
- Scales model capacity without proportional compute cost
- **File**: `moe.py`

---

## 🔧 Configuration Tips

### For Small GPU (8GB):
```json
{
  "d_model": 512,
  "num_layers": 2,
  "num_routed_experts": 16,
  "batch_size": 1,
  "gradient_accumulation_steps": 32,
  "use_eight_bit_optimizer": true
}
```

### For Large GPU (80GB):
```json
{
  "d_model": 2048,
  "num_layers": 8,
  "num_routed_experts": 64,
  "batch_size": 16,
  "gradient_accumulation_steps": 4
}
```

---

## 📝 License

This is an educational implementation. For production use, refer to DeepSeek's official repository and license.

---

## 🙏 Acknowledgments

Based on the DeepSeek-V2 paper:
- Paper: https://arxiv.org/abs/2405.04434
- Official Implementation: https://huggingface.co/deepseek-ai/DeepSeek-V2

---

## 🐛 Reporting Issues

Found a bug? Check the [Code Issues & Fixes](#code-issues--fixes) section first!