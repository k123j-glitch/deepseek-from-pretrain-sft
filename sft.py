import os
import argparse
import torch
import json
from config import DeepSeekConfig
from deepseek import DeepSeekModelForCausalLM
import torch.nn as nn
import math
from contextlib import nullcontext
import bitsandbytes as bnb
from tokenizer import Tokenizer
import datasets
from torch.utils.data import DataLoader, Dataset
from datacollator import DataCollatorForChatMl, ChatMlSpecialTokens

from dotenv import load_dotenv

load_dotenv()

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class SFTDataset(Dataset):
    """Custom PyTorch Dataset for SFT"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["messages"]


def configure_optimizers(
        model, weight_decay, learning_rate, betas, fused, use_eight_bit_optimizer=False
):
    """Configure AdamW optimizer with weight decay"""
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    if use_eight_bit_optimizer:
        # fuse is not supported
        optimizer = bnb.optim.AdamW8bit(
            optim_groups, lr=learning_rate, betas=betas
        )
    else:
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=fused
        )
    return optimizer


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Learning rate decay scheduler (cosine with warmup)"""
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_next_batch(dataloader_iter, dataloader):
    """Get next batch from dataloader, reset iterator if needed"""
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        # Reset the iterator
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
    return batch, dataloader_iter


@torch.no_grad()
def evaluate(
        model: nn.Module,
        eval_iters: int,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        train_iter: iter,
        eval_iter: iter,
        device: str,
):
    """Evaluate model on both train and eval datasets"""
    model.eval()
    loss_iters = torch.zeros(eval_iters)
    losses = {}

    # Evaluate on eval dataset
    for i in range(eval_iters):
        eval_batch, eval_iter = get_next_batch(eval_iter, eval_dataloader)
        _, loss, _ = model(
            x=eval_batch["input_ids"].to(device),
            targets=eval_batch["labels"].to(device),
            past_key_value=None,
            attention_mask=eval_batch["attention_mask"].to(device),
        )
        loss_iters[i] = loss.item()
    losses["eval"] = loss_iters.mean()

    # Evaluate on train dataset
    for i in range(eval_iters):
        train_batch, train_iter = get_next_batch(train_iter, train_dataloader)
        _, loss, _ = model(
            x=train_batch["input_ids"].to(device),
            targets=train_batch["labels"].to(device),
            past_key_value=None,
            attention_mask=train_batch["attention_mask"].to(device),
        )
        loss_iters[i] = loss.item()
    losses["train"] = loss_iters.mean()

    model.train()
    return losses, train_iter, eval_iter


def get_model_config(model_config_path):
    """Load model config from JSON file"""
    with open(model_config_path, "r") as f:
        config = json.load(f)
    return config


def get_model(args):
    """Initialize model and convert to appropriate dtype"""
    config = get_model_config(args.model_config_path)
    model = DeepSeekModelForCausalLM(DeepSeekConfig(**config))
    # Convert model to the appropriate dtype before moving to device
    model = model.to(dtype=ptdtype[args.dtype])
    model = model.to(args.device)
    total_params, activated_params = model.get_total_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Activated parameters: {activated_params:,}")
    print(f"Activated parameters ratio: {activated_params / total_params:.2%}")
    return model


def get_dataloaders(args):
    """Create train and eval dataloaders with data collator"""
    # Initialize tokenizer
    tokenizer = Tokenizer(args.tokenizer_type)
    tokenizer.add_special_tokens(
        [ChatMlSpecialTokens().bos_token, ChatMlSpecialTokens().eos_token]
    )

    # Load datasets
    print(f"Loading dataset: {args.dataset_name}")
    train_dataset = datasets.load_dataset(args.dataset_name, split=args.train_split)
    eval_dataset = datasets.load_dataset(args.dataset_name, split=args.eval_split)

    sft_train_dataset = SFTDataset(train_dataset)
    sft_eval_dataset = SFTDataset(eval_dataset)

    # Create data collator
    data_collator = DataCollatorForChatMl(
        tokenizer,
        tokenizer.eos_token_id,
        -100,  # ignore_index for CrossEntropyLoss
        ChatMlSpecialTokens().assistant,
        tokenizer.eos_token_id,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        sft_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )
    eval_dataloader = DataLoader(
        sft_eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator.process,
    )

    return train_dataloader, eval_dataloader


def train(args):
    """Main training loop for supervised fine-tuning"""
    # No autocast needed - we convert the entire model to bfloat16
    ctx = nullcontext()

    if args.wandb_log:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=get_wandb_config(args),
        )

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(args)

    # Initialize model
    model = get_model(args)
    os.makedirs(args.out_dir, exist_ok=True)

    # Configure optimizer
    optimizer = configure_optimizers(
        model,
        args.adamw_weight_decay,
        args.learning_rate,
        (args.adamw_beta1, args.adamw_beta2),
        args.adamw_use_fused,
        args.use_eight_bit_optimizer,
    )
    optimizer.zero_grad(set_to_none=True)

    best_val_loss = 1e9
    iter_num = 0

    # Resume from checkpoint if requested
    if args.resume:
        checkpoint = torch.load(os.path.join(args.out_dir, args.checkpoint_path))
        config = checkpoint["model_config"]
        model = DeepSeekModelForCausalLM(config)
        # Convert model to the appropriate dtype before moving to device
        model = model.to(dtype=ptdtype[args.dtype])
        model = model.to(args.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_val_loss = checkpoint["best_val_loss"]
        iter_num = checkpoint["iter_num"]
        print(f"Resumed from checkpoint at step {iter_num}")

    # Initialize dataloader iterators
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # Training loop
    while iter_num < args.max_train_steps:
        # Determine and set the learning rate for this iteration
        lr = (
            get_lr(
                iter_num,
                args.warmup_iters,
                args.lr_decay_iters,
                args.learning_rate,
                args.min_learning_rate,
            )
            if args.decay_lr
            else args.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        for _ in range(args.gradient_accumulation_steps):
            batch, train_iter = get_next_batch(train_iter, train_loader)
            with ctx:
                _, train_loss, _ = model(
                    x=batch["input_ids"].to(args.device),
                    targets=batch["labels"].to(args.device),
                    past_key_value=None,
                    attention_mask=batch["attention_mask"].to(args.device),
                )
            train_loss = train_loss / args.gradient_accumulation_steps
            train_loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Evaluation
        if (iter_num + 1) % args.eval_interval == 0:
            losses, train_iter, val_iter = evaluate(
                model,
                args.eval_iters,
                train_loader,
                val_loader,
                train_iter,
                val_iter,
                args.device,
            )

            if args.wandb_log:
                import wandb
                wandb.log(
                    {
                        "Step": iter_num,
                        "Train Loss": losses["train"],
                        "Val Loss": losses["eval"],
                        "Learning Rate": lr,
                    }
                )

            # Save checkpoint if validation loss improved
            if losses["eval"] < best_val_loss:
                best_val_loss = losses["eval"]
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": model.config,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "training_args": vars(args),
                    }
                    torch.save(
                        checkpoint, os.path.join(args.out_dir, args.checkpoint_path)
                    )
                    print(f"Saved checkpoint at step {iter_num + 1}")

            print(
                f"step {iter_num + 1}: train loss: {losses['train']:.4f}, val loss: {losses['eval']:.4f}"
            )

        iter_num += 1


def get_wandb_config(args):
    """Get config for wandb logging"""
    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "use_fused_adamw": args.adamw_use_fused,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataset_name": args.dataset_name,
    }
    config.update(get_model_config(args.model_config_path))
    return config


def main(args):
    """Main entry point"""
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for DeepSeek")

    # Device and dtype
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # Learning rate
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--lr-decay-iters", type=int, default=1000)
    parser.add_argument("--decay-lr", type=bool, default=True)

    # Training
    parser.add_argument("--max-train-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Evaluation
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=10)

    # Wandb
    parser.add_argument("--wandb-log", type=bool, default=True)
    parser.add_argument("--wandb-project", type=str, default="deepseek sft")
    parser.add_argument("--wandb-run-name", type=str, default="sft_8_bit_optimizer_16_batch_size")

    # Optimizer
    parser.add_argument("--adamw-use-fused", type=bool, default=True)
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.95)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--use-eight-bit-optimizer", type=bool, default=True)

    # Checkpointing
    parser.add_argument("--out-dir", type=str, default="sft_output")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--checkpoint-path", type=str, default="sft_ckpt.pt")

    # Model and dataset
    parser.add_argument("--model-config-path", type=str, default="config.json")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--train-split", type=str, default="train_sft")
    parser.add_argument("--eval-split", type=str, default="test_sft")
    parser.add_argument("--tokenizer-type", type=str, default="cl100k_base")

    args = parser.parse_args()
    main(args)