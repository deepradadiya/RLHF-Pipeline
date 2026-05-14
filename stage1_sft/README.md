# Stage 1: Supervised Fine-Tuning (SFT) for Phi-3 Mini

This directory contains the implementation of Stage 1 of the RLHF pipeline: Supervised Fine-Tuning (SFT).

## What is SFT?

SFT (Supervised Fine-Tuning) is the first stage of RLHF where we take a pretrained base model and teach it to follow instructions using high-quality examples. Think of it like teaching a student using a textbook before giving them practice exams.

## Files Overview

### `dataset.py`
- Loads the "HuggingFaceH4/ultrachat_200k" dataset (train_sft split)
- Formats examples using Phi-3's chat template
- Validates data quality and filters problematic examples
- Prints dataset statistics
- **Colab Constraint**: Uses only first 10,000 examples to avoid memory issues

### `train_sft.py`
- Loads Phi-3 Mini with 4-bit quantization (saves VRAM on Colab)
- Applies LoRA (Low-Rank Adaptation) for efficient training
- Uses TRL's SFTTrainer with Colab-optimized hyperparameters
- Saves checkpoints to Google Drive every 100 steps
- Logs training metrics to Weights & Biases
- Uploads final model to HuggingFace Hub

### `test_sft.py`
- Loads both base model and fine-tuned SFT model
- Runs 5 test prompts comparing responses side-by-side
- Demonstrates that SFT improved instruction following

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers datasets peft trl bitsandbytes wandb
```

### 2. Run Training
```bash
cd stage1_sft
python train_sft.py --hf_username your_username
```

### 3. Test the Model
```bash
python test_sft.py --checkpoint_path ./sft_checkpoints
```

## Colab Constraints Handled

- **Memory**: 4-bit quantization + LoRA reduces memory usage
- **Batch Size**: Uses gradient_accumulation_steps=4 to simulate larger batches
- **Sequence Length**: Limited to 512 tokens to avoid OOM errors
- **Checkpointing**: Saves every 100 steps with Google Drive backup
- **Error Handling**: Try/catch around training loop with auto-save on crash

## Key Hyperparameters

- **LoRA Rank**: 16 (trains ~10M parameters instead of 3.8B)
- **Learning Rate**: 2e-4 (standard for LoRA fine-tuning)
- **Batch Size**: 1 per device × 4 accumulation = effective batch size 4
- **Max Length**: 512 tokens (T4 GPU constraint)
- **Epochs**: 1 (usually sufficient for SFT)

## Expected Results

After training, the SFT model should show:
- Better instruction following
- More helpful and structured responses
- Improved task completion
- Reduced refusal of reasonable requests

The test script will demonstrate these improvements by comparing base vs SFT model responses side-by-side.