"""
SFT (Supervised Fine-Tuning) training script for Phi-3 Mini.

What SFT means: We take a pretrained base model and teach it to follow instructions 
using high-quality examples. Think of it like teaching a student using a textbook 
before giving them practice exams.
"""

import os
import logging
import torch
from typing import Optional, Dict, Any
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset
import sys
sys.path.append('..')
from stage1_sft.dataset import prepare_sft_dataset

# Add colab helpers import (assuming it exists in the main package)
try:
    from notebooks.colab_utils import save_checkpoint_to_drive, setup_colab_environment
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False
    print("Colab utilities not available - running in local mode")

logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    use_4bit: bool = True
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Phi-3 Mini model with 4-bit quantization.
    
    Why 4-bit quantization: Saves VRAM on Colab T4 GPU (16GB -> ~4GB model size)
    This allows us to fit the 3.8B parameter model in limited GPU memory.
    
    Args:
        model_name: HuggingFace model identifier
        use_4bit: Whether to use 4-bit quantization (recommended for Colab)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model {model_name} with 4-bit quantization: {use_4bit}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization for memory efficiency
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Enable 4-bit loading
            bnb_4bit_quant_type="nf4",           # Use normalized float 4-bit
            bnb_4bit_compute_dtype=torch.float16, # Compute in float16 for speed
            bnb_4bit_use_double_quant=True,      # Double quantization for better accuracy
        )
        logger.info("Using 4-bit quantization to save VRAM")
    else:
        bnb_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",                       # Automatically distribute across available GPUs
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_4bit else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    # Enable gradient checkpointing to save memory during training
    model.gradient_checkpointing_enable()
    
    logger.info(f"Model loaded successfully. Memory footprint reduced with quantization.")
    return model, tokenizer


def setup_lora_config() -> LoraConfig:
    """
    Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    
    What LoRA does: Instead of training all 3.8B parameters, we only train ~10M 
    adapter parameters. This dramatically reduces memory usage and training time
    while maintaining most of the performance benefits.
    
    Returns:
        LoRA configuration
    """
    # Target modules for Phi-3 architecture
    # These are the attention and MLP layers where LoRA adapters will be applied
    target_modules = [
        "q_proj",    # Query projection in attention
        "k_proj",    # Key projection in attention  
        "v_proj",    # Value projection in attention
        "o_proj",    # Output projection in attention
        "gate_proj", # Gate projection in MLP
        "up_proj",   # Up projection in MLP
        "down_proj", # Down projection in MLP
    ]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,    # Causal language modeling task
        r=16,                            # LoRA rank - higher = more parameters but better performance
        lora_alpha=32,                   # LoRA scaling parameter (typically 2x rank)
        lora_dropout=0.1,                # Dropout for regularization
        target_modules=target_modules,    # Which modules to apply LoRA to
        bias="none",                     # Don't adapt bias terms
        inference_mode=False,            # Training mode
    )
    
    logger.info(f"LoRA config: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
    logger.info(f"Target modules: {target_modules}")
    return lora_config


def setup_training_arguments(
    output_dir: str = "./sft_checkpoints",
    num_train_epochs: int = 1,           # 1 epoch is often sufficient for SFT
    per_device_train_batch_size: int = 1, # Small batch size for T4 GPU
    gradient_accumulation_steps: int = 4,  # Simulate larger batch size (effective batch = 1*4=4)
    learning_rate: float = 2e-4,         # Standard learning rate for LoRA fine-tuning
    max_seq_length: int = 512,           # Shorter sequences to avoid OOM on T4
    save_steps: int = 100,               # Save checkpoint every 100 steps for Colab
    logging_steps: int = 10,             # Log metrics every 10 steps
    warmup_steps: int = 100,             # Learning rate warmup
) -> TrainingArguments:
    """
    Configure training arguments optimized for Google Colab T4 GPU.
    
    Args:
        output_dir: Directory to save checkpoints
        num_train_epochs: Number of training epochs (1 is usually enough for SFT)
        per_device_train_batch_size: Batch size per device (1 for T4 memory constraints)
        gradient_accumulation_steps: Steps to accumulate gradients (simulates larger batch)
        learning_rate: Learning rate for AdamW optimizer
        max_seq_length: Maximum sequence length (512 to avoid OOM errors)
        save_steps: Save checkpoint every N steps (100 for frequent Colab saves)
        logging_steps: Log metrics every N steps
        warmup_steps: Number of warmup steps for learning rate scheduler
    
    Returns:
        TrainingArguments object
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Effective batch size = 1*4 = 4
        learning_rate=learning_rate,
        weight_decay=0.01,                    # L2 regularization
        logging_steps=logging_steps,
        save_steps=save_steps,                # Frequent saves for Colab session management
        save_total_limit=3,                   # Keep only 3 most recent checkpoints
        prediction_loss_only=True,            # Only compute loss (faster)
        remove_unused_columns=False,          # Keep all columns for SFTTrainer
        dataloader_pin_memory=False,          # Disable for memory efficiency
        dataloader_num_workers=0,             # Single worker to avoid multiprocessing issues
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",           # Cosine learning rate decay
        optim="adamw_torch",                  # AdamW optimizer
        bf16=torch.cuda.is_bf16_supported(),  # Use bfloat16 if supported
        fp16=not torch.cuda.is_bf16_supported(), # Fallback to float16
        gradient_checkpointing=True,          # Save memory during backprop
        report_to="wandb",                    # Log to Weights & Biases
        run_name=f"phi3-sft-ultrachat",      # W&B run name
        seed=42,                             # Reproducibility
    )
    
    logger.info(f"Training arguments configured:")
    logger.info(f"  Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Max sequence length: {max_seq_length}")
    logger.info(f"  Save every {save_steps} steps")
    
    return training_args


def train_sft_model(
    dataset: Dataset,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    lora_config: LoraConfig,
    max_seq_length: int = 512
) -> SFTTrainer:
    """
    Train the SFT model using TRL's SFTTrainer.
    
    Args:
        dataset: Prepared training dataset
        model: Quantized Phi-3 model
        tokenizer: Phi-3 tokenizer
        training_args: Training configuration
        lora_config: LoRA configuration
        max_seq_length: Maximum sequence length for training
    
    Returns:
        Trained SFTTrainer instance
    """
    logger.info("Setting up SFT training...")
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Show how many parameters we're actually training
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not masked language modeling
        pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
    )
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_text_field="text",        # Field containing the formatted chat text
        max_seq_length=max_seq_length,    # Truncate long sequences
        packing=False,                    # Don't pack multiple examples together
        dataset_kwargs={
            "add_special_tokens": False,  # Special tokens already in chat template
            "append_concat_token": False, # Don't add extra tokens
        }
    )
    
    # Custom callback for Google Drive checkpoint saving
    if COLAB_AVAILABLE:
        class ColabCheckpointCallback:
            def __init__(self, save_every_n_steps=100):
                self.save_every_n_steps = save_every_n_steps
            
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % self.save_every_n_steps == 0:
                    try:
                        save_checkpoint_to_drive(
                            checkpoint_dir=args.output_dir,
                            step=state.global_step
                        )
                        logger.info(f"Checkpoint saved to Google Drive at step {state.global_step}")
                    except Exception as e:
                        logger.warning(f"Failed to save to Google Drive: {e}")
        
        trainer.add_callback(ColabCheckpointCallback())
    
    return trainer


def main(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    output_dir: str = "./sft_checkpoints",
    max_samples: int = 10000,  # Colab limitation: use only first 10k examples
    max_seq_length: int = 512, # Colab T4 limitation: shorter sequences to avoid OOM
    hf_username: Optional[str] = None,  # For HuggingFace Hub upload
    wandb_project: str = "phi3-sft-training"
):
    """
    Main training function for SFT stage.
    
    Args:
        model_name: HuggingFace model to fine-tune
        dataset_name: HuggingFace dataset for training
        output_dir: Directory to save checkpoints
        max_samples: Maximum training samples (10k for Colab)
        max_seq_length: Maximum sequence length (512 for T4 GPU)
        hf_username: HuggingFace username for model upload
        wandb_project: Weights & Biases project name
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting SFT training for Phi-3 Mini")
    
    # Initialize Weights & Biases
    wandb.init(
        project=wandb_project,
        name="phi3-sft-ultrachat",
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "max_samples": max_samples,
            "max_seq_length": max_seq_length,
        }
    )
    
    try:
        # Setup Colab environment if available
        if COLAB_AVAILABLE:
            setup_colab_environment()
        
        # Prepare dataset
        logger.info("Preparing SFT dataset...")
        dataset, tokenizer, stats = prepare_sft_dataset(
            dataset_name=dataset_name,
            model_name=model_name,
            max_samples=max_samples,
            max_length=max_seq_length
        )
        
        # Load model with quantization
        logger.info("Loading model with 4-bit quantization...")
        model, tokenizer = setup_model_and_tokenizer(model_name, use_4bit=True)
        
        # Setup LoRA configuration
        lora_config = setup_lora_config()
        
        # Setup training arguments
        training_args = setup_training_arguments(
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            save_steps=100  # Save every 100 steps for Colab
        )
        
        # Initialize trainer
        trainer = train_sft_model(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            lora_config=lora_config,
            max_seq_length=max_seq_length
        )
        
        # Start training with error handling for Colab crashes
        logger.info("Starting training...")
        try:
            trainer.train()
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training interrupted: {e}")
            # Auto-save current state if Colab crashes
            logger.info("Attempting to save current training state...")
            trainer.save_model(f"{output_dir}/emergency_checkpoint")
            if COLAB_AVAILABLE:
                try:
                    save_checkpoint_to_drive(f"{output_dir}/emergency_checkpoint", "emergency")
                except:
                    pass
            raise
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model(f"{output_dir}/final_model")
        
        # Merge LoRA adapters and save merged model
        logger.info("Merging LoRA adapters...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(f"{output_dir}/merged_model")
        tokenizer.save_pretrained(f"{output_dir}/merged_model")
        
        # Upload to HuggingFace Hub if username provided
        if hf_username:
            repo_name = f"{hf_username}/phi3-sft-ultrachat"
            logger.info(f"Uploading merged model to HuggingFace Hub: {repo_name}")
            
            try:
                merged_model.push_to_hub(
                    repo_name,
                    commit_message="SFT fine-tuned Phi-3 Mini on UltraChat 200k",
                    private=False
                )
                tokenizer.push_to_hub(repo_name)
                logger.info(f"Model successfully uploaded to {repo_name}")
                
            except Exception as e:
                logger.error(f"Failed to upload to HuggingFace Hub: {e}")
                logger.info("Model saved locally. You can upload manually later.")
        
        # Log final metrics to W&B
        wandb.log({
            "training_completed": True,
            "final_loss": trainer.state.log_history[-1].get("train_loss", 0),
            "total_steps": trainer.state.global_step,
        })
        
        logger.info("SFT training pipeline completed successfully!")
        return trainer
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        wandb.log({"training_failed": True, "error": str(e)})
        raise
    
    finally:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SFT model for Phi-3")
    parser.add_argument("--model_name", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset_name", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--output_dir", default="./sft_checkpoints")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--hf_username", help="HuggingFace username for model upload")
    parser.add_argument("--wandb_project", default="phi3-sft-training")
    
    args = parser.parse_args()
    
    main(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        hf_username=args.hf_username,
        wandb_project=args.wandb_project
    )