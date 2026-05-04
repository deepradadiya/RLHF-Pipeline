"""
Hypothesis configuration for property-based testing.

This module configures Hypothesis settings for consistent and comprehensive
property-based testing across the RLHF Phi-3 Pipeline.
"""

from hypothesis import settings, Verbosity
from hypothesis.strategies import composite, integers, floats, text, lists, dictionaries, sampled_from, one_of, none

# Global Hypothesis settings
settings.register_profile("default", 
    max_examples=50,
    deadline=5000,  # 5 seconds per test
    verbosity=Verbosity.normal,
    suppress_health_check=[],
    report_multiple_bugs=True
)

settings.register_profile("fast", 
    max_examples=10,
    deadline=1000,  # 1 second per test
    verbosity=Verbosity.quiet
)

settings.register_profile("thorough", 
    max_examples=200,
    deadline=30000,  # 30 seconds per test
    verbosity=Verbosity.verbose,
    report_multiple_bugs=True
)

settings.register_profile("ci", 
    max_examples=100,
    deadline=10000,  # 10 seconds per test
    verbosity=Verbosity.normal,
    suppress_health_check=[],
    report_multiple_bugs=False  # Fail fast in CI
)

# Load the appropriate profile
import os
profile = os.getenv("HYPOTHESIS_PROFILE", "default")
settings.load_profile(profile)

# Common strategies for RLHF pipeline testing

# Text strategies
safe_text = text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .-_",
    min_size=1,
    max_size=100
)

model_names = sampled_from([
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/DialoGPT-small",
    "gpt2",
    "test-model",
    "mock/test-model"
])

file_paths = text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789/_.-",
    min_size=1,
    max_size=200
).filter(lambda x: x.strip() and not x.startswith('/') and '..' not in x)

# Numeric strategies with realistic bounds
learning_rates = floats(min_value=1e-6, max_value=1e-2, allow_nan=False, allow_infinity=False)
batch_sizes = integers(min_value=1, max_value=64)
epochs = integers(min_value=1, max_value=100)
steps = integers(min_value=1, max_value=10000)
lora_ranks = integers(min_value=1, max_value=256)
dropout_rates = floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Configuration strategies
device_types = sampled_from(["auto", "cpu", "cuda", "mps"])
optimizer_types = sampled_from(["adamw_torch", "adamw_hf", "sgd", "adafactor"])
scheduler_types = sampled_from(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"])
lora_bias_types = sampled_from(["none", "all", "lora_only"])
task_types = sampled_from(["CAUSAL_LM", "SEQ_CLS", "TOKEN_CLS"])
log_levels = sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

# Dataset strategies
@composite
def conversation_messages(draw):
    """Generate realistic conversation message structures."""
    num_turns = draw(integers(min_value=1, max_value=5))
    messages = []
    
    for i in range(num_turns):
        if i % 2 == 0:
            # User message
            messages.append({
                "role": "user",
                "content": draw(text(min_size=10, max_size=200))
            })
        else:
            # Assistant message
            messages.append({
                "role": "assistant", 
                "content": draw(text(min_size=20, max_size=500))
            })
    
    return messages

@composite
def preference_pairs(draw):
    """Generate preference data pairs."""
    prompt = draw(text(min_size=10, max_size=200))
    chosen = draw(text(min_size=20, max_size=500))
    rejected = draw(text(min_size=10, max_size=400))
    
    # Ensure chosen and rejected are different
    if chosen == rejected:
        rejected = rejected + " (different)"
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

# Evaluation strategies
mt_bench_scores = floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
performance_metrics = floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False)

@composite
def evaluation_results(draw):
    """Generate realistic evaluation result structures."""
    return {
        "mt_bench_score": draw(mt_bench_scores),
        "category_scores": {
            category: draw(mt_bench_scores)
            for category in ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]
        },
        "helpfulness_score": draw(mt_bench_scores),
        "harmlessness_score": draw(mt_bench_scores),
        "honesty_score": draw(mt_bench_scores),
        "tokens_per_second": draw(performance_metrics),
        "memory_usage_mb": draw(floats(min_value=100.0, max_value=16000.0)),
        "sample_responses": draw(lists(
            dictionaries(
                keys=sampled_from(["prompt", "response"]),
                values=text(min_size=10, max_size=200)
            ),
            min_size=1,
            max_size=10
        ))
    }

# Error and edge case strategies
@composite
def invalid_parameters(draw):
    """Generate invalid parameter combinations for testing validation."""
    param_type = draw(sampled_from([
        "negative_learning_rate",
        "zero_batch_size", 
        "negative_epochs",
        "invalid_dropout",
        "zero_rank",
        "empty_string",
        "too_large_value"
    ]))
    
    if param_type == "negative_learning_rate":
        return ("learning_rate", draw(floats(max_value=-1e-6)))
    elif param_type == "zero_batch_size":
        return ("batch_size", 0)
    elif param_type == "negative_epochs":
        return ("epochs", draw(integers(max_value=0)))
    elif param_type == "invalid_dropout":
        return ("dropout", draw(one_of(
            floats(max_value=-0.1),
            floats(min_value=1.1, max_value=2.0)
        )))
    elif param_type == "zero_rank":
        return ("rank", 0)
    elif param_type == "empty_string":
        return ("name", "")
    else:  # too_large_value
        return ("max_length", draw(integers(min_value=100000)))

# Memory and performance strategies
memory_sizes = integers(min_value=100, max_value=16000)  # MB
gpu_memory_percentages = floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Checkpoint and metadata strategies
@composite
def checkpoint_metadata(draw):
    """Generate checkpoint metadata structures."""
    stage = draw(sampled_from(["sft", "reward", "ppo"]))
    epoch = draw(integers(min_value=1, max_value=10))
    step = draw(integers(min_value=1, max_value=1000))
    
    return {
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "timestamp": "2024-01-01T00:00:00Z",  # Fixed for testing
        "model_path": f"/path/to/{stage}_model_epoch_{epoch}.bin",
        "optimizer_path": f"/path/to/{stage}_optimizer_epoch_{epoch}.pt",
        "config_hash": f"hash_{stage}_{epoch}_{step}",
        "metrics": {
            "loss": draw(floats(min_value=0.1, max_value=10.0)),
            "accuracy": draw(floats(min_value=0.0, max_value=1.0))
        }
    }

# Utility functions for property testing

def assume_valid_config_combination(config_dict):
    """Apply assumptions for valid configuration combinations."""
    from hypothesis import assume
    
    # Ensure effective batch sizes are reasonable
    if "training" in config_dict:
        for stage in ["sft", "reward", "ppo"]:
            if stage in config_dict["training"]:
                stage_config = config_dict["training"][stage]
                if "batch_size" in stage_config and "gradient_accumulation_steps" in stage_config:
                    effective_batch_size = stage_config["batch_size"] * stage_config["gradient_accumulation_steps"]
                    assume(effective_batch_size <= 128)
    
    # Ensure Phi-3 models have reasonable max_length
    if "model" in config_dict and "name" in config_dict["model"]:
        if "phi-3" in config_dict["model"]["name"].lower():
            if "max_length" in config_dict["model"]:
                assume(config_dict["model"]["max_length"] <= 4096)
    
    # Ensure checkpoint save_steps is reasonable
    if "checkpointing" in config_dict and "training" in config_dict:
        save_steps = config_dict["checkpointing"].get("save_steps", 100)
        min_max_steps = float('inf')
        
        for stage in ["sft", "reward", "ppo"]:
            if stage in config_dict["training"]:
                max_steps = config_dict["training"][stage].get("max_steps", 1000)
                min_max_steps = min(min_max_steps, max_steps)
        
        if min_max_steps != float('inf'):
            assume(save_steps <= min_max_steps)

def generate_realistic_bounds():
    """Generate realistic parameter bounds for validation testing."""
    return {
        "learning_rate": (1e-6, 1e-2),
        "batch_size": (1, 64),
        "epochs": (1, 100),
        "dropout": (0.0, 1.0),
        "lora_rank": (1, 256),
        "max_length": (1, 32768),
        "effective_batch_size": (1, 128)
    }

# Export commonly used strategies
__all__ = [
    "safe_text", "model_names", "file_paths",
    "learning_rates", "batch_sizes", "epochs", "steps", "lora_ranks", "dropout_rates",
    "device_types", "optimizer_types", "scheduler_types", "lora_bias_types", "task_types", "log_levels",
    "conversation_messages", "preference_pairs", "evaluation_results",
    "invalid_parameters", "memory_sizes", "gpu_memory_percentages", "checkpoint_metadata",
    "assume_valid_config_combination", "generate_realistic_bounds"
]