"""
Test utilities and helper functions for RLHF Phi-3 Pipeline tests.

This module provides common utilities, assertions, and helper functions
used across different test modules.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
import pytest

# Test environment utilities

@contextmanager
def temporary_directory():
    """Context manager for creating and cleaning up temporary directories."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@contextmanager
def temporary_file(suffix: str = ".tmp", content: Optional[str] = None):
    """Context manager for creating and cleaning up temporary files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        if content:
            f.write(content)
        temp_path = Path(f.name)
    
    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)

@contextmanager
def mock_environment_variables(**env_vars):
    """Context manager for temporarily setting environment variables."""
    original_values = {}
    
    # Store original values and set new ones
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = str(value)
    
    try:
        yield
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

# Configuration testing utilities

def assert_config_equality(config1: Dict[str, Any], config2: Dict[str, Any], path: str = ""):
    """Assert that two configuration dictionaries are equal with detailed error messages."""
    if isinstance(config1, dict) and isinstance(config2, dict):
        # Check all keys in config1
        for key in config1:
            current_path = f"{path}.{key}" if path else key
            assert key in config2, f"Key '{current_path}' missing in second config"
            assert_config_equality(config1[key], config2[key], current_path)
        
        # Check for extra keys in config2
        for key in config2:
            current_path = f"{path}.{key}" if path else key
            assert key in config1, f"Extra key '{current_path}' in second config"
    
    elif isinstance(config1, list) and isinstance(config2, list):
        assert len(config1) == len(config2), f"List length mismatch at '{path}': {len(config1)} vs {len(config2)}"
        for i, (item1, item2) in enumerate(zip(config1, config2)):
            assert_config_equality(item1, item2, f"{path}[{i}]")
    
    else:
        assert config1 == config2, f"Value mismatch at '{path}': {config1} != {config2}"

def validate_config_structure(config: Dict[str, Any], required_sections: List[str]) -> List[str]:
    """Validate that a configuration has all required sections and return any errors."""
    errors = []
    
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
        elif not isinstance(config[section], dict):
            errors.append(f"Section '{section}' must be a dictionary")
    
    return errors

def create_minimal_valid_config() -> Dict[str, Any]:
    """Create a minimal valid configuration for testing."""
    return {
        "model": {
            "name": "test-model",
            "max_length": 512,
            "device": "cpu"
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training": {
            "sft": {
                "epochs": 1,
                "learning_rate": 2e-4,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 10,
                "max_steps": 50
            },
            "reward": {
                "epochs": 1,
                "learning_rate": 1e-4,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 5,
                "max_steps": 25
            },
            "ppo": {
                "learning_rate": 1e-5,
                "batch_size": 1,
                "mini_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "ppo_epochs": 1,
                "max_steps": 10
            }
        },
        "optimization": {
            "optimizer_type": "adamw_torch",
            "scheduler_type": "linear",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "fp16": False,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0
        },
        "paths": {
            "base_output_dir": "/tmp/test_output",
            "cache_dir": "/tmp/test_cache",
            "logs_dir": "/tmp/test_logs"
        },
        "wandb": {
            "project": "test-project",
            "entity": None,
            "tags": ["test"]
        },
        "datasets": {
            "sft": {
                "name": "test-sft-dataset",
                "split": "train",
                "max_samples": 10
            },
            "preference": {
                "name": "test-preference-dataset",
                "split": "train",
                "max_samples": 5
            }
        },
        "evaluation": {
            "mt_bench": {
                "num_samples": 5,
                "temperature": 0.7,
                "max_new_tokens": 50
            }
        },
        "checkpointing": {
            "save_steps": 10,
            "save_total_limit": 2,
            "resume_from_checkpoint": None
        },
        "logging": {
            "level": "INFO",
            "log_steps": 5,
            "eval_steps": 10
        }
    }

# Dataset testing utilities

def create_mock_sft_dataset(size: int = 10) -> List[Dict[str, Any]]:
    """Create a mock SFT dataset for testing."""
    dataset = []
    for i in range(size):
        dataset.append({
            "messages": [
                {"role": "user", "content": f"Test question {i}?"},
                {"role": "assistant", "content": f"Test answer {i}."}
            ]
        })
    return dataset

def create_mock_preference_dataset(size: int = 5) -> List[Dict[str, Any]]:
    """Create a mock preference dataset for testing."""
    dataset = []
    for i in range(size):
        dataset.append({
            "prompt": f"Test prompt {i}",
            "chosen": f"Good response {i}",
            "rejected": f"Bad response {i}"
        })
    return dataset

def validate_dataset_format(dataset: List[Dict[str, Any]], dataset_type: str) -> List[str]:
    """Validate dataset format and return any errors."""
    errors = []
    
    if not isinstance(dataset, list):
        errors.append("Dataset must be a list")
        return errors
    
    if len(dataset) == 0:
        errors.append("Dataset cannot be empty")
        return errors
    
    for i, item in enumerate(dataset):
        if not isinstance(item, dict):
            errors.append(f"Item {i} must be a dictionary")
            continue
        
        if dataset_type == "sft":
            if "messages" not in item:
                errors.append(f"Item {i} missing 'messages' field")
            elif not isinstance(item["messages"], list):
                errors.append(f"Item {i} 'messages' must be a list")
            else:
                for j, message in enumerate(item["messages"]):
                    if not isinstance(message, dict):
                        errors.append(f"Item {i}, message {j} must be a dictionary")
                    elif "role" not in message or "content" not in message:
                        errors.append(f"Item {i}, message {j} missing 'role' or 'content'")
        
        elif dataset_type == "preference":
            required_fields = ["prompt", "chosen", "rejected"]
            for field in required_fields:
                if field not in item:
                    errors.append(f"Item {i} missing '{field}' field")
                elif not isinstance(item[field], str):
                    errors.append(f"Item {i} '{field}' must be a string")
    
    return errors

# Model testing utilities

class MockModel:
    """Mock model class for testing."""
    
    def __init__(self, model_name: str = "test-model"):
        self.model_name = model_name
        self.device = "cpu"
        self.training = False
        self.config = {"model_type": "test", "vocab_size": 1000}
        self.parameters_count = 1000000
    
    def to(self, device: str):
        self.device = device
        return self
    
    def train(self, mode: bool = True):
        self.training = mode
        return self
    
    def eval(self):
        self.training = False
        return self
    
    def parameters(self):
        """Mock parameters iterator."""
        import torch
        for _ in range(10):
            yield torch.randn(100, 100, requires_grad=True)
    
    def named_parameters(self):
        """Mock named parameters iterator."""
        import torch
        for i in range(10):
            yield f"layer_{i}.weight", torch.randn(100, 100, requires_grad=True)
    
    def state_dict(self):
        """Mock state dict."""
        import torch
        return {f"layer_{i}.weight": torch.randn(100, 100) for i in range(10)}
    
    def load_state_dict(self, state_dict):
        """Mock state dict loading."""
        pass
    
    def save_pretrained(self, path: str):
        """Mock model saving."""
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "config.json", 'w') as f:
            json.dump(self.config, f)

def assert_model_state_preserved(model_before, model_after, tolerance: float = 1e-6):
    """Assert that model state is preserved between operations."""
    state_before = model_before.state_dict()
    state_after = model_after.state_dict()
    
    assert set(state_before.keys()) == set(state_after.keys()), "Model parameter keys changed"
    
    for key in state_before.keys():
        param_before = state_before[key]
        param_after = state_after[key]
        
        assert param_before.shape == param_after.shape, f"Parameter {key} shape changed"
        
        # Check if parameters are close (for floating point comparison)
        import torch
        if torch.is_tensor(param_before) and torch.is_tensor(param_after):
            assert torch.allclose(param_before, param_after, atol=tolerance), f"Parameter {key} values changed"

# Checkpoint testing utilities

def create_mock_checkpoint_structure(base_dir: Path, stage: str, epoch: int = 1) -> Dict[str, Path]:
    """Create a mock checkpoint directory structure and return file paths."""
    checkpoint_dir = base_dir / f"{stage}_checkpoint_epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    files = {
        "model": checkpoint_dir / "pytorch_model.bin",
        "optimizer": checkpoint_dir / "optimizer.pt", 
        "trainer_state": checkpoint_dir / "trainer_state.json",
        "config": checkpoint_dir / "config.json",
        "metadata": checkpoint_dir / "metadata.json"
    }
    
    # Create mock files
    files["model"].write_text("mock model weights")
    files["optimizer"].write_text("mock optimizer state")
    files["trainer_state"].write_text(json.dumps({"epoch": epoch, "global_step": epoch * 100}))
    files["config"].write_text(json.dumps({"model_type": "test"}))
    
    metadata = {
        "stage": stage,
        "epoch": epoch,
        "step": epoch * 100,
        "timestamp": "2024-01-01T00:00:00Z",
        "model_path": str(files["model"]),
        "optimizer_path": str(files["optimizer"]),
        "config_hash": f"hash_{stage}_{epoch}",
        "metrics": {"loss": 2.5 - (epoch * 0.1), "accuracy": 0.6 + (epoch * 0.05)}
    }
    files["metadata"].write_text(json.dumps(metadata, indent=2))
    
    return files

def validate_checkpoint_integrity(checkpoint_dir: Path) -> List[str]:
    """Validate checkpoint directory integrity and return any errors."""
    errors = []
    
    if not checkpoint_dir.exists():
        errors.append(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return errors
    
    required_files = ["pytorch_model.bin", "optimizer.pt", "trainer_state.json", "config.json", "metadata.json"]
    
    for filename in required_files:
        file_path = checkpoint_dir / filename
        if not file_path.exists():
            errors.append(f"Missing required file: {filename}")
        elif file_path.stat().st_size == 0:
            errors.append(f"Empty file: {filename}")
    
    # Validate metadata format
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            required_metadata_fields = ["stage", "epoch", "step", "model_path", "optimizer_path", "config_hash"]
            for field in required_metadata_fields:
                if field not in metadata:
                    errors.append(f"Missing metadata field: {field}")
        
        except json.JSONDecodeError:
            errors.append("Invalid JSON in metadata.json")
    
    return errors

# Evaluation testing utilities

def create_mock_evaluation_results(score_range: tuple = (0.0, 10.0)) -> Dict[str, Any]:
    """Create mock evaluation results for testing."""
    import random
    
    min_score, max_score = score_range
    
    return {
        "mt_bench_score": random.uniform(min_score, max_score),
        "category_scores": {
            category: random.uniform(min_score, max_score)
            for category in ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]
        },
        "helpfulness_score": random.uniform(min_score, max_score),
        "harmlessness_score": random.uniform(min_score, max_score),
        "honesty_score": random.uniform(min_score, max_score),
        "tokens_per_second": random.uniform(1.0, 50.0),
        "memory_usage_mb": random.uniform(1000.0, 8000.0),
        "sample_responses": [
            {
                "prompt": f"Test prompt {i}",
                "response": f"Test response {i}"
            }
            for i in range(5)
        ]
    }

def validate_evaluation_results(results: Dict[str, Any]) -> List[str]:
    """Validate evaluation results format and return any errors."""
    errors = []
    
    required_fields = ["mt_bench_score", "category_scores", "helpfulness_score", "harmlessness_score", "honesty_score"]
    
    for field in required_fields:
        if field not in results:
            errors.append(f"Missing required field: {field}")
        elif field == "category_scores":
            if not isinstance(results[field], dict):
                errors.append(f"Field '{field}' must be a dictionary")
        else:
            if not isinstance(results[field], (int, float)):
                errors.append(f"Field '{field}' must be a number")
            elif not (0.0 <= results[field] <= 10.0):
                errors.append(f"Field '{field}' must be between 0.0 and 10.0")
    
    return errors

# Performance testing utilities

@contextmanager
def measure_execution_time():
    """Context manager to measure execution time."""
    import time
    start_time = time.time()
    try:
        yield lambda: time.time() - start_time
    finally:
        pass

def assert_execution_time_under(max_seconds: float):
    """Decorator to assert that a test completes within a time limit."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = test_func(*args, **kwargs)
                execution_time = time.time() - start_time
                assert execution_time < max_seconds, f"Test took {execution_time:.2f}s, expected < {max_seconds}s"
                return result
            except Exception:
                execution_time = time.time() - start_time
                if execution_time >= max_seconds:
                    pytest.fail(f"Test timed out after {execution_time:.2f}s (limit: {max_seconds}s)")
                raise
        return wrapper
    return decorator

# Mock utilities for external dependencies

@contextmanager
def mock_wandb():
    """Mock Weights & Biases for testing."""
    with patch('wandb.init') as mock_init, \
         patch('wandb.log') as mock_log, \
         patch('wandb.finish') as mock_finish:
        
        mock_run = Mock()
        mock_run.log = mock_log
        mock_run.finish = mock_finish
        mock_init.return_value = mock_run
        
        yield {
            'init': mock_init,
            'log': mock_log,
            'finish': mock_finish,
            'run': mock_run
        }

@contextmanager
def mock_huggingface_hub():
    """Mock HuggingFace Hub for testing."""
    with patch('huggingface_hub.HfApi') as mock_api, \
         patch('huggingface_hub.upload_folder') as mock_upload:
        
        mock_api_instance = Mock()
        mock_api.return_value = mock_api_instance
        
        yield {
            'api': mock_api,
            'api_instance': mock_api_instance,
            'upload_folder': mock_upload
        }

@contextmanager
def mock_google_drive():
    """Mock Google Drive integration for testing."""
    with patch('google.auth.default') as mock_auth, \
         patch('googleapiclient.discovery.build') as mock_build:
        
        mock_credentials = Mock()
        mock_auth.return_value = (mock_credentials, "test-project")
        
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        yield {
            'auth': mock_auth,
            'build': mock_build,
            'credentials': mock_credentials,
            'service': mock_service
        }

# Test data validation utilities

def assert_valid_json(data: Union[str, Dict, List]):
    """Assert that data is valid JSON or can be serialized to JSON."""
    if isinstance(data, str):
        try:
            json.loads(data)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON string: {e}")
    else:
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Data not JSON serializable: {e}")

def assert_no_sensitive_data(data: Union[str, Dict, List], sensitive_patterns: Optional[List[str]] = None):
    """Assert that data doesn't contain sensitive information."""
    if sensitive_patterns is None:
        sensitive_patterns = [
            "password", "secret", "key", "token", "credential",
            "sk-", "hf_", "ghp_", "gho_", "ghu_", "ghs_"
        ]
    
    data_str = json.dumps(data) if not isinstance(data, str) else data
    data_lower = data_str.lower()
    
    for pattern in sensitive_patterns:
        if pattern.lower() in data_lower:
            pytest.fail(f"Potentially sensitive data found: pattern '{pattern}' in data")

# Export all utilities
__all__ = [
    # Context managers
    "temporary_directory", "temporary_file", "mock_environment_variables",
    "mock_wandb", "mock_huggingface_hub", "mock_google_drive", "measure_execution_time",
    
    # Configuration utilities
    "assert_config_equality", "validate_config_structure", "create_minimal_valid_config",
    
    # Dataset utilities
    "create_mock_sft_dataset", "create_mock_preference_dataset", "validate_dataset_format",
    
    # Model utilities
    "MockModel", "assert_model_state_preserved",
    
    # Checkpoint utilities
    "create_mock_checkpoint_structure", "validate_checkpoint_integrity",
    
    # Evaluation utilities
    "create_mock_evaluation_results", "validate_evaluation_results",
    
    # Performance utilities
    "assert_execution_time_under",
    
    # Validation utilities
    "assert_valid_json", "assert_no_sensitive_data"
]