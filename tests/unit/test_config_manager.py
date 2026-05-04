"""
Unit tests for Configuration Manager component.

Tests the core functionality of the Config dataclass including validation,
serialization, and stage-specific configuration subsets.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from dataclasses import asdict

from rlhf_phi3.config.config_manager import (
    Config, ModelConfig, LoRAConfig, StageTrainingConfig, PPOTrainingConfig,
    TrainingConfig, OptimizationConfig, PathsConfig, WandBConfig, 
    DatasetConfig, DatasetsConfig, EvaluationConfig, CheckpointingConfig,
    LoggingConfig, load_default_config, load_colab_config
)


class TestModelConfig:
    """Test ModelConfig validation and functionality."""
    
    def test_default_model_config(self):
        """Test default model configuration is valid."""
        config = ModelConfig()
        assert config.name == "microsoft/Phi-3-mini-4k-instruct"
        assert config.max_length == 2048
        assert config.device == "auto"
        assert config.validate() == []
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Test empty name
        config = ModelConfig(name="")
        errors = config.validate()
        assert any("name cannot be empty" in error for error in errors)
        
        # Test invalid max_length
        config = ModelConfig(max_length=0)
        errors = config.validate()
        assert any("max_length must be positive" in error for error in errors)
        
        config = ModelConfig(max_length=50000)
        errors = config.validate()
        assert any("max_length should not exceed" in error for error in errors)
        
        # Test invalid device
        config = ModelConfig(device="invalid")
        errors = config.validate()
        assert any("Device must be one of" in error for error in errors)


class TestLoRAConfig:
    """Test LoRAConfig validation and functionality."""
    
    def test_default_lora_config(self):
        """Test default LoRA configuration is valid."""
        config = LoRAConfig()
        assert config.r == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
        assert config.validate() == []
    
    def test_lora_config_validation(self):
        """Test LoRA configuration validation."""
        # Test invalid rank
        config = LoRAConfig(r=0)
        errors = config.validate()
        assert any("rank (r) must be positive" in error for error in errors)
        
        config = LoRAConfig(r=300)
        errors = config.validate()
        assert any("rank (r) should not exceed" in error for error in errors)
        
        # Test invalid alpha
        config = LoRAConfig(alpha=0)
        errors = config.validate()
        assert any("alpha must be positive" in error for error in errors)
        
        # Test invalid dropout
        config = LoRAConfig(dropout=-0.1)
        errors = config.validate()
        assert any("dropout must be between" in error for error in errors)
        
        config = LoRAConfig(dropout=1.5)
        errors = config.validate()
        assert any("dropout must be between" in error for error in errors)
        
        # Test invalid bias
        config = LoRAConfig(bias="invalid")
        errors = config.validate()
        assert any("bias must be one of" in error for error in errors)
        
        # Test empty target modules
        config = LoRAConfig(target_modules=[])
        errors = config.validate()
        assert any("target_modules cannot be empty" in error for error in errors)


class TestStageTrainingConfig:
    """Test StageTrainingConfig validation and functionality."""
    
    def test_default_stage_config(self):
        """Test default stage training configuration is valid."""
        config = StageTrainingConfig()
        assert config.epochs == 1
        assert config.learning_rate == 2e-4
        assert config.validate() == []
    
    def test_stage_config_validation(self):
        """Test stage training configuration validation."""
        # Test invalid epochs
        config = StageTrainingConfig(epochs=0)
        errors = config.validate()
        assert any("Epochs must be positive" in error for error in errors)
        
        # Test invalid learning rate
        config = StageTrainingConfig(learning_rate=1e-7)
        errors = config.validate()
        assert any("Learning rate must be between" in error for error in errors)
        
        config = StageTrainingConfig(learning_rate=1e-1)
        errors = config.validate()
        assert any("Learning rate must be between" in error for error in errors)
        
        # Test invalid batch size
        config = StageTrainingConfig(batch_size=0)
        errors = config.validate()
        assert any("Batch size must be positive" in error for error in errors)
        
        config = StageTrainingConfig(batch_size=100)
        errors = config.validate()
        assert any("Batch size should not exceed" in error for error in errors)


class TestPPOTrainingConfig:
    """Test PPOTrainingConfig validation and functionality."""
    
    def test_default_ppo_config(self):
        """Test default PPO training configuration is valid."""
        config = PPOTrainingConfig()
        assert config.learning_rate == 1e-5
        assert config.batch_size == 1
        assert config.validate() == []
    
    def test_ppo_config_validation(self):
        """Test PPO training configuration validation."""
        # Test mini batch size larger than batch size
        config = PPOTrainingConfig(batch_size=2, mini_batch_size=4)
        errors = config.validate()
        assert any("mini batch size cannot exceed batch size" in error for error in errors)


class TestConfig:
    """Test main Config class functionality."""
    
    def test_default_config_is_valid(self):
        """Test that default configuration is valid."""
        config = Config()
        assert config.validate_config() is True
        assert config.get_validation_errors() == []
    
    def test_config_validation_with_errors(self):
        """Test configuration validation with errors."""
        config = Config()
        config.model.max_length = -1  # Invalid value
        
        assert config.validate_config() is False
        errors = config.get_validation_errors()
        assert len(errors) > 0
        assert any("max_length must be positive" in error for error in errors)
    
    def test_cross_section_validation(self):
        """Test cross-section validation logic."""
        config = Config()
        
        # Set very large effective batch size
        config.training.sft.batch_size = 64
        config.training.sft.gradient_accumulation_steps = 4  # 64 * 4 = 256 > 128
        
        errors = config.get_validation_errors()
        assert any("effective batch size" in error and "should not exceed 128" in error for error in errors)
    
    def test_get_stage_config_sft(self):
        """Test getting SFT stage configuration."""
        config = Config()
        sft_config = config.get_stage_config("sft")
        
        # Check that required sections are present
        assert "model" in sft_config
        assert "lora" in sft_config
        assert "training" in sft_config
        assert "dataset" in sft_config
        assert "optimization" in sft_config
        
        # Check that training config matches SFT stage
        assert sft_config["training"]["epochs"] == config.training.sft.epochs
        assert sft_config["training"]["learning_rate"] == config.training.sft.learning_rate
        
        # Check that dataset config matches SFT dataset
        assert sft_config["dataset"]["name"] == config.datasets.sft.name
    
    def test_get_stage_config_reward(self):
        """Test getting reward stage configuration."""
        config = Config()
        reward_config = config.get_stage_config("reward")
        
        # Check that training config matches reward stage
        assert reward_config["training"]["epochs"] == config.training.reward.epochs
        assert reward_config["training"]["learning_rate"] == config.training.reward.learning_rate
        
        # Check that dataset config matches preference dataset
        assert reward_config["dataset"]["name"] == config.datasets.preference.name
    
    def test_get_stage_config_ppo(self):
        """Test getting PPO stage configuration."""
        config = Config()
        ppo_config = config.get_stage_config("ppo")
        
        # Check that training config matches PPO stage
        assert ppo_config["training"]["learning_rate"] == config.training.ppo.learning_rate
        assert ppo_config["training"]["ppo_epochs"] == config.training.ppo.ppo_epochs
        
        # Check that both datasets are included for PPO
        assert "datasets" in ppo_config
        assert "sft" in ppo_config["datasets"]
        assert "preference" in ppo_config["datasets"]
    
    def test_get_stage_config_invalid_stage(self):
        """Test getting configuration for invalid stage."""
        config = Config()
        
        with pytest.raises(ValueError, match="Invalid stage"):
            config.get_stage_config("invalid")
    
    def test_config_serialization_yaml(self):
        """Test configuration serialization to YAML."""
        config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.save_config(temp_path, format="yaml")
            
            # Verify file was created and contains valid YAML
            assert temp_path.exists()
            with open(temp_path, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            assert "model" in loaded_data
            assert "training" in loaded_data
            assert loaded_data["model"]["name"] == config.model.name
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_config_serialization_json(self):
        """Test configuration serialization to JSON."""
        config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.save_config(temp_path, format="json")
            
            # Verify file was created and contains valid JSON
            assert temp_path.exists()
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert "model" in loaded_data
            assert "training" in loaded_data
            assert loaded_data["model"]["name"] == config.model.name
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_config_serialization_invalid_format(self):
        """Test configuration serialization with invalid format."""
        config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                config.save_config(temp_path, format="txt")
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_config_load_yaml(self):
        """Test loading configuration from YAML file."""
        original_config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save and then load configuration
            original_config.save_config(temp_path, format="yaml")
            loaded_config = Config.load_config(temp_path)
            
            # Verify loaded configuration matches original
            assert loaded_config.model.name == original_config.model.name
            assert loaded_config.training.sft.epochs == original_config.training.sft.epochs
            assert loaded_config.lora.r == original_config.lora.r
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_config_load_json(self):
        """Test loading configuration from JSON file."""
        original_config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save and then load configuration
            original_config.save_config(temp_path, format="json")
            loaded_config = Config.load_config(temp_path)
            
            # Verify loaded configuration matches original
            assert loaded_config.model.name == original_config.model.name
            assert loaded_config.training.sft.epochs == original_config.training.sft.epochs
            assert loaded_config.lora.r == original_config.lora.r
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_config_load_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            Config.load_config("/nonexistent/path/config.yaml")
    
    def test_config_load_unsupported_format(self):
        """Test loading configuration from unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
            f.write("some content")
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                Config.load_config(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_config_copy(self):
        """Test configuration deep copy."""
        original_config = Config()
        copied_config = original_config.copy()
        
        # Verify it's a different object
        assert copied_config is not original_config
        assert copied_config.model is not original_config.model
        
        # Verify values are the same
        assert copied_config.model.name == original_config.model.name
        assert copied_config.training.sft.epochs == original_config.training.sft.epochs
        
        # Verify changes to copy don't affect original
        copied_config.model.name = "different-model"
        assert original_config.model.name != copied_config.model.name
    
    def test_config_update_from_dict(self):
        """Test updating configuration from dictionary."""
        config = Config()
        original_name = config.model.name
        
        updates = {
            "model": {"name": "new-model-name"},
            "training": {"sft": {"epochs": 5}}
        }
        
        updated_config = config.update_from_dict(updates)
        
        # Verify original config is unchanged
        assert config.model.name == original_name
        
        # Verify updated config has changes
        assert updated_config.model.name == "new-model-name"
        assert updated_config.training.sft.epochs == 5
        
        # Verify other values are preserved
        assert updated_config.lora.r == config.lora.r


class TestConvenienceFunctions:
    """Test convenience functions for loading configurations."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_default_config()
        assert isinstance(config, Config)
        assert config.validate_config() is True
    
    def test_load_colab_config(self):
        """Test loading Colab-optimized configuration."""
        config = load_colab_config()
        assert isinstance(config, Config)
        assert config.validate_config() is True
        
        # Verify Colab-specific optimizations
        assert config.model.max_length == 1024  # Reduced for memory
        assert config.lora.r == 8  # Smaller rank
        assert config.training.sft.batch_size == 2  # Smaller batch size
        assert config.datasets.sft.max_samples == 2000  # Smaller dataset
        assert config.checkpointing.save_steps == 50  # More frequent saves


class TestDatasetConfig:
    """Test DatasetConfig validation and functionality."""
    
    def test_dataset_config_validation(self):
        """Test dataset configuration validation."""
        # Test empty name
        config = DatasetConfig(name="", split="train", max_samples=100)
        errors = config.validate()
        assert any("name cannot be empty" in error for error in errors)
        
        # Test empty split
        config = DatasetConfig(name="dataset", split="", max_samples=100)
        errors = config.validate()
        assert any("split cannot be empty" in error for error in errors)
        
        # Test invalid max_samples
        config = DatasetConfig(name="dataset", split="train", max_samples=0)
        errors = config.validate()
        assert any("Max samples must be positive" in error for error in errors)


class TestOptimizationConfig:
    """Test OptimizationConfig validation and functionality."""
    
    def test_optimization_config_validation(self):
        """Test optimization configuration validation."""
        # Test invalid optimizer
        config = OptimizationConfig(optimizer_type="invalid")
        errors = config.validate()
        assert any("Optimizer type must be one of" in error for error in errors)
        
        # Test invalid scheduler
        config = OptimizationConfig(scheduler_type="invalid")
        errors = config.validate()
        assert any("Scheduler type must be one of" in error for error in errors)
        
        # Test invalid weight decay
        config = OptimizationConfig(weight_decay=-0.1)
        errors = config.validate()
        assert any("Weight decay must be between" in error for error in errors)
        
        # Test invalid max grad norm
        config = OptimizationConfig(max_grad_norm=0)
        errors = config.validate()
        assert any("Max gradient norm must be positive" in error for error in errors)