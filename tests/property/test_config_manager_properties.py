"""
Property-based tests for Configuration Manager component.

This module implements property-based tests using Hypothesis to validate
the correctness properties of the Configuration Manager across all possible
valid inputs and edge cases.

Properties tested:
- Property 17: Configuration Serialization Round-Trip (Requirements 8.1, 8.4)
- Property 18: Configuration Validation Accuracy (Requirement 8.2)
- Property 19: Stage Configuration Subsetting (Requirement 8.3)
- Property 20: Parameter Bounds Enforcement (Requirement 8.5)
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict

from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite

from rlhf_phi3.config.config_manager import (
    Config, ModelConfig, LoRAConfig, StageTrainingConfig, PPOTrainingConfig,
    TrainingConfig, OptimizationConfig, PathsConfig, WandBConfig, 
    DatasetConfig, DatasetsConfig, EvaluationConfig, CheckpointingConfig,
    LoggingConfig
)


# Hypothesis strategies for generating valid configuration data

@composite
def valid_model_config(draw):
    """Generate valid ModelConfig instances."""
    return ModelConfig(
        name=draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip())),
        max_length=draw(st.integers(min_value=1, max_value=32768)),
        device=draw(st.sampled_from(["auto", "cpu", "cuda", "mps"]))
    )

@composite
def valid_lora_config(draw):
    """Generate valid LoRAConfig instances."""
    return LoRAConfig(
        r=draw(st.integers(min_value=1, max_value=256)),
        alpha=draw(st.integers(min_value=1, max_value=512)),
        dropout=draw(st.floats(min_value=0.0, max_value=1.0)),
        target_modules=draw(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10)),
        bias=draw(st.sampled_from(["none", "all", "lora_only"])),
        task_type=draw(st.sampled_from(["CAUSAL_LM", "SEQ_CLS", "TOKEN_CLS"]))
    )

@composite
def valid_stage_training_config(draw):
    """Generate valid StageTrainingConfig instances."""
    return StageTrainingConfig(
        epochs=draw(st.integers(min_value=1, max_value=100)),
        learning_rate=draw(st.floats(min_value=1e-6, max_value=1e-2)),
        batch_size=draw(st.integers(min_value=1, max_value=64)),
        gradient_accumulation_steps=draw(st.integers(min_value=1, max_value=32)),
        warmup_steps=draw(st.integers(min_value=0, max_value=1000)),
        max_steps=draw(st.integers(min_value=1, max_value=10000))
    )

@composite
def valid_ppo_training_config(draw):
    """Generate valid PPOTrainingConfig instances."""
    batch_size = draw(st.integers(min_value=1, max_value=64))
    mini_batch_size = draw(st.integers(min_value=1, max_value=batch_size))
    
    return PPOTrainingConfig(
        learning_rate=draw(st.floats(min_value=1e-6, max_value=1e-2)),
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=draw(st.integers(min_value=1, max_value=32)),
        ppo_epochs=draw(st.integers(min_value=1, max_value=10)),
        max_steps=draw(st.integers(min_value=1, max_value=10000))
    )

@composite
def valid_optimization_config(draw):
    """Generate valid OptimizationConfig instances."""
    return OptimizationConfig(
        optimizer_type=draw(st.sampled_from(["adamw_torch", "adamw_hf", "sgd", "adafactor"])),
        scheduler_type=draw(st.sampled_from(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"])),
        weight_decay=draw(st.floats(min_value=0.0, max_value=1.0)),
        max_grad_norm=draw(st.floats(min_value=0.1, max_value=10.0)),
        fp16=draw(st.booleans()),
        gradient_checkpointing=draw(st.booleans()),
        dataloader_num_workers=draw(st.integers(min_value=0, max_value=8))
    )

@composite
def valid_paths_config(draw):
    """Generate valid PathsConfig instances."""
    return PathsConfig(
        base_output_dir=draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip())),
        cache_dir=draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip())),
        logs_dir=draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    )

@composite
def valid_wandb_config(draw):
    """Generate valid WandBConfig instances."""
    return WandBConfig(
        project=draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip())),
        entity=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        tags=draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=10))
    )

@composite
def valid_dataset_config(draw):
    """Generate valid DatasetConfig instances."""
    return DatasetConfig(
        name=draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip())),
        split=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        max_samples=draw(st.integers(min_value=1, max_value=100000))
    )

@composite
def valid_evaluation_config(draw):
    """Generate valid EvaluationConfig instances."""
    return EvaluationConfig(
        mt_bench={
            "num_samples": draw(st.integers(min_value=1, max_value=1000)),
            "temperature": draw(st.floats(min_value=0.0, max_value=2.0)),
            "max_new_tokens": draw(st.integers(min_value=1, max_value=2048))
        }
    )

@composite
def valid_checkpointing_config(draw):
    """Generate valid CheckpointingConfig instances."""
    return CheckpointingConfig(
        save_steps=draw(st.integers(min_value=1, max_value=1000)),
        save_total_limit=draw(st.integers(min_value=1, max_value=10)),
        resume_from_checkpoint=draw(st.one_of(st.none(), st.text(min_size=1, max_size=200)))
    )

@composite
def valid_logging_config(draw):
    """Generate valid LoggingConfig instances."""
    return LoggingConfig(
        level=draw(st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])),
        log_steps=draw(st.integers(min_value=1, max_value=1000)),
        eval_steps=draw(st.integers(min_value=1, max_value=1000))
    )

@composite
def valid_config(draw):
    """Generate valid Config instances."""
    sft_config = draw(valid_stage_training_config())
    reward_config = draw(valid_stage_training_config())
    ppo_config = draw(valid_ppo_training_config())
    
    # Ensure effective batch sizes don't exceed 128 for cross-section validation
    sft_effective = sft_config.batch_size * sft_config.gradient_accumulation_steps
    reward_effective = reward_config.batch_size * reward_config.gradient_accumulation_steps
    ppo_effective = ppo_config.batch_size * ppo_config.gradient_accumulation_steps
    
    assume(sft_effective <= 128)
    assume(reward_effective <= 128)
    assume(ppo_effective <= 128)
    
    sft_dataset = draw(valid_dataset_config())
    preference_dataset = draw(valid_dataset_config())
    
    model_config = draw(valid_model_config())
    
    # Ensure Phi-3 models have reasonable max_length
    if "phi-3" in model_config.name.lower():
        assume(model_config.max_length <= 4096)
    
    checkpointing_config = draw(valid_checkpointing_config())
    
    # Ensure checkpoint save_steps is reasonable compared to training steps
    min_max_steps = min(sft_config.max_steps, reward_config.max_steps, ppo_config.max_steps)
    assume(checkpointing_config.save_steps <= min_max_steps)
    
    return Config(
        model=model_config,
        lora=draw(valid_lora_config()),
        training=TrainingConfig(sft=sft_config, reward=reward_config, ppo=ppo_config),
        optimization=draw(valid_optimization_config()),
        paths=draw(valid_paths_config()),
        wandb=draw(valid_wandb_config()),
        datasets=DatasetsConfig(sft=sft_dataset, preference=preference_dataset),
        evaluation=draw(valid_evaluation_config()),
        checkpointing=checkpointing_config,
        logging=draw(valid_logging_config())
    )

# Strategies for generating invalid configurations for validation testing

@composite
def invalid_model_config(draw):
    """Generate invalid ModelConfig instances."""
    choice = draw(st.integers(min_value=0, max_value=3))
    
    if choice == 0:
        # Empty name
        return ModelConfig(name="", max_length=2048, device="auto")
    elif choice == 1:
        # Invalid max_length
        return ModelConfig(name="test", max_length=draw(st.integers(max_value=0)), device="auto")
    elif choice == 2:
        # Too large max_length
        return ModelConfig(name="test", max_length=draw(st.integers(min_value=50000)), device="auto")
    else:
        # Invalid device
        return ModelConfig(name="test", max_length=2048, device="invalid_device")

@composite
def invalid_lora_config(draw):
    """Generate invalid LoRAConfig instances."""
    choice = draw(st.integers(min_value=0, max_value=5))
    
    if choice == 0:
        # Invalid rank
        return LoRAConfig(r=draw(st.integers(max_value=0)))
    elif choice == 1:
        # Too large rank
        return LoRAConfig(r=draw(st.integers(min_value=300)))
    elif choice == 2:
        # Invalid alpha
        return LoRAConfig(alpha=draw(st.integers(max_value=0)))
    elif choice == 3:
        # Invalid dropout
        return LoRAConfig(dropout=draw(st.floats(min_value=1.1, max_value=2.0)))
    elif choice == 4:
        # Invalid bias
        return LoRAConfig(bias="invalid_bias")
    else:
        # Empty target modules
        return LoRAConfig(target_modules=[])

@composite
def invalid_stage_training_config(draw):
    """Generate invalid StageTrainingConfig instances."""
    choice = draw(st.integers(min_value=0, max_value=4))
    
    if choice == 0:
        # Invalid epochs
        return StageTrainingConfig(epochs=draw(st.integers(max_value=0)))
    elif choice == 1:
        # Invalid learning rate (too low)
        return StageTrainingConfig(learning_rate=draw(st.floats(min_value=1e-8, max_value=1e-7)))
    elif choice == 2:
        # Invalid learning rate (too high)
        return StageTrainingConfig(learning_rate=draw(st.floats(min_value=1e-1, max_value=1.0)))
    elif choice == 3:
        # Invalid batch size
        return StageTrainingConfig(batch_size=draw(st.integers(max_value=0)))
    else:
        # Too large batch size
        return StageTrainingConfig(batch_size=draw(st.integers(min_value=100)))

@composite
def out_of_bounds_parameter(draw):
    """Generate parameter values that are out of bounds."""
    param_type = draw(st.sampled_from([
        "learning_rate_low", "learning_rate_high", "batch_size_zero", "batch_size_large",
        "dropout_negative", "dropout_large", "rank_zero", "rank_large"
    ]))
    
    if param_type == "learning_rate_low":
        return ("learning_rate", draw(st.floats(min_value=1e-8, max_value=1e-7)))
    elif param_type == "learning_rate_high":
        return ("learning_rate", draw(st.floats(min_value=1e-1, max_value=1.0)))
    elif param_type == "batch_size_zero":
        return ("batch_size", 0)
    elif param_type == "batch_size_large":
        return ("batch_size", draw(st.integers(min_value=100, max_value=1000)))
    elif param_type == "dropout_negative":
        return ("dropout", draw(st.floats(min_value=-1.0, max_value=-0.1)))
    elif param_type == "dropout_large":
        return ("dropout", draw(st.floats(min_value=1.1, max_value=2.0)))
    elif param_type == "rank_zero":
        return ("rank", 0)
    else:  # rank_large
        return ("rank", draw(st.integers(min_value=300, max_value=1000)))


class TestConfigurationManagerProperties:
    """Property-based tests for Configuration Manager correctness properties."""
    
    @given(valid_config())
    @settings(max_examples=50, deadline=None)
    def test_property_17_configuration_serialization_round_trip(self, config: Config):
        """
        **Validates: Requirements 8.1, 8.4**
        
        Property 17: Configuration Serialization Round-Trip
        
        For any hyperparameter combination, the Configuration_Manager SHALL maintain 
        serializable configuration objects with perfect round-trip preservation.
        
        This property ensures that:
        1. Any valid configuration can be serialized to JSON/YAML
        2. The serialized configuration can be loaded back
        3. The loaded configuration is identical to the original
        4. All nested dataclass structures are preserved
        """
        # Test JSON round-trip
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = Path(f.name)
        
        try:
            # Save to JSON
            config.save_config(json_path, format="json")
            
            # Load from JSON
            loaded_json_config = Config.load_config(json_path)
            
            # Verify round-trip preservation
            assert asdict(config) == asdict(loaded_json_config)
            assert config.model.name == loaded_json_config.model.name
            assert config.lora.r == loaded_json_config.lora.r
            assert config.training.sft.learning_rate == loaded_json_config.training.sft.learning_rate
            assert config.datasets.sft.name == loaded_json_config.datasets.sft.name
            
        finally:
            json_path.unlink(missing_ok=True)
        
        # Test YAML round-trip (if available)
        try:
            import yaml
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml_path = Path(f.name)
            
            try:
                # Save to YAML
                config.save_config(yaml_path, format="yaml")
                
                # Load from YAML
                loaded_yaml_config = Config.load_config(yaml_path)
                
                # Verify round-trip preservation
                assert asdict(config) == asdict(loaded_yaml_config)
                assert config.model.name == loaded_yaml_config.model.name
                assert config.lora.r == loaded_yaml_config.lora.r
                assert config.training.sft.learning_rate == loaded_yaml_config.training.sft.learning_rate
                
            finally:
                yaml_path.unlink(missing_ok=True)
                
        except ImportError:
            # YAML not available, skip YAML test
            pass
    
    @given(st.one_of(
        invalid_model_config(),
        invalid_lora_config(), 
        invalid_stage_training_config()
    ))
    @settings(max_examples=30, deadline=None)
    def test_property_18_configuration_validation_accuracy_invalid(self, invalid_config):
        """
        **Validates: Requirement 8.2**
        
        Property 18: Configuration Validation Accuracy (Invalid Cases)
        
        For any parameter combination (valid or invalid), the Configuration_Manager 
        SHALL validate configuration consistency and flag invalid parameter combinations.
        
        This test verifies that invalid configurations are correctly detected.
        """
        # Create a config with the invalid component
        config = Config()
        
        if isinstance(invalid_config, ModelConfig):
            config.model = invalid_config
        elif isinstance(invalid_config, LoRAConfig):
            config.lora = invalid_config
        elif isinstance(invalid_config, StageTrainingConfig):
            config.training.sft = invalid_config
        
        # Validation should fail
        assert config.validate_config() is False
        
        # Should have validation errors
        errors = config.get_validation_errors()
        assert len(errors) > 0
        assert all(isinstance(error, str) for error in errors)
    
    @given(valid_config())
    @settings(max_examples=30, deadline=None)
    def test_property_18_configuration_validation_accuracy_valid(self, config: Config):
        """
        **Validates: Requirement 8.2**
        
        Property 18: Configuration Validation Accuracy (Valid Cases)
        
        This test verifies that valid configurations pass validation.
        """
        # Validation should pass
        assert config.validate_config() is True
        
        # Should have no validation errors
        errors = config.get_validation_errors()
        assert len(errors) == 0
    
    @given(valid_config(), st.sampled_from(["sft", "reward", "ppo"]))
    @settings(max_examples=50, deadline=None)
    def test_property_19_stage_configuration_subsetting(self, config: Config, stage: str):
        """
        **Validates: Requirement 8.3**
        
        Property 19: Stage Configuration Subsetting
        
        For any full configuration, the Configuration_Manager SHALL provide 
        stage-specific configuration subsets containing exactly the correct 
        parameters for modular training.
        
        This property ensures that:
        1. Stage configs contain all required sections
        2. Training parameters match the specific stage
        3. Dataset configuration is appropriate for the stage
        4. All other shared configurations are included
        """
        stage_config = config.get_stage_config(stage)
        
        # Verify required sections are present
        required_sections = ["model", "lora", "training", "optimization", "paths", 
                           "checkpointing", "logging", "wandb"]
        for section in required_sections:
            assert section in stage_config, f"Missing required section: {section}"
        
        # Verify training configuration matches the stage
        if stage == "sft":
            assert stage_config["training"]["epochs"] == config.training.sft.epochs
            assert stage_config["training"]["learning_rate"] == config.training.sft.learning_rate
            assert stage_config["training"]["batch_size"] == config.training.sft.batch_size
            
            # SFT should have single dataset config
            assert "dataset" in stage_config
            assert stage_config["dataset"]["name"] == config.datasets.sft.name
            assert "datasets" not in stage_config
            
        elif stage == "reward":
            assert stage_config["training"]["epochs"] == config.training.reward.epochs
            assert stage_config["training"]["learning_rate"] == config.training.reward.learning_rate
            assert stage_config["training"]["batch_size"] == config.training.reward.batch_size
            
            # Reward should have preference dataset
            assert "dataset" in stage_config
            assert stage_config["dataset"]["name"] == config.datasets.preference.name
            assert "datasets" not in stage_config
            
        elif stage == "ppo":
            assert stage_config["training"]["learning_rate"] == config.training.ppo.learning_rate
            assert stage_config["training"]["batch_size"] == config.training.ppo.batch_size
            assert stage_config["training"]["ppo_epochs"] == config.training.ppo.ppo_epochs
            
            # PPO should have both datasets
            assert "datasets" in stage_config
            assert "sft" in stage_config["datasets"]
            assert "preference" in stage_config["datasets"]
            assert stage_config["datasets"]["sft"]["name"] == config.datasets.sft.name
            assert stage_config["datasets"]["preference"]["name"] == config.datasets.preference.name
            assert "dataset" not in stage_config
        
        # Verify shared configurations are identical
        assert stage_config["model"]["name"] == config.model.name
        assert stage_config["lora"]["r"] == config.lora.r
        assert stage_config["optimization"]["optimizer_type"] == config.optimization.optimizer_type
        assert stage_config["paths"]["base_output_dir"] == config.paths.base_output_dir
    
    @given(out_of_bounds_parameter())
    @settings(max_examples=30, deadline=None)
    def test_property_20_parameter_bounds_enforcement(self, param_info):
        """
        **Validates: Requirement 8.5**
        
        Property 20: Parameter Bounds Enforcement
        
        For any parameter value, the Configuration_Manager SHALL enforce 
        parameter bounds correctly (e.g., learning rate between 1e-6 and 1e-2).
        
        This property ensures that:
        1. Learning rates outside [1e-6, 1e-2] are rejected
        2. Batch sizes <= 0 or > 64 are rejected  
        3. Dropout values outside [0.0, 1.0] are rejected
        4. LoRA ranks <= 0 or > 256 are rejected
        """
        param_name, param_value = param_info
        config = Config()
        
        # Apply the out-of-bounds parameter
        if param_name == "learning_rate":
            config.training.sft.learning_rate = param_value
        elif param_name == "batch_size":
            config.training.sft.batch_size = param_value
        elif param_name == "dropout":
            config.lora.dropout = param_value
        elif param_name == "rank":
            config.lora.r = param_value
        
        # Validation should fail due to parameter bounds
        assert config.validate_config() is False
        
        # Should have specific validation errors related to bounds
        errors = config.get_validation_errors()
        assert len(errors) > 0
        
        # Verify the error message mentions the parameter bounds
        error_text = " ".join(errors).lower()
        if param_name == "learning_rate":
            assert "learning rate" in error_text and "between" in error_text
        elif param_name == "batch_size":
            assert "batch size" in error_text and ("positive" in error_text or "exceed" in error_text)
        elif param_name == "dropout":
            assert "dropout" in error_text and "between" in error_text
        elif param_name == "rank":
            assert ("rank" in error_text or "r" in error_text) and ("positive" in error_text or "exceed" in error_text)
    
    @given(valid_config())
    @settings(max_examples=20, deadline=None)
    def test_configuration_copy_independence(self, config: Config):
        """
        Additional property: Configuration copies should be independent.
        
        This ensures that modifications to copied configurations don't affect
        the original configuration.
        """
        copied_config = config.copy()
        
        # Verify it's a different object
        assert copied_config is not config
        assert copied_config.model is not config.model
        assert copied_config.training is not config.training
        
        # Verify values are initially the same
        assert copied_config.model.name == config.model.name
        assert copied_config.lora.r == config.lora.r
        
        # Modify the copy
        copied_config.model.name = "modified-model"
        copied_config.lora.r = 999
        
        # Original should be unchanged
        assert config.model.name != "modified-model"
        assert config.lora.r != 999
    
    @given(valid_config(), st.dictionaries(
        st.sampled_from(["model", "lora", "training"]),
        st.dictionaries(st.text(min_size=1, max_size=20), st.integers(min_value=1, max_value=100)),
        min_size=1, max_size=3
    ))
    @settings(max_examples=20, deadline=None)
    def test_configuration_update_from_dict_preservation(self, config: Config, updates: Dict[str, Any]):
        """
        Additional property: Configuration updates should preserve original and create new instance.
        
        This ensures that update_from_dict creates a new configuration without
        modifying the original.
        """
        original_dict = asdict(config)
        updated_config = config.update_from_dict(updates)
        
        # Original should be unchanged
        assert asdict(config) == original_dict
        
        # Updated config should be different object
        assert updated_config is not config
        
        # Updated config should have the updates applied (where valid)
        for section, section_updates in updates.items():
            if hasattr(updated_config, section):
                section_obj = getattr(updated_config, section)
                for key, value in section_updates.items():
                    if hasattr(section_obj, key):
                        # Value should be updated if the field exists
                        assert getattr(section_obj, key) == value


# Additional edge case tests for comprehensive coverage

class TestConfigurationEdgeCases:
    """Test edge cases and boundary conditions for configuration management."""
    
    def test_invalid_stage_name_rejection(self):
        """Test that invalid stage names are properly rejected."""
        config = Config()
        
        with pytest.raises(ValueError, match="Invalid stage"):
            config.get_stage_config("invalid_stage")
    
    @given(st.text().filter(lambda x: x not in ["yaml", "json"]))
    def test_unsupported_serialization_format_rejection(self, invalid_format: str):
        """Test that unsupported serialization formats are rejected."""
        config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                config.save_config(temp_path, format=invalid_format)
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_nonexistent_file_loading_error(self):
        """Test that loading from nonexistent files raises appropriate errors."""
        with pytest.raises(FileNotFoundError):
            Config.load_config("/nonexistent/path/config.yaml")
    
    @given(valid_config())
    def test_cross_section_validation_effective_batch_size(self, config: Config):
        """Test cross-section validation for effective batch sizes."""
        # Force large effective batch size
        config.training.sft.batch_size = 64
        config.training.sft.gradient_accumulation_steps = 4  # 64 * 4 = 256 > 128
        
        errors = config.get_validation_errors()
        assert any("effective batch size" in error and "should not exceed 128" in error for error in errors)
    
    def test_phi3_model_max_length_validation(self):
        """Test that Phi-3 models have appropriate max_length validation."""
        config = Config()
        config.model.name = "microsoft/Phi-3-mini-4k-instruct"
        config.model.max_length = 8192  # Exceeds typical Phi-3 limit
        
        errors = config.get_validation_errors()
        assert any("Phi-3 models typically support max_length up to 4096" in error for error in errors)
    
    def test_checkpoint_save_steps_validation(self):
        """Test that checkpoint save_steps is validated against training steps."""
        config = Config()
        config.checkpointing.save_steps = 10000  # Larger than any max_steps
        config.training.sft.max_steps = 100
        config.training.reward.max_steps = 50
        config.training.ppo.max_steps = 25
        
        errors = config.get_validation_errors()
        assert any("save_steps should be less than the minimum max_steps" in error for error in errors)