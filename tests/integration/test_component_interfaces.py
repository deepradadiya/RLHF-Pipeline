"""
Integration tests for component interfaces and interactions.

This module tests the interfaces between components without requiring
external dependencies like PyTorch, transformers, etc. It focuses on
verifying that components can communicate correctly and handle
cross-component data flow.

Task 7: Checkpoint - Core Components Integration Test (Interface Testing)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

# Test the configuration manager independently
def test_config_manager_interface():
    """Test Configuration Manager interface and validation."""
    try:
        from rlhf_phi3.config.config_manager import Config, ModelConfig, LoRAConfig
        
        # Test 1: Basic configuration creation
        config = Config()
        assert hasattr(config, 'model')
        assert hasattr(config, 'lora')
        assert hasattr(config, 'training')
        assert hasattr(config, 'paths')
        
        # Test 2: Configuration validation
        validation_errors = config.get_validation_errors()
        assert isinstance(validation_errors, list)
        
        # Test 3: Stage-specific configuration
        sft_config = config.get_stage_config("sft")
        assert isinstance(sft_config, dict)
        assert "model" in sft_config
        assert "training" in sft_config
        
        reward_config = config.get_stage_config("reward")
        assert isinstance(reward_config, dict)
        assert "model" in reward_config
        assert "training" in reward_config
        
        ppo_config = config.get_stage_config("ppo")
        assert isinstance(ppo_config, dict)
        assert "model" in ppo_config
        assert "training" in ppo_config
        
        # Test 4: Configuration serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.save_config(temp_path, format="json")
            assert temp_path.exists()
            
            # Verify JSON is valid
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            assert "model" in loaded_data
            assert "training" in loaded_data
            
            # Test loading
            loaded_config = Config.load_config(temp_path)
            assert loaded_config.model.name == config.model.name
            
        finally:
            temp_path.unlink(missing_ok=True)
        
        print("✓ Configuration Manager interface tests passed")
        return True
        
    except ImportError as e:
        print(f"✗ Configuration Manager import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Configuration Manager interface test failed: {e}")
        return False


def test_component_interface_compatibility():
    """Test that component interfaces are compatible with each other."""
    try:
        # Test configuration compatibility across components
        from rlhf_phi3.config.config_manager import Config
        
        config = Config()
        
        # Test 1: All stage configurations have required fields
        stages = ["sft", "reward", "ppo"]
        required_sections = ["model", "lora", "training", "optimization", "paths"]
        
        for stage in stages:
            stage_config = config.get_stage_config(stage)
            for section in required_sections:
                assert section in stage_config, f"Missing {section} in {stage} config"
        
        # Test 2: Configuration validation is comprehensive
        # Create invalid configuration
        config.model.max_length = -1  # Invalid value
        errors = config.get_validation_errors()
        assert len(errors) > 0
        assert any("max_length must be positive" in error for error in errors)
        
        # Test 3: Cross-section validation works
        config = Config()  # Reset to valid config
        config.training.sft.batch_size = 100  # Very large batch size
        config.training.sft.gradient_accumulation_steps = 10  # 100 * 10 = 1000 > 128
        
        errors = config.get_validation_errors()
        # Should have cross-section validation error
        assert any("effective batch size" in error for error in errors)
        
        # Test 4: Configuration copying preserves structure
        config = Config()  # Reset to valid config
        copied_config = config.copy()
        
        assert copied_config is not config  # Different objects
        assert copied_config.model.name == config.model.name  # Same values
        
        # Modify copy shouldn't affect original
        copied_config.model.name = "different-model"
        assert config.model.name != copied_config.model.name
        
        print("✓ Component interface compatibility tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Component interface compatibility test failed: {e}")
        return False


def test_checkpoint_metadata_interface():
    """Test checkpoint metadata interface without external dependencies."""
    try:
        from rlhf_phi3.checkpoints.checkpoint_manager import CheckpointMetadata
        from datetime import datetime, timezone
        
        # Test 1: Metadata creation and validation
        metadata = CheckpointMetadata(
            stage="sft",
            epoch=1,
            step=100,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_path="model/pytorch_model.bin",
            optimizer_path="optimizer.pt",
            config_hash="test_hash_123",
            metrics={"loss": 2.5, "accuracy": 0.75},
            file_hashes={"model": "hash1", "optimizer": "hash2"}
        )
        
        # Test 2: Serialization
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["stage"] == "sft"
        assert metadata_dict["epoch"] == 1
        assert metadata_dict["step"] == 100
        
        # Test 3: Deserialization
        restored_metadata = CheckpointMetadata.from_dict(metadata_dict)
        assert restored_metadata.stage == metadata.stage
        assert restored_metadata.epoch == metadata.epoch
        assert restored_metadata.step == metadata.step
        assert restored_metadata.metrics == metadata.metrics
        
        print("✓ Checkpoint metadata interface tests passed")
        return True
        
    except ImportError as e:
        print(f"✗ Checkpoint metadata import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Checkpoint metadata interface test failed: {e}")
        return False


def test_training_orchestrator_interface():
    """Test training orchestrator interface and state management."""
    try:
        from rlhf_phi3.training.training_orchestrator import TrainingStage, StageResult, PipelineState
        from datetime import datetime, timezone
        
        # Test 1: Training stage enumeration
        assert TrainingStage.SFT.value == "sft"
        assert TrainingStage.REWARD.value == "reward"
        assert TrainingStage.PPO.value == "ppo"
        
        # Test 2: Stage result creation
        stage_result = StageResult(
            stage=TrainingStage.SFT,
            success=True,
            checkpoint_path="/path/to/checkpoint",
            metrics={"loss": 1.8, "accuracy": 0.82},
            duration_seconds=300.5,
            memory_peak_gb=4.2
        )
        
        assert stage_result.stage == TrainingStage.SFT
        assert stage_result.success is True
        assert stage_result.metrics["loss"] == 1.8
        
        # Test 3: Pipeline state management
        pipeline_state = PipelineState()
        assert pipeline_state.completed_stages == []
        assert pipeline_state.stage_results == {}
        assert pipeline_state.failure_count == 0
        
        # Add completed stage
        pipeline_state.completed_stages.append(TrainingStage.SFT)
        pipeline_state.stage_results[TrainingStage.SFT] = stage_result
        
        assert len(pipeline_state.completed_stages) == 1
        assert TrainingStage.SFT in pipeline_state.stage_results
        
        print("✓ Training orchestrator interface tests passed")
        return True
        
    except ImportError as e:
        print(f"✗ Training orchestrator import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Training orchestrator interface test failed: {e}")
        return False


def test_integration_data_flow():
    """Test data flow between component interfaces."""
    try:
        from rlhf_phi3.config.config_manager import Config
        
        # Test 1: Configuration flows correctly between stages
        config = Config()
        
        # Simulate training pipeline data flow
        sft_config = config.get_stage_config("sft")
        reward_config = config.get_stage_config("reward")
        ppo_config = config.get_stage_config("ppo")
        
        # Verify each stage has what it needs
        assert sft_config["training"]["epochs"] == config.training.sft.epochs
        assert reward_config["training"]["epochs"] == config.training.reward.epochs
        assert ppo_config["training"]["ppo_epochs"] == config.training.ppo.ppo_epochs
        
        # Test 2: Dataset configuration flows correctly
        assert sft_config["dataset"]["name"] == config.datasets.sft.name
        assert reward_config["dataset"]["name"] == config.datasets.preference.name
        
        # PPO should have both datasets
        assert "datasets" in ppo_config
        assert "sft" in ppo_config["datasets"]
        assert "preference" in ppo_config["datasets"]
        
        # Test 3: Model configuration is consistent across stages
        for stage_config in [sft_config, reward_config, ppo_config]:
            assert stage_config["model"]["name"] == config.model.name
            assert stage_config["model"]["max_length"] == config.model.max_length
            assert stage_config["lora"]["r"] == config.lora.r
        
        # Test 4: Checkpoint configuration is consistent
        for stage_config in [sft_config, reward_config, ppo_config]:
            assert stage_config["checkpointing"]["save_steps"] == config.checkpointing.save_steps
            assert stage_config["checkpointing"]["save_total_limit"] == config.checkpointing.save_total_limit
        
        print("✓ Integration data flow tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration data flow test failed: {e}")
        return False


def test_error_handling_interfaces():
    """Test error handling interfaces across components."""
    try:
        from rlhf_phi3.config.config_manager import Config
        
        # Test 1: Configuration validation catches errors
        config = Config()
        
        # Create various invalid configurations
        test_cases = [
            ("model.max_length", -1, "max_length must be positive"),
            ("lora.r", 0, "rank (r) must be positive"),
            ("training.sft.learning_rate", 1e-7, "Learning rate must be between"),
            ("training.sft.batch_size", 0, "Batch size must be positive"),
        ]
        
        for field_path, invalid_value, expected_error_fragment in test_cases:
            # Reset to valid config
            config = Config()
            
            # Set invalid value
            obj = config
            parts = field_path.split('.')
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], invalid_value)
            
            # Check validation catches the error
            errors = config.get_validation_errors()
            assert len(errors) > 0, f"No errors found for invalid {field_path}={invalid_value}"
            assert any(expected_error_fragment in error for error in errors), \
                f"Expected error fragment '{expected_error_fragment}' not found in errors: {errors}"
        
        # Test 2: Invalid stage names are handled
        config = Config()
        try:
            config.get_stage_config("invalid_stage")
            assert False, "Should have raised ValueError for invalid stage"
        except ValueError as e:
            assert "Invalid stage" in str(e)
        
        print("✓ Error handling interface tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling interface test failed: {e}")
        return False


def run_all_interface_tests():
    """Run all interface tests and report results."""
    print("Running Core Components Integration Interface Tests...")
    print("=" * 60)
    
    tests = [
        ("Configuration Manager Interface", test_config_manager_interface),
        ("Component Interface Compatibility", test_component_interface_compatibility),
        ("Checkpoint Metadata Interface", test_checkpoint_metadata_interface),
        ("Training Orchestrator Interface", test_training_orchestrator_interface),
        ("Integration Data Flow", test_integration_data_flow),
        ("Error Handling Interfaces", test_error_handling_interfaces),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All core component integration interface tests passed!")
        return True
    else:
        print(f"❌ {failed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_interface_tests()
    exit(0 if success else 1)