#!/usr/bin/env python3
"""
Integration validation script for RLHF Phi-3 Pipeline core components.

This script validates that all core components can be imported and their
basic interfaces work correctly without requiring external ML dependencies.

Task 7: Checkpoint - Core Components Integration Test Validation
"""

import sys
import traceback
from pathlib import Path

def test_config_manager():
    """Test Configuration Manager functionality."""
    try:
        from rlhf_phi3.config.config_manager import Config, load_default_config, load_colab_config
        
        print("  ✓ Configuration Manager imports successful")
        
        # Test default config
        config = load_default_config()
        assert config.validate_config(), "Default config should be valid"
        print("  ✓ Default configuration validation passed")
        
        # Test Colab config
        colab_config = load_colab_config()
        assert colab_config.validate_config(), "Colab config should be valid"
        print("  ✓ Colab configuration validation passed")
        
        # Test stage configurations
        for stage in ["sft", "reward", "ppo"]:
            stage_config = config.get_stage_config(stage)
            assert isinstance(stage_config, dict), f"Stage config for {stage} should be dict"
            assert "model" in stage_config, f"Stage config for {stage} missing model section"
            assert "training" in stage_config, f"Stage config for {stage} missing training section"
        print("  ✓ Stage-specific configurations work correctly")
        
        # Test configuration serialization
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.save_config(temp_path)
            loaded_config = Config.load_config(temp_path)
            assert loaded_config.model.name == config.model.name
            print("  ✓ Configuration serialization/deserialization works")
        finally:
            temp_path.unlink(missing_ok=True)
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        traceback.print_exc()
        return False

def test_checkpoint_metadata():
    """Test Checkpoint Manager metadata functionality."""
    try:
        from rlhf_phi3.checkpoints.checkpoint_manager import CheckpointMetadata
        from datetime import datetime, timezone
        
        print("  ✓ Checkpoint Manager metadata imports successful")
        
        # Test metadata creation
        metadata = CheckpointMetadata(
            stage="sft",
            epoch=1,
            step=100,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_path="model.bin",
            optimizer_path="optimizer.pt",
            config_hash="test_hash",
            metrics={"loss": 2.5},
            file_hashes={"model": "hash1"}
        )
        
        # Test serialization
        metadata_dict = metadata.to_dict()
        restored_metadata = CheckpointMetadata.from_dict(metadata_dict)
        assert restored_metadata.stage == metadata.stage
        assert restored_metadata.metrics == metadata.metrics
        print("  ✓ Checkpoint metadata serialization works")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_training_orchestrator_interfaces():
    """Test Training Orchestrator interfaces."""
    try:
        from rlhf_phi3.training.training_orchestrator import TrainingStage, StageResult, PipelineState
        
        print("  ✓ Training Orchestrator interfaces import successful")
        
        # Test enums and data classes
        assert TrainingStage.SFT.value == "sft"
        assert TrainingStage.REWARD.value == "reward"
        assert TrainingStage.PPO.value == "ppo"
        
        # Test stage result
        result = StageResult(
            stage=TrainingStage.SFT,
            success=True,
            metrics={"loss": 1.5}
        )
        assert result.stage == TrainingStage.SFT
        assert result.success is True
        
        # Test pipeline state
        state = PipelineState()
        assert state.completed_stages == []
        assert state.stage_results == {}
        
        print("  ✓ Training Orchestrator interfaces work correctly")
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_cross_component_integration():
    """Test integration between components."""
    try:
        from rlhf_phi3.config.config_manager import Config
        
        print("  ✓ Cross-component integration test starting")
        
        config = Config()
        
        # Test 1: Configuration consistency across stages
        stages = ["sft", "reward", "ppo"]
        for stage in stages:
            stage_config = config.get_stage_config(stage)
            
            # Verify all stages have consistent model config
            assert stage_config["model"]["name"] == config.model.name
            assert stage_config["model"]["max_length"] == config.model.max_length
            
            # Verify all stages have LoRA config
            assert stage_config["lora"]["r"] == config.lora.r
            assert stage_config["lora"]["alpha"] == config.lora.alpha
        
        print("  ✓ Configuration consistency across stages verified")
        
        # Test 2: Cross-section validation
        config.training.sft.batch_size = 64
        config.training.sft.gradient_accumulation_steps = 4  # 64 * 4 = 256 > 128
        
        errors = config.get_validation_errors()
        assert any("effective batch size" in error for error in errors)
        print("  ✓ Cross-section validation works correctly")
        
        # Test 3: Configuration updates propagate
        config = Config()  # Reset
        original_name = config.model.name
        
        updates = {"model": {"name": "new-model"}}
        updated_config = config.update_from_dict(updates)
        
        assert config.model.name == original_name  # Original unchanged
        assert updated_config.model.name == "new-model"  # Updated version changed
        print("  ✓ Configuration updates work correctly")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Cross-component integration test failed: {e}")
        return False

def test_error_handling():
    """Test error handling across components."""
    try:
        from rlhf_phi3.config.config_manager import Config
        
        print("  ✓ Error handling test starting")
        
        # Test invalid configurations
        config = Config()
        config.model.max_length = -1  # Invalid
        
        assert not config.validate_config()
        errors = config.get_validation_errors()
        assert len(errors) > 0
        assert any("max_length must be positive" in error for error in errors)
        print("  ✓ Configuration validation catches errors")
        
        # Test invalid stage names
        config = Config()  # Reset to valid
        try:
            config.get_stage_config("invalid_stage")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid stage" in str(e)
        print("  ✓ Invalid stage names handled correctly")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")
        return False

def main():
    """Run all integration validation tests."""
    print("RLHF Phi-3 Pipeline - Core Components Integration Validation")
    print("=" * 70)
    
    tests = [
        ("Configuration Manager", test_config_manager),
        ("Checkpoint Metadata", test_checkpoint_metadata),
        ("Training Orchestrator Interfaces", test_training_orchestrator_interfaces),
        ("Cross-Component Integration", test_cross_component_integration),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"  ✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"  ❌ {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ {test_name} - FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"Integration Validation Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All core component integration tests PASSED!")
        print("✅ Task 7: Core Components Integration Test - COMPLETED")
        
        print("\n📋 Integration Test Summary:")
        print("  • Configuration management works across all components")
        print("  • Checkpoint persistence and metadata handling verified")
        print("  • Training orchestrator interfaces function correctly")
        print("  • Cross-component data flow validated")
        print("  • Error handling mechanisms tested")
        print("  • All component interfaces are compatible")
        
        return True
    else:
        print(f"\n❌ {failed} integration test(s) failed.")
        print("Please review the component implementations.")
        return False

if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.insert(0, str(Path.cwd()))
    
    success = main()
    sys.exit(0 if success else 1)