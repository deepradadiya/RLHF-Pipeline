"""
Integration tests for reproducibility utilities with the configuration system.

Tests how the reproducibility features integrate with the existing Config system
and other pipeline components.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import reproducibility utilities and config system directly for testing
import sys
sys.path.append('.')

# Load the config system
exec(open('rlhf_phi3/config/config_manager.py').read())

# Load reproducibility utilities
exec(open('rlhf_phi3/utils/reproducibility.py').read())


class TestReproducibilityConfigIntegration:
    """Test integration between reproducibility and configuration systems."""
    
    def test_reproducibility_with_config(self):
        """Test using reproducibility manager with pipeline configuration."""
        # Create a configuration
        config = Config()
        config.model.name = "microsoft/Phi-3-mini-4k-instruct"
        config.training.sft.learning_rate = 2e-4
        config.training.sft.batch_size = 4
        
        # Setup reproducibility
        seed = 42
        manager = ReproducibilityManager(seed=seed)
        manager.setup_deterministic_training()
        
        # Create training fingerprint with config
        config_dict = {
            'model_name': config.model.name,
            'sft_learning_rate': config.training.sft.learning_rate,
            'sft_batch_size': config.training.sft.batch_size,
            'lora_r': config.lora.r,
            'lora_alpha': config.lora.alpha
        }
        
        fingerprint = create_training_fingerprint(seed, config_dict)
        
        # Verify fingerprint is generated
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64
        
        # Verify reproducibility
        fingerprint2 = create_training_fingerprint(seed, config_dict)
        assert fingerprint == fingerprint2
    
    def test_config_serialization_with_reproducibility(self):
        """Test saving config with reproducibility information."""
        config = Config()
        seed = 123
        
        # Setup reproducibility
        manager = ReproducibilityManager(seed=seed)
        env_info = manager.log_environment_info()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save config
            config_path = Path(temp_dir) / "config.json"
            config.save_config(config_path)
            
            # Save environment info alongside
            env_path = Path(temp_dir) / "environment.json"
            manager.save_environment_info(env_path)
            
            # Verify both files exist
            assert config_path.exists()
            assert env_path.exists()
            
            # Load and verify config
            loaded_config = Config.load_config(config_path)
            assert loaded_config.model.name == config.model.name
            
            # Load and verify environment info
            with open(env_path, 'r') as f:
                loaded_env = json.load(f)
            assert loaded_env['seed'] == seed
    
    def test_stage_config_with_reproducibility(self):
        """Test stage-specific configuration with reproducibility tracking."""
        config = Config()
        seed = 456
        
        # Get stage-specific config
        sft_config = config.get_stage_config('sft')
        
        # Create reproducibility context for this stage
        stage_fingerprint = create_training_fingerprint(seed, sft_config)
        
        # Verify stage config contains expected keys
        expected_keys = ['model', 'lora', 'training', 'dataset', 'optimization']
        for key in expected_keys:
            assert key in sft_config
        
        # Verify fingerprint is unique for this stage
        assert isinstance(stage_fingerprint, str)
        assert len(stage_fingerprint) == 64
        
        # Different stage should have different fingerprint
        reward_config = config.get_stage_config('reward')
        reward_fingerprint = create_training_fingerprint(seed, reward_config)
        assert stage_fingerprint != reward_fingerprint
    
    def test_reproducibility_summary_with_config(self):
        """Test generating reproducibility summary with configuration."""
        config = Config()
        seed = 789
        
        manager = ReproducibilityManager(seed=seed)
        summary = manager.get_reproducibility_summary()
        
        # Combine with config information
        full_summary = {
            'reproducibility': summary,
            'configuration': {
                'model_name': config.model.name,
                'max_length': config.model.max_length,
                'lora_r': config.lora.r,
                'sft_epochs': config.training.sft.epochs,
                'sft_learning_rate': config.training.sft.learning_rate
            }
        }
        
        # Verify combined summary structure
        assert 'reproducibility' in full_summary
        assert 'configuration' in full_summary
        assert full_summary['reproducibility']['seed'] == seed
        assert full_summary['configuration']['model_name'] == config.model.name
    
    def test_deterministic_training_setup_with_config_validation(self):
        """Test deterministic training setup with configuration validation."""
        config = Config()
        
        # Ensure config is valid
        assert config.validate_config() is True
        
        # Setup deterministic training
        seed = 999
        manager = setup_reproducible_training(seed=seed, enable_deterministic=True)
        
        # Verify deterministic setup
        assert manager.seed == seed
        assert manager.enable_deterministic is True
        
        # Create reproducibility record
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save complete reproducibility record
            record = {
                'seed': seed,
                'config': config.get_stage_config('sft'),
                'environment': manager.log_environment_info(),
                'reproducibility_hash': manager.create_reproducibility_hash()
            }
            
            record_path = Path(temp_dir) / "reproducibility_record.json"
            with open(record_path, 'w') as f:
                json.dump(record, f, indent=2, default=str)
            
            assert record_path.exists()
            
            # Verify record completeness
            with open(record_path, 'r') as f:
                loaded_record = json.load(f)
            
            assert loaded_record['seed'] == seed
            assert 'config' in loaded_record
            assert 'environment' in loaded_record
            assert 'reproducibility_hash' in loaded_record


class TestReproducibilityWorkflow:
    """Test complete reproducibility workflow scenarios."""
    
    def test_training_preparation_workflow(self):
        """Test complete training preparation with reproducibility."""
        # Step 1: Create and validate configuration
        config = Config()
        config.model.max_length = 1024  # Smaller for testing
        config.training.sft.batch_size = 2
        config.training.sft.max_steps = 100
        
        assert config.validate_config() is True
        
        # Step 2: Setup reproducible environment
        seed = 12345
        manager = setup_reproducible_training(seed=seed)
        
        # Step 3: Log complete environment
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "training_environment.json"
            env_info = log_training_environment(env_path, seed=seed)
            
            # Step 4: Create training fingerprint
            sft_config = config.get_stage_config('sft')
            fingerprint = create_training_fingerprint(seed, sft_config)
            
            # Step 5: Save complete training metadata
            training_metadata = {
                'timestamp': env_info['timestamp'],
                'seed': seed,
                'fingerprint': fingerprint,
                'config_hash': manager.create_reproducibility_hash(),
                'stage': 'sft',
                'config': sft_config
            }
            
            metadata_path = Path(temp_dir) / "training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(training_metadata, f, indent=2, default=str)
            
            # Verify all files created
            assert env_path.exists()
            assert metadata_path.exists()
            
            # Verify metadata completeness
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            required_keys = ['timestamp', 'seed', 'fingerprint', 'config_hash', 'stage', 'config']
            for key in required_keys:
                assert key in metadata
    
    def test_reproducibility_validation_workflow(self):
        """Test reproducibility validation across different runs."""
        config = Config()
        seed = 54321
        
        # First run
        manager1 = ReproducibilityManager(seed=seed)
        hash1 = manager1.create_reproducibility_hash()
        
        # Second run with same seed
        manager2 = ReproducibilityManager(seed=seed)
        hash2 = manager2.create_reproducibility_hash()
        
        # Hashes should match
        assert hash1 == hash2
        assert manager2.validate_reproducibility(hash1) is True
        
        # Third run with different seed
        manager3 = ReproducibilityManager(seed=seed + 1)
        hash3 = manager3.create_reproducibility_hash()
        
        # Hash should be different
        assert hash3 != hash1
        assert manager3.validate_reproducibility(hash1) is False
    
    def test_multi_stage_reproducibility(self):
        """Test reproducibility across multiple training stages."""
        config = Config()
        seed = 67890
        
        stages = ['sft', 'reward', 'ppo']
        fingerprints = {}
        
        for stage in stages:
            stage_config = config.get_stage_config(stage)
            fingerprint = create_training_fingerprint(seed, stage_config)
            fingerprints[stage] = fingerprint
        
        # All fingerprints should be different (different configs)
        unique_fingerprints = set(fingerprints.values())
        assert len(unique_fingerprints) == len(stages)
        
        # But reproducible within same stage
        for stage in stages:
            stage_config = config.get_stage_config(stage)
            fingerprint2 = create_training_fingerprint(seed, stage_config)
            assert fingerprints[stage] == fingerprint2


if __name__ == "__main__":
    # Run basic integration tests
    test_integration = TestReproducibilityConfigIntegration()
    test_workflow = TestReproducibilityWorkflow()
    
    print("Running reproducibility integration tests...")
    
    # Test config integration
    test_integration.test_reproducibility_with_config()
    print("✓ Config integration test passed")
    
    test_integration.test_config_serialization_with_reproducibility()
    print("✓ Config serialization test passed")
    
    test_integration.test_stage_config_with_reproducibility()
    print("✓ Stage config test passed")
    
    test_integration.test_reproducibility_summary_with_config()
    print("✓ Summary integration test passed")
    
    test_integration.test_deterministic_training_setup_with_config_validation()
    print("✓ Deterministic setup test passed")
    
    # Test workflows
    test_workflow.test_training_preparation_workflow()
    print("✓ Training preparation workflow test passed")
    
    test_workflow.test_reproducibility_validation_workflow()
    print("✓ Reproducibility validation workflow test passed")
    
    test_workflow.test_multi_stage_reproducibility()
    print("✓ Multi-stage reproducibility test passed")
    
    print("\nAll integration tests passed! ✅")