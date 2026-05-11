"""
Unit tests for reproducibility utilities.

Tests the ReproducibilityManager and related functions for seed management,
environment logging, and deterministic training setup.

Requirements tested:
- 15.1: Configuration snapshots with checkpoints
- 15.2: Fixed random seeds for deterministic training
- 15.3: Environment and library version logging
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

from rlhf_phi3.utils.reproducibility import (
    ReproducibilityManager,
    setup_reproducible_training,
    log_training_environment,
    create_training_fingerprint,
    ensure_deterministic_environment
)

from rlhf_phi3.utils.training_provenance import (
    TrainingProvenanceManager,
    TrainingStageProvenance,
    TrainingProvenance,
    create_provenance_manager
)

from rlhf_phi3.config.config_manager import Config


class TestReproducibilityManager:
    """Test cases for ReproducibilityManager class."""
    
    def test_initialization_with_seed(self):
        """Test manager initialization with explicit seed."""
        seed = 42
        manager = ReproducibilityManager(seed=seed)
        
        assert manager.seed == seed
        assert manager.enable_deterministic is True
        assert manager.environment_info is None
    
    def test_initialization_without_seed(self):
        """Test manager initialization without explicit seed."""
        manager = ReproducibilityManager()
        
        assert isinstance(manager.seed, int)
        assert manager.seed > 0
        assert manager.enable_deterministic is True
    
    def test_initialization_deterministic_disabled(self):
        """Test manager initialization with deterministic mode disabled."""
        manager = ReproducibilityManager(seed=42, enable_deterministic=False)
        
        assert manager.seed == 42
        assert manager.enable_deterministic is False
    
    @patch('rlhf_phi3.utils.reproducibility.random')
    @patch('rlhf_phi3.utils.reproducibility.np')
    def test_set_random_seeds_basic(self, mock_np, mock_random):
        """Test basic random seed setting without ML libraries."""
        manager = ReproducibilityManager(seed=42)
        manager.set_random_seeds()
        
        mock_random.seed.assert_called_once_with(42)
        mock_np.random.seed.assert_called_once_with(42)
        assert os.environ.get('PYTHONHASHSEED') == '42'
    
    @patch('rlhf_phi3.utils.reproducibility.TORCH_AVAILABLE', True)
    @patch('rlhf_phi3.utils.reproducibility.torch')
    def test_set_random_seeds_with_torch(self, mock_torch):
        """Test random seed setting with PyTorch available."""
        mock_torch.use_deterministic_algorithms = MagicMock()
        mock_torch.backends.cudnn = MagicMock()
        
        manager = ReproducibilityManager(seed=42)
        manager.set_random_seeds()
        
        mock_torch.manual_seed.assert_called_once_with(42)
        mock_torch.cuda.manual_seed.assert_called_once_with(42)
        mock_torch.cuda.manual_seed_all.assert_called_once_with(42)
        assert mock_torch.backends.cudnn.deterministic is True
        assert mock_torch.backends.cudnn.benchmark is False
    
    @patch('rlhf_phi3.utils.reproducibility.TRANSFORMERS_AVAILABLE', True)
    @patch('rlhf_phi3.utils.reproducibility.transformers')
    def test_set_random_seeds_with_transformers(self, mock_transformers):
        """Test random seed setting with transformers available."""
        manager = ReproducibilityManager(seed=42)
        manager.set_random_seeds()
        
        mock_transformers.set_seed.assert_called_once_with(42)
    
    def test_set_random_seeds_with_custom_seed(self):
        """Test setting random seeds with a different seed."""
        manager = ReproducibilityManager(seed=42)
        manager.set_random_seeds(seed=123)
        
        assert manager.seed == 123
        assert os.environ.get('PYTHONHASHSEED') == '123'
    
    @patch('rlhf_phi3.utils.reproducibility.TORCH_AVAILABLE', True)
    @patch('rlhf_phi3.utils.reproducibility.torch')
    def test_setup_deterministic_training(self, mock_torch):
        """Test deterministic training setup."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.set_device = MagicMock()
        mock_torch.set_num_threads = MagicMock()
        
        manager = ReproducibilityManager(seed=42)
        manager.setup_deterministic_training()
        
        # Check environment variables
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        assert os.environ.get('CUDA_LAUNCH_BLOCKING') == '1'
        
        # Check PyTorch settings
        mock_torch.set_num_threads.assert_called_once_with(1)
        mock_torch.cuda.set_device.assert_called_once_with(0)
    
    def test_log_environment_info_structure(self):
        """Test that environment info has the expected structure."""
        manager = ReproducibilityManager(seed=42)
        env_info = manager.log_environment_info()
        
        # Check top-level keys
        expected_keys = [
            'timestamp', 'seed', 'deterministic_mode', 'system', 'python',
            'libraries', 'cuda', 'environment_variables', 'git', 'hardware'
        ]
        for key in expected_keys:
            assert key in env_info
        
        # Check nested structures
        assert isinstance(env_info['system'], dict)
        assert isinstance(env_info['python'], dict)
        assert isinstance(env_info['libraries'], dict)
        assert isinstance(env_info['cuda'], dict)
        assert isinstance(env_info['environment_variables'], dict)
        assert isinstance(env_info['git'], dict)
        assert isinstance(env_info['hardware'], dict)
    
    def test_log_environment_info_seed_consistency(self):
        """Test that logged environment info contains correct seed."""
        seed = 12345
        manager = ReproducibilityManager(seed=seed)
        env_info = manager.log_environment_info()
        
        assert env_info['seed'] == seed
        assert env_info['deterministic_mode'] is True
    
    def test_save_environment_info_json(self):
        """Test saving environment info to JSON file."""
        manager = ReproducibilityManager(seed=42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "env_info.json"
            manager.save_environment_info(output_path, format='json')
            
            assert output_path.exists()
            
            # Verify JSON content
            with open(output_path, 'r') as f:
                loaded_info = json.load(f)
            
            assert loaded_info['seed'] == 42
            assert 'timestamp' in loaded_info
    
    def test_save_environment_info_creates_directory(self):
        """Test that save_environment_info creates parent directories."""
        manager = ReproducibilityManager(seed=42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "env_info.json"
            manager.save_environment_info(output_path)
            
            assert output_path.exists()
            assert output_path.parent.exists()
    
    def test_save_environment_info_invalid_format(self):
        """Test error handling for invalid format."""
        manager = ReproducibilityManager(seed=42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "env_info.txt"
            
            with pytest.raises(ValueError, match="Unsupported format"):
                manager.save_environment_info(output_path, format='txt')
    
    def test_create_reproducibility_hash_consistency(self):
        """Test that reproducibility hash is consistent for same environment."""
        manager1 = ReproducibilityManager(seed=42)
        manager2 = ReproducibilityManager(seed=42)
        
        hash1 = manager1.create_reproducibility_hash()
        hash2 = manager2.create_reproducibility_hash()
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex string length
    
    def test_create_reproducibility_hash_different_seeds(self):
        """Test that different seeds produce different hashes."""
        manager1 = ReproducibilityManager(seed=42)
        manager2 = ReproducibilityManager(seed=123)
        
        hash1 = manager1.create_reproducibility_hash()
        hash2 = manager2.create_reproducibility_hash()
        
        assert hash1 != hash2
    
    def test_validate_reproducibility_matching(self):
        """Test reproducibility validation with matching hash."""
        manager = ReproducibilityManager(seed=42)
        reference_hash = manager.create_reproducibility_hash()
        
        assert manager.validate_reproducibility(reference_hash) is True
    
    def test_validate_reproducibility_non_matching(self):
        """Test reproducibility validation with non-matching hash."""
        manager = ReproducibilityManager(seed=42)
        fake_hash = "a" * 64  # Fake SHA-256 hash
        
        assert manager.validate_reproducibility(fake_hash) is False
    
    def test_get_reproducibility_summary_structure(self):
        """Test reproducibility summary structure."""
        manager = ReproducibilityManager(seed=42)
        summary = manager.get_reproducibility_summary()
        
        expected_keys = [
            'seed', 'deterministic_mode', 'reproducibility_hash',
            'key_versions', 'timestamp'
        ]
        for key in expected_keys:
            assert key in summary
        
        assert summary['seed'] == 42
        assert summary['deterministic_mode'] is True
        assert isinstance(summary['key_versions'], dict)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @patch('rlhf_phi3.utils.reproducibility.ReproducibilityManager')
    def test_setup_reproducible_training(self, mock_manager_class):
        """Test setup_reproducible_training convenience function."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        result = setup_reproducible_training(seed=42, enable_deterministic=True)
        
        mock_manager_class.assert_called_once_with(seed=42, enable_deterministic=True)
        mock_manager.setup_deterministic_training.assert_called_once()
        assert result == mock_manager
    
    @patch('rlhf_phi3.utils.reproducibility.ReproducibilityManager')
    def test_log_training_environment(self, mock_manager_class):
        """Test log_training_environment convenience function."""
        mock_manager = MagicMock()
        mock_env_info = {'seed': 42, 'timestamp': '2024-01-01'}
        mock_manager.log_environment_info.return_value = mock_env_info
        mock_manager_class.return_value = mock_manager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "env.json"
            result = log_training_environment(output_path, seed=42)
            
            mock_manager_class.assert_called_once_with(seed=42)
            mock_manager.log_environment_info.assert_called_once()
            mock_manager.save_environment_info.assert_called_once_with(output_path)
            assert result == mock_env_info
    
    @patch('rlhf_phi3.utils.reproducibility.ReproducibilityManager')
    def test_create_training_fingerprint(self, mock_manager_class):
        """Test create_training_fingerprint convenience function."""
        mock_manager = MagicMock()
        mock_env_info = {
            'python': {'version': '3.8.0'},
            'libraries': {'torch': '2.0.0', 'transformers': '4.36.0'},
            'cuda': {'version': '11.8'}
        }
        mock_manager.log_environment_info.return_value = mock_env_info
        mock_manager_class.return_value = mock_manager
        
        config_dict = {'learning_rate': 0.001, 'batch_size': 4}
        result = create_training_fingerprint(seed=42, config_dict=config_dict)
        
        mock_manager_class.assert_called_once_with(seed=42)
        mock_manager.log_environment_info.assert_called_once()
        
        # Result should be a SHA-256 hash
        assert isinstance(result, str)
        assert len(result) == 64
    
    @patch('rlhf_phi3.utils.reproducibility.ReproducibilityManager')
    def test_ensure_deterministic_environment(self, mock_manager_class):
        """Test ensure_deterministic_environment convenience function."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        ensure_deterministic_environment()
        
        mock_manager_class.assert_called_once()
        mock_manager.setup_deterministic_training.assert_called_once()


class TestEnvironmentInfoMethods:
    """Test cases for environment information collection methods."""
    
    def test_get_system_info(self):
        """Test system information collection."""
        manager = ReproducibilityManager(seed=42)
        system_info = manager._get_system_info()
        
        expected_keys = [
            'platform', 'system', 'release', 'version', 'machine',
            'processor', 'architecture', 'hostname'
        ]
        for key in expected_keys:
            assert key in system_info
            assert system_info[key] is not None
    
    def test_get_python_info(self):
        """Test Python information collection."""
        manager = ReproducibilityManager(seed=42)
        python_info = manager._get_python_info()
        
        expected_keys = [
            'version', 'version_info', 'executable', 'path', 'prefix', 'base_prefix'
        ]
        for key in expected_keys:
            assert key in python_info
        
        # Check version_info structure
        version_info = python_info['version_info']
        assert 'major' in version_info
        assert 'minor' in version_info
        assert isinstance(version_info['major'], int)
        assert isinstance(version_info['minor'], int)
    
    def test_get_library_versions(self):
        """Test library version collection."""
        manager = ReproducibilityManager(seed=42)
        library_versions = manager._get_library_versions()
        
        # Should contain key libraries
        expected_libraries = ['torch', 'transformers', 'numpy', 'pandas']
        for lib in expected_libraries:
            assert lib in library_versions
            # Version should be string (either version number or 'not_installed')
            assert isinstance(library_versions[lib], str)
    
    def test_get_cuda_info_structure(self):
        """Test CUDA information structure."""
        manager = ReproducibilityManager(seed=42)
        cuda_info = manager._get_cuda_info()
        
        expected_keys = ['available', 'version', 'device_count', 'devices']
        for key in expected_keys:
            assert key in cuda_info
        
        assert isinstance(cuda_info['available'], bool)
        assert isinstance(cuda_info['device_count'], int)
        assert isinstance(cuda_info['devices'], list)
    
    def test_get_relevant_env_vars(self):
        """Test relevant environment variables collection."""
        manager = ReproducibilityManager(seed=42)
        env_vars = manager._get_relevant_env_vars()
        
        expected_vars = [
            'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED', 'WANDB_PROJECT'
        ]
        for var in expected_vars:
            assert var in env_vars
            assert isinstance(env_vars[var], str)
    
    def test_get_git_info_structure(self):
        """Test git information structure."""
        manager = ReproducibilityManager(seed=42)
        git_info = manager._get_git_info()
        
        expected_keys = [
            'available', 'commit_hash', 'branch', 'remote_url', 'is_dirty'
        ]
        for key in expected_keys:
            assert key in git_info
        
        assert isinstance(git_info['available'], bool)
    
    def test_get_hardware_info(self):
        """Test hardware information collection."""
        manager = ReproducibilityManager(seed=42)
        hardware_info = manager._get_hardware_info()
        
        assert 'cpu_count' in hardware_info
        assert 'memory' in hardware_info
        assert isinstance(hardware_info['cpu_count'], int)
        assert isinstance(hardware_info['memory'], dict)


class TestIntegrationScenarios:
    """Integration test scenarios for reproducibility features."""
    
    def test_full_reproducibility_workflow(self):
        """Test complete reproducibility workflow."""
        seed = 42
        
        # Setup reproducible training
        manager = setup_reproducible_training(seed=seed)
        
        # Log environment
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "environment.json"
            log_training_environment(env_path, seed=seed)
            
            assert env_path.exists()
            
            # Verify saved content
            with open(env_path, 'r') as f:
                env_data = json.load(f)
            
            assert env_data['seed'] == seed
        
        # Create training fingerprint
        config = {'learning_rate': 0.001, 'batch_size': 4}
        fingerprint = create_training_fingerprint(seed, config)
        
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64
    
    def test_reproducibility_hash_stability(self):
        """Test that reproducibility hashes are stable across multiple calls."""
        manager = ReproducibilityManager(seed=42)
        
        # Generate hash multiple times
        hashes = [manager.create_reproducibility_hash() for _ in range(3)]
        
        # All hashes should be identical
        assert len(set(hashes)) == 1
        assert all(len(h) == 64 for h in hashes)
    
    def test_environment_logging_completeness(self):
        """Test that environment logging captures all required information."""
        manager = ReproducibilityManager(seed=42)
        env_info = manager.log_environment_info()
        
        # Verify completeness
        assert env_info['seed'] == 42
        assert 'timestamp' in env_info
        assert len(env_info['libraries']) > 0
        assert 'python' in env_info
        assert 'system' in env_info
        
        # Verify summary generation
        summary = manager.get_reproducibility_summary()
        assert summary['seed'] == 42
        assert 'key_versions' in summary
        assert 'reproducibility_hash' in summary


class TestTrainingProvenanceIntegration:
    """Test cases for training provenance integration with reproducibility."""
    
    def test_provenance_manager_initialization(self):
        """Test training provenance manager initialization with reproducibility."""
        config = Config()
        seed = 42
        
        provenance_manager = create_provenance_manager(config, seed=seed)
        
        assert provenance_manager.seed == seed
        assert provenance_manager.config == config
        assert provenance_manager.pipeline_id is not None
        assert provenance_manager.provenance.seed == seed
        assert provenance_manager.provenance.environment_info is not None
        assert provenance_manager.provenance.reproducibility_hash is not None
    
    def test_provenance_manager_stage_tracking(self):
        """Test training stage tracking with reproducibility metadata."""
        config = Config()
        provenance_manager = create_provenance_manager(config, seed=123)
        
        # Start a stage
        stage = provenance_manager.start_stage("sft")
        
        assert stage.stage_name == "sft"
        assert stage.start_time is not None
        assert len(provenance_manager.provenance.stages) == 1
        
        # Add metrics
        provenance_manager.update_stage_metrics("sft", 10, {"loss": 1.0, "lr": 2e-4})
        provenance_manager.update_stage_metrics("sft", 20, {"loss": 0.8, "lr": 1.8e-4})
        
        # Check metrics were recorded
        sft_stage = provenance_manager.provenance.get_stage("sft")
        assert len(sft_stage.metrics_history) == 2
        assert sft_stage.total_steps == 20
        
        # Finalize stage
        provenance_manager.finalize_stage("sft", final_loss=0.5, checkpoint_path="checkpoints/sft")
        
        assert sft_stage.end_time is not None
        assert sft_stage.final_loss == 0.5
        assert sft_stage.checkpoint_path == "checkpoints/sft"
    
    def test_provenance_manager_training_fingerprint(self):
        """Test training fingerprint generation with provenance."""
        config = Config()
        seed = 456
        provenance_manager = create_provenance_manager(config, seed=seed)
        
        # Generate fingerprints
        fingerprint1 = provenance_manager.create_training_fingerprint()
        fingerprint2 = provenance_manager.create_training_fingerprint()
        
        # Should be consistent
        assert fingerprint1 == fingerprint2
        assert len(fingerprint1) == 64  # SHA-256 hex string
        
        # Should be different for different seeds
        provenance_manager2 = create_provenance_manager(config, seed=seed + 1)
        fingerprint3 = provenance_manager2.create_training_fingerprint()
        
        assert fingerprint1 != fingerprint3
    
    def test_provenance_manager_save_and_load(self):
        """Test saving and loading training provenance."""
        config = Config()
        provenance_manager = create_provenance_manager(config, seed=789)
        
        # Add some training data
        provenance_manager.start_stage("sft")
        provenance_manager.update_stage_metrics("sft", 10, {"loss": 1.0})
        provenance_manager.finalize_stage("sft", final_loss=0.5)
        
        provenance_manager.add_evaluation_results({"mt_bench_score": 7.5})
        provenance_manager.finalize_training(final_model_path="models/test")
        
        # Save provenance
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            provenance_file = provenance_manager.save_provenance(output_dir)
            
            assert provenance_file.exists()
            
            # Load and verify
            loaded_provenance = TrainingProvenance.load_from_file(provenance_file)
            
            assert loaded_provenance.seed == 789
            assert len(loaded_provenance.stages) == 1
            assert loaded_provenance.stages[0].stage_name == "sft"
            assert loaded_provenance.evaluation_results["mt_bench_score"] == 7.5
            assert loaded_provenance.final_model_path == "models/test"
    
    def test_provenance_summary_generation(self):
        """Test provenance summary generation."""
        config = Config()
        provenance_manager = create_provenance_manager(config, seed=999)
        
        # Add multiple stages
        stages = ["sft", "reward", "ppo"]
        for stage_name in stages:
            provenance_manager.start_stage(stage_name)
            provenance_manager.update_stage_metrics(stage_name, 10, {"loss": 1.0})
            provenance_manager.finalize_stage(stage_name, final_loss=0.5)
        
        provenance_manager.finalize_training()
        
        # Generate summary
        summary = provenance_manager.get_provenance_summary()
        
        assert summary["seed"] == 999
        assert summary["stages_completed"] == 3
        assert set(summary["stage_names"]) == set(stages)
        assert summary["reproducibility_hash"] is not None
        assert summary["training_fingerprint"] is not None
        
        # Check stage summaries
        assert len(summary["stages"]) == 3
        for stage_summary in summary["stages"]:
            assert stage_summary["name"] in stages
            assert stage_summary["completed"] is True
            assert stage_summary["total_steps"] == 10


class TestConfigurationSnapshotIntegration:
    """Test cases for configuration snapshot integration with reproducibility."""
    
    def test_config_checkpoint_snapshot_creation(self):
        """Test configuration checkpoint snapshot creation with reproducibility."""
        config = Config()
        checkpoint_id = "test_checkpoint_001"
        training_metadata = {"stage": "sft", "epoch": 1, "step": 100}
        
        snapshot = config.create_checkpoint_snapshot(checkpoint_id, training_metadata)
        
        assert snapshot["checkpoint_id"] == checkpoint_id
        assert snapshot["training_metadata"] == training_metadata
        assert "timestamp" in snapshot
        assert "config" in snapshot
        assert "config_hash" in snapshot
        assert "reproducibility" in snapshot
        
        # Config hash should be consistent
        snapshot2 = config.create_checkpoint_snapshot(checkpoint_id, training_metadata)
        assert snapshot["config_hash"] == snapshot2["config_hash"]
    
    def test_config_checkpoint_snapshot_save_load(self):
        """Test saving and loading configuration snapshots."""
        config = Config()
        checkpoint_id = "test_checkpoint_002"
        training_metadata = {"stage": "reward", "epoch": 1, "step": 50}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            
            # Save snapshot
            config.save_checkpoint_snapshot(checkpoint_path, checkpoint_id, training_metadata)
            
            # Verify file exists
            snapshot_file = checkpoint_path / "config_snapshot.json"
            assert snapshot_file.exists()
            
            # Load snapshot
            loaded_config, loaded_snapshot = Config.load_checkpoint_snapshot(checkpoint_path)
            
            # Verify loaded data
            assert loaded_config.model.name == config.model.name
            assert loaded_snapshot["checkpoint_id"] == checkpoint_id
            assert loaded_snapshot["training_metadata"] == training_metadata
    
    def test_config_snapshot_integrity_verification(self):
        """Test configuration snapshot integrity verification."""
        config = Config()
        checkpoint_id = "test_checkpoint_003"
        
        snapshot = config.create_checkpoint_snapshot(checkpoint_id)
        
        # Verify hash integrity
        config_hash = snapshot["config_hash"]
        assert len(config_hash) == 64  # SHA-256 hex string
        
        # Modify config and verify hash changes
        config.model.max_length = 4096
        snapshot2 = config.create_checkpoint_snapshot(checkpoint_id)
        
        assert snapshot2["config_hash"] != config_hash


class TestDeterministicTrainingFeatures:
    """Test cases for deterministic training features (Requirement 15.2)."""
    
    def test_deterministic_training_setup_with_fixed_seed(self):
        """Test deterministic training setup with fixed seeds."""
        seed = 12345
        
        # Setup deterministic training
        manager = setup_reproducible_training(seed=seed, enable_deterministic=True)
        
        assert manager.seed == seed
        assert manager.enable_deterministic is True
        assert os.environ.get('PYTHONHASHSEED') == str(seed)
        
        # Environment variables should be set for deterministic training
        expected_vars = ['CUBLAS_WORKSPACE_CONFIG', 'CUDA_LAUNCH_BLOCKING']
        for var in expected_vars:
            assert var in os.environ
    
    def test_deterministic_training_reproducibility_across_runs(self):
        """Test that deterministic training produces reproducible results."""
        seed = 54321
        
        # Create two managers with same seed
        manager1 = setup_reproducible_training(seed=seed)
        manager2 = setup_reproducible_training(seed=seed)
        
        # Both should have identical reproducibility hashes
        hash1 = manager1.create_reproducibility_hash()
        hash2 = manager2.create_reproducibility_hash()
        
        assert hash1 == hash2
        
        # Both should validate each other's hashes
        assert manager1.validate_reproducibility(hash2)
        assert manager2.validate_reproducibility(hash1)
    
    def test_deterministic_training_different_seeds_produce_different_results(self):
        """Test that different seeds produce different reproducibility hashes."""
        seed1 = 11111
        seed2 = 22222
        
        manager1 = setup_reproducible_training(seed=seed1)
        manager2 = setup_reproducible_training(seed=seed2)
        
        hash1 = manager1.create_reproducibility_hash()
        hash2 = manager2.create_reproducibility_hash()
        
        # Different seeds should produce different hashes
        assert hash1 != hash2
        
        # Cross-validation should fail
        assert not manager1.validate_reproducibility(hash2)
        assert not manager2.validate_reproducibility(hash1)
    
    @patch('rlhf_phi3.utils.reproducibility.TORCH_AVAILABLE', True)
    @patch('rlhf_phi3.utils.reproducibility.torch')
    def test_deterministic_training_torch_configuration(self, mock_torch):
        """Test PyTorch-specific deterministic training configuration."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.set_device = MagicMock()
        mock_torch.set_num_threads = MagicMock()
        mock_torch.backends.cudnn = MagicMock()
        
        seed = 98765
        manager = ReproducibilityManager(seed=seed, enable_deterministic=True)
        manager.setup_deterministic_training()
        
        # Verify PyTorch deterministic settings
        mock_torch.manual_seed.assert_called_with(seed)
        mock_torch.cuda.manual_seed.assert_called_with(seed)
        mock_torch.cuda.manual_seed_all.assert_called_with(seed)
        mock_torch.set_num_threads.assert_called_with(1)
        mock_torch.cuda.set_device.assert_called_with(0)
        
        assert mock_torch.backends.cudnn.deterministic is True
        assert mock_torch.backends.cudnn.benchmark is False


class TestEnvironmentLoggingFeatures:
    """Test cases for environment logging features (Requirement 15.3)."""
    
    def test_environment_logging_completeness(self):
        """Test that environment logging captures all required information."""
        seed = 13579
        manager = ReproducibilityManager(seed=seed)
        
        env_info = manager.log_environment_info()
        
        # Check required top-level fields
        required_fields = [
            'timestamp', 'seed', 'deterministic_mode', 'system', 'python',
            'libraries', 'cuda', 'environment_variables', 'git', 'hardware'
        ]
        
        for field in required_fields:
            assert field in env_info, f"Missing required field: {field}"
        
        # Check seed consistency
        assert env_info['seed'] == seed
        
        # Check nested structure completeness
        assert isinstance(env_info['system'], dict)
        assert isinstance(env_info['python'], dict)
        assert isinstance(env_info['libraries'], dict)
        assert isinstance(env_info['cuda'], dict)
        assert isinstance(env_info['environment_variables'], dict)
        assert isinstance(env_info['git'], dict)
        assert isinstance(env_info['hardware'], dict)
    
    def test_environment_logging_library_version_tracking(self):
        """Test library version tracking in environment logging."""
        manager = ReproducibilityManager(seed=24680)
        env_info = manager.log_environment_info()
        
        libraries = env_info['libraries']
        
        # Check that key ML libraries are tracked
        expected_libraries = [
            'torch', 'transformers', 'datasets', 'peft', 'trl',
            'numpy', 'pandas', 'wandb', 'accelerate'
        ]
        
        for lib in expected_libraries:
            assert lib in libraries
            # Version should be string (either version number or status)
            assert isinstance(libraries[lib], str)
    
    def test_environment_logging_system_information(self):
        """Test system information capture in environment logging."""
        manager = ReproducibilityManager(seed=97531)
        env_info = manager.log_environment_info()
        
        system_info = env_info['system']
        
        # Check required system fields
        required_system_fields = [
            'platform', 'system', 'release', 'version', 
            'machine', 'processor', 'architecture', 'hostname'
        ]
        
        for field in required_system_fields:
            assert field in system_info
            assert system_info[field] is not None
    
    def test_environment_logging_python_information(self):
        """Test Python information capture in environment logging."""
        manager = ReproducibilityManager(seed=86420)
        env_info = manager.log_environment_info()
        
        python_info = env_info['python']
        
        # Check required Python fields
        required_python_fields = [
            'version', 'version_info', 'executable', 'path', 'prefix'
        ]
        
        for field in required_python_fields:
            assert field in python_info
        
        # Check version_info structure
        version_info = python_info['version_info']
        assert 'major' in version_info
        assert 'minor' in version_info
        assert isinstance(version_info['major'], int)
        assert isinstance(version_info['minor'], int)
    
    def test_environment_logging_save_to_file(self):
        """Test saving environment information to file."""
        manager = ReproducibilityManager(seed=75319)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "env_info.json"
            
            # Save environment info
            manager.save_environment_info(output_path, format='json')
            
            assert output_path.exists()
            
            # Load and verify
            with open(output_path, 'r') as f:
                loaded_info = json.load(f)
            
            assert loaded_info['seed'] == 75319
            assert 'timestamp' in loaded_info
            assert 'libraries' in loaded_info
    
    def test_environment_logging_yaml_format(self):
        """Test saving environment information in YAML format."""
        manager = ReproducibilityManager(seed=95173)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "env_info.yaml"
            
            try:
                # Try to save in YAML format
                manager.save_environment_info(output_path, format='yaml')
                
                # If YAML is available, file should exist
                assert output_path.exists()
                
            except ImportError:
                # YAML not available, should raise ImportError
                with pytest.raises(ImportError, match="PyYAML is required"):
                    manager.save_environment_info(output_path, format='yaml')


class TestConfigurationSnapshotFeatures:
    """Test cases for configuration snapshot features (Requirement 15.1)."""
    
    def test_configuration_snapshot_with_checkpoints_basic(self):
        """Test basic configuration snapshot functionality with checkpoints."""
        config = Config()
        checkpoint_id = "snapshot_test_001"
        training_metadata = {
            "stage": "sft",
            "epoch": 2,
            "step": 250,
            "loss": 0.85
        }
        
        snapshot = config.create_checkpoint_snapshot(checkpoint_id, training_metadata)
        
        # Verify snapshot structure
        assert snapshot["checkpoint_id"] == checkpoint_id
        assert snapshot["training_metadata"] == training_metadata
        assert "timestamp" in snapshot
        assert "config" in snapshot
        assert "config_hash" in snapshot
        assert "config_version" in snapshot
        assert "snapshot_type" in snapshot
        
        # Verify config hash is valid SHA-256
        config_hash = snapshot["config_hash"]
        assert len(config_hash) == 64
        assert all(c in '0123456789abcdef' for c in config_hash.lower())
    
    def test_configuration_snapshot_with_reproducibility_integration(self):
        """Test configuration snapshot integration with reproducibility manager."""
        config = Config()
        checkpoint_id = "snapshot_test_002"
        
        # Create snapshot (should include reproducibility info)
        snapshot = config.create_checkpoint_snapshot(checkpoint_id)
        
        # Should include reproducibility information
        assert "reproducibility" in snapshot
        
        # If reproducibility manager is available, should have detailed info
        if snapshot["reproducibility"] is not None:
            repro_info = snapshot["reproducibility"]
            assert "seed" in repro_info
            assert "deterministic_mode" in repro_info
            assert "reproducibility_hash" in repro_info
    
    def test_configuration_snapshot_consistency_across_calls(self):
        """Test that configuration snapshots are consistent across multiple calls."""
        config = Config()
        checkpoint_id = "snapshot_test_003"
        training_metadata = {"test": "consistency"}
        
        # Create multiple snapshots
        snapshot1 = config.create_checkpoint_snapshot(checkpoint_id, training_metadata)
        snapshot2 = config.create_checkpoint_snapshot(checkpoint_id, training_metadata)
        
        # Config hashes should be identical
        assert snapshot1["config_hash"] == snapshot2["config_hash"]
        
        # Config content should be identical
        assert snapshot1["config"] == snapshot2["config"]
    
    def test_configuration_snapshot_detects_config_changes(self):
        """Test that configuration snapshots detect configuration changes."""
        config = Config()
        checkpoint_id = "snapshot_test_004"
        
        # Create initial snapshot
        snapshot1 = config.create_checkpoint_snapshot(checkpoint_id)
        hash1 = snapshot1["config_hash"]
        
        # Modify configuration
        config.model.max_length = 4096
        config.training.sft.learning_rate = 1e-4
        
        # Create new snapshot
        snapshot2 = config.create_checkpoint_snapshot(checkpoint_id)
        hash2 = snapshot2["config_hash"]
        
        # Hashes should be different
        assert hash1 != hash2
    
    def test_configuration_snapshot_save_and_load_workflow(self):
        """Test complete save and load workflow for configuration snapshots."""
        config = Config()
        checkpoint_id = "snapshot_test_005"
        training_metadata = {
            "stage": "reward",
            "epoch": 1,
            "step": 100,
            "accuracy": 0.85
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)
            
            # Save snapshot
            config.save_checkpoint_snapshot(checkpoint_path, checkpoint_id, training_metadata)
            
            # Verify snapshot file exists
            snapshot_file = checkpoint_path / "config_snapshot.json"
            assert snapshot_file.exists()
            
            # Load snapshot
            loaded_config, loaded_snapshot = Config.load_checkpoint_snapshot(checkpoint_path)
            
            # Verify loaded configuration matches original
            assert loaded_config.model.name == config.model.name
            assert loaded_config.training.sft.learning_rate == config.training.sft.learning_rate
            
            # Verify loaded snapshot metadata
            assert loaded_snapshot["checkpoint_id"] == checkpoint_id
            assert loaded_snapshot["training_metadata"] == training_metadata


class TestTrainingProvenanceFeatures:
    """Test cases for training provenance features (Requirement 15.4)."""
    
    def test_training_provenance_in_model_metadata_basic(self):
        """Test basic training provenance inclusion in model metadata."""
        config = Config()
        seed = 11223
        provenance_manager = create_provenance_manager(config, seed=seed)
        
        # Simulate training stages
        stages = ["sft", "reward", "ppo"]
        for stage_name in stages:
            provenance_manager.start_stage(stage_name)
            provenance_manager.update_stage_metrics(stage_name, 10, {"loss": 1.0})
            provenance_manager.finalize_stage(stage_name, final_loss=0.5)
        
        # Add evaluation results
        evaluation_results = {"mt_bench_score": 7.8, "helpfulness": 8.2}
        provenance_manager.add_evaluation_results(evaluation_results)
        provenance_manager.finalize_training(final_model_path="models/test")
        
        # Get provenance summary for model metadata
        summary = provenance_manager.get_provenance_summary()
        
        # Verify provenance information is complete
        assert summary["seed"] == seed
        assert summary["stages_completed"] == 3
        assert set(summary["stage_names"]) == set(stages)
        assert summary["has_evaluation_results"] is True
        assert summary["final_model_path"] == "models/test"
        assert summary["reproducibility_hash"] is not None
        assert summary["training_fingerprint"] is not None
    
    def test_training_provenance_stage_details(self):
        """Test detailed training provenance for individual stages."""
        config = Config()
        provenance_manager = create_provenance_manager(config, seed=33445)
        
        # Start and track a detailed stage
        stage = provenance_manager.start_stage("sft")
        
        # Add multiple metrics over time
        metrics_sequence = [
            (10, {"loss": 2.0, "lr": 2e-4, "grad_norm": 1.5}),
            (20, {"loss": 1.8, "lr": 1.9e-4, "grad_norm": 1.3}),
            (30, {"loss": 1.5, "lr": 1.8e-4, "grad_norm": 1.1}),
            (40, {"loss": 1.2, "lr": 1.7e-4, "grad_norm": 0.9}),
            (50, {"loss": 1.0, "lr": 1.6e-4, "grad_norm": 0.8})
        ]
        
        for step, metrics in metrics_sequence:
            provenance_manager.update_stage_metrics("sft", step, metrics)
        
        # Finalize stage
        provenance_manager.finalize_stage("sft", final_loss=0.75, checkpoint_path="checkpoints/sft_final")
        
        # Verify stage provenance
        sft_stage = provenance_manager.provenance.get_stage("sft")
        
        assert sft_stage.stage_name == "sft"
        assert sft_stage.total_steps == 50
        assert sft_stage.final_loss == 0.75
        assert sft_stage.checkpoint_path == "checkpoints/sft_final"
        assert len(sft_stage.metrics_history) == 5
        
        # Verify metrics progression
        for i, (expected_step, expected_metrics) in enumerate(metrics_sequence):
            recorded_metrics = sft_stage.metrics_history[i]
            assert recorded_metrics["step"] == expected_step
            assert recorded_metrics["metrics"] == expected_metrics
    
    def test_training_provenance_environment_integration(self):
        """Test training provenance integration with environment information."""
        config = Config()
        seed = 55667
        provenance_manager = create_provenance_manager(config, seed=seed)
        
        # Verify environment information is captured
        assert provenance_manager.provenance.environment_info is not None
        
        env_info = provenance_manager.provenance.environment_info
        
        # Check that environment info contains expected fields
        assert env_info["seed"] == seed
        assert "timestamp" in env_info
        assert "libraries" in env_info
        assert "system" in env_info
        assert "python" in env_info
        
        # Check reproducibility hash is available
        assert provenance_manager.provenance.reproducibility_hash is not None
        
        # Verify training fingerprint includes environment
        fingerprint = provenance_manager.create_training_fingerprint()
        assert len(fingerprint) == 64  # SHA-256 hex string
    
    def test_training_provenance_serialization_and_persistence(self):
        """Test training provenance serialization and persistence."""
        config = Config()
        provenance_manager = create_provenance_manager(config, seed=77889)
        
        # Create comprehensive training provenance
        provenance_manager.start_stage("sft")
        provenance_manager.update_stage_metrics("sft", 25, {"loss": 1.5})
        provenance_manager.finalize_stage("sft", final_loss=0.8)
        
        provenance_manager.add_evaluation_results({"score": 8.5})
        provenance_manager.finalize_training(final_model_path="models/final")
        
        # Test serialization
        provenance_dict = provenance_manager.provenance.to_dict()
        
        assert isinstance(provenance_dict, dict)
        assert provenance_dict["seed"] == 77889
        assert len(provenance_dict["stages"]) == 1
        assert provenance_dict["evaluation_results"]["score"] == 8.5
        
        # Test deserialization
        loaded_provenance = TrainingProvenance.from_dict(provenance_dict)
        
        assert loaded_provenance.seed == 77889
        assert len(loaded_provenance.stages) == 1
        assert loaded_provenance.stages[0].stage_name == "sft"
        assert loaded_provenance.evaluation_results["score"] == 8.5
        
        # Test file persistence
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            provenance_file = provenance_manager.save_provenance(output_dir)
            
            assert provenance_file.exists()
            
            # Load from file
            file_loaded_provenance = TrainingProvenance.load_from_file(provenance_file)
            
            assert file_loaded_provenance.seed == 77889
            assert len(file_loaded_provenance.stages) == 1


class TestReproducibilityScriptsFeatures:
    """Test cases for reproducibility scripts features (Requirement 15.5)."""
    
    def test_environment_recreation_script_generation(self):
        """Test generation of environment recreation scripts."""
        # This test verifies that the environment recreation script exists and is functional
        script_path = Path(__file__).parent.parent / "scripts" / "recreate_environment.py"
        
        assert script_path.exists(), "Environment recreation script should exist"
        
        # Verify script is executable
        assert script_path.stat().st_mode & 0o111, "Script should be executable"
        
        # Verify script contains expected functionality
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Check for key components
        assert "EnvironmentRecreator" in script_content
        assert "recreate_from_metadata" in script_content
        assert "generate_requirements_file" in script_content
        assert "generate_environment_script" in script_content
    
    def test_training_fingerprint_consistency_for_recreation(self):
        """Test training fingerprint consistency for environment recreation."""
        seed = 99001
        config_dict = {
            "learning_rate": 2e-4,
            "batch_size": 4,
            "max_steps": 1000,
            "model_name": "microsoft/Phi-3-mini-4k-instruct"
        }
        
        # Generate fingerprints multiple times
        fingerprints = []
        for _ in range(5):
            fingerprint = create_training_fingerprint(seed, config_dict)
            fingerprints.append(fingerprint)
        
        # All fingerprints should be identical
        assert len(set(fingerprints)) == 1, "Training fingerprints should be consistent"
        
        # Fingerprint should be valid SHA-256
        fingerprint = fingerprints[0]
        assert len(fingerprint) == 64
        assert all(c in '0123456789abcdef' for c in fingerprint.lower())
    
    def test_reproducibility_validation_workflow(self):
        """Test complete reproducibility validation workflow."""
        seed = 11335
        
        # Create initial environment
        manager1 = setup_reproducible_training(seed=seed)
        env_info1 = manager1.log_environment_info()
        hash1 = manager1.create_reproducibility_hash()
        
        # Simulate environment recreation
        manager2 = ReproducibilityManager(seed=seed)
        manager2.setup_deterministic_training()
        hash2 = manager2.create_reproducibility_hash()
        
        # Validation should pass
        assert manager2.validate_reproducibility(hash1)
        assert hash1 == hash2
        
        # Test with different seed (should fail)
        manager3 = ReproducibilityManager(seed=seed + 1)
        hash3 = manager3.create_reproducibility_hash()
        
        assert not manager3.validate_reproducibility(hash1)
        assert hash1 != hash3


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])