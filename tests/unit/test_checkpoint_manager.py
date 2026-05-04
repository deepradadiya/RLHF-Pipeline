"""
Unit tests for Checkpoint Manager component.

This module implements unit tests for the Checkpoint Manager to validate
specific examples, edge cases, and component functionality.

Tests cover:
- Checkpoint save/load with state preservation (Requirement 4.2)
- Google Drive synchronization (Requirement 4.1)
- Cleanup policy enforcement (Requirement 4.5)
- Integrity verification (Requirement 4.4)
"""

import pytest
import tempfile
import shutil
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.optim import AdamW
from peft import PeftModel, LoraConfig, get_peft_model

from rlhf_phi3.checkpoints.checkpoint_manager import (
    CheckpointManager, CheckpointMetadata, GoogleDriveManager
)


class MockPeftModel:
    """Mock PEFT model for testing purposes."""
    
    def __init__(self, base_model_name: str = "test-model"):
        self.base_model_name = base_model_name
        self.config = Mock()
        self.config.name_or_path = base_model_name
        
    def save_pretrained(self, path: Path):
        """Mock save_pretrained method."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Create mock model files
        (path / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": self.base_model_name,
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32
        }))
        
        # Create mock adapter weights
        mock_weights = torch.randn(100, 50)
        torch.save({"default": mock_weights}, path / "adapter_model.bin")
        
        # Create mock training args
        (path / "training_args.bin").write_bytes(b"mock_training_args_data")


class TestCheckpointManager:
    """Unit tests for CheckpointManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CheckpointManager(
            base_path=self.temp_dir,
            enable_drive_sync=False  # Disable for unit tests
        )
        
        # Create mock model and optimizer
        self.mock_model = MockPeftModel("microsoft/Phi-3-mini-4k-instruct")
        self.mock_optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-4)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_save_basic_functionality(self):
        """Test basic checkpoint saving functionality."""
        # Save checkpoint
        checkpoint_id = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            epoch=1,
            step=100,
            stage="sft",
            metrics={"loss": 2.5, "accuracy": 0.85},
            config_hash="test_config_hash_123"
        )
        
        # Verify checkpoint ID is generated
        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)
        assert "sft_epoch_1_step_100" in checkpoint_id
        
        # Verify checkpoint directory exists
        checkpoint_dir = Path(self.temp_dir) / checkpoint_id
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()
        
        # Verify model files exist
        model_dir = checkpoint_dir / "model"
        assert model_dir.exists()
        assert (model_dir / "adapter_config.json").exists()
        assert (model_dir / "adapter_model.bin").exists()
        
        # Verify optimizer file exists
        optimizer_file = checkpoint_dir / "optimizer.pt"
        assert optimizer_file.exists()
        
        # Verify metadata file exists
        metadata_file = checkpoint_dir / "metadata.json"
        assert metadata_file.exists()
        
        # Verify metadata content
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        assert metadata_dict["stage"] == "sft"
        assert metadata_dict["epoch"] == 1
        assert metadata_dict["step"] == 100
        assert metadata_dict["metrics"]["loss"] == 2.5
        assert metadata_dict["metrics"]["accuracy"] == 0.85
        assert metadata_dict["config_hash"] == "test_config_hash_123"
        
        # Verify checkpoint is in cache
        assert checkpoint_id in self.manager.metadata_cache
    
    def test_checkpoint_load_basic_functionality(self):
        """Test basic checkpoint loading functionality."""
        # First save a checkpoint
        checkpoint_id = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            epoch=2,
            step=200,
            stage="reward",
            metrics={"loss": 1.8, "reward_accuracy": 0.92}
        )
        
        # Load the checkpoint
        model_path, optimizer_state, metadata = self.manager.load_checkpoint(checkpoint_id)
        
        # Verify loaded components
        assert model_path is not None
        assert isinstance(model_path, str)
        assert Path(model_path).exists()
        
        assert optimizer_state is not None
        assert isinstance(optimizer_state, dict)
        
        assert metadata is not None
        assert isinstance(metadata, CheckpointMetadata)
        
        # Verify metadata content
        assert metadata.stage == "reward"
        assert metadata.epoch == 2
        assert metadata.step == 200
        assert metadata.metrics["loss"] == 1.8
        assert metadata.metrics["reward_accuracy"] == 0.92
        
        # Verify model can be loaded from path
        model_dir = Path(model_path)
        assert (model_dir / "adapter_config.json").exists()
        
        with open(model_dir / "adapter_config.json", 'r') as f:
            adapter_config = json.load(f)
        assert adapter_config["base_model_name_or_path"] == "microsoft/Phi-3-mini-4k-instruct"
        assert adapter_config["peft_type"] == "LORA"
    
    def test_checkpoint_save_load_state_preservation(self):
        """
        Test checkpoint save/load with exact state preservation.
        Validates Requirement 4.2: Preserve exact model state, optimizer state, and training step.
        """
        # Create specific optimizer state
        optimizer = AdamW([torch.randn(5, requires_grad=True)], lr=2e-4, weight_decay=0.01)
        
        # Perform some optimization steps to create state
        for _ in range(3):
            loss = torch.sum(torch.randn(5) ** 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        original_optimizer_state = optimizer.state_dict()
        
        # Save checkpoint with specific state
        checkpoint_id = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=optimizer,
            epoch=5,
            step=500,
            stage="ppo",
            metrics={"policy_loss": 0.3, "value_loss": 0.15},
            config_hash="specific_config_hash"
        )
        
        # Load checkpoint
        model_path, loaded_optimizer_state, metadata = self.manager.load_checkpoint(checkpoint_id)
        
        # Verify exact state preservation
        assert metadata.epoch == 5
        assert metadata.step == 500
        assert metadata.stage == "ppo"
        assert metadata.config_hash == "specific_config_hash"
        assert metadata.metrics["policy_loss"] == 0.3
        assert metadata.metrics["value_loss"] == 0.15
        
        # Verify optimizer state preservation
        assert set(loaded_optimizer_state.keys()) == set(original_optimizer_state.keys())
        
        # Verify state dict structure is preserved
        if 'state' in original_optimizer_state:
            assert 'state' in loaded_optimizer_state
        if 'param_groups' in original_optimizer_state:
            assert 'param_groups' in loaded_optimizer_state
            assert len(loaded_optimizer_state['param_groups']) == len(original_optimizer_state['param_groups'])
    
    def test_checkpoint_integrity_verification(self):
        """
        Test checkpoint integrity verification using hash validation.
        Validates Requirement 4.4: Maintain checkpoint integrity verification using hash validation.
        """
        # Save checkpoint
        checkpoint_id = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            epoch=1,
            step=100,
            stage="sft"
        )
        
        # Verify integrity passes for valid checkpoint
        metadata = self.manager.metadata_cache[checkpoint_id]
        checkpoint_dir = Path(self.temp_dir) / checkpoint_id
        
        assert metadata.validate_integrity(checkpoint_dir) is True
        
        # Test integrity failure scenarios
        
        # 1. Corrupt model file
        model_file = checkpoint_dir / "model" / "adapter_model.bin"
        original_content = model_file.read_bytes()
        model_file.write_bytes(b"corrupted_content")
        
        assert metadata.validate_integrity(checkpoint_dir) is False
        
        # Restore file
        model_file.write_bytes(original_content)
        assert metadata.validate_integrity(checkpoint_dir) is True
        
        # 2. Corrupt optimizer file
        optimizer_file = checkpoint_dir / "optimizer.pt"
        original_optimizer_content = optimizer_file.read_bytes()
        optimizer_file.write_bytes(b"corrupted_optimizer")
        
        assert metadata.validate_integrity(checkpoint_dir) is False
        
        # Restore file
        optimizer_file.write_bytes(original_optimizer_content)
        assert metadata.validate_integrity(checkpoint_dir) is True
        
        # 3. Missing file
        model_file.unlink()
        assert metadata.validate_integrity(checkpoint_dir) is False
    
    def test_checkpoint_cleanup_policy_enforcement(self):
        """
        Test cleanup policy enforcement.
        Validates Requirement 4.5: Automatically clean up old checkpoints, keeping the 3 most recent per stage.
        """
        # Create multiple checkpoints for the same stage
        checkpoint_ids = []
        
        for i in range(5):
            checkpoint_id = self.manager.save_checkpoint(
                model=self.mock_model,
                optimizer=self.mock_optimizer,
                epoch=i + 1,
                step=(i + 1) * 100,
                stage="sft",
                metrics={"loss": 5.0 - i}  # Decreasing loss
            )
            checkpoint_ids.append(checkpoint_id)
            
            # Small delay to ensure different timestamps
            import time
            time.sleep(0.01)
        
        # Verify all checkpoints exist
        assert len(self.manager.list_checkpoints("sft")) == 5
        for checkpoint_id in checkpoint_ids:
            checkpoint_dir = Path(self.temp_dir) / checkpoint_id
            assert checkpoint_dir.exists()
        
        # Apply cleanup policy (keep 3 most recent)
        self.manager.cleanup_old_checkpoints(keep_last=3)
        
        # Verify only 3 checkpoints remain
        remaining_checkpoints = self.manager.list_checkpoints("sft")
        assert len(remaining_checkpoints) == 3
        
        # Verify the 3 most recent checkpoints are kept
        # (list_checkpoints returns sorted by timestamp, newest first)
        expected_remaining = checkpoint_ids[-3:]  # Last 3 created
        
        for checkpoint_id in expected_remaining:
            assert checkpoint_id in remaining_checkpoints
            checkpoint_dir = Path(self.temp_dir) / checkpoint_id
            assert checkpoint_dir.exists()
        
        # Verify older checkpoints are removed
        removed_checkpoints = checkpoint_ids[:-3]  # First 2 created
        
        for checkpoint_id in removed_checkpoints:
            assert checkpoint_id not in remaining_checkpoints
            checkpoint_dir = Path(self.temp_dir) / checkpoint_id
            assert not checkpoint_dir.exists()
            assert checkpoint_id not in self.manager.metadata_cache
    
    def test_checkpoint_cleanup_multi_stage_isolation(self):
        """Test that cleanup policy works correctly across different stages."""
        # Create checkpoints for different stages
        sft_checkpoints = []
        reward_checkpoints = []
        
        # Create 4 SFT checkpoints
        for i in range(4):
            checkpoint_id = self.manager.save_checkpoint(
                model=self.mock_model,
                optimizer=self.mock_optimizer,
                epoch=i + 1,
                step=(i + 1) * 100,
                stage="sft"
            )
            sft_checkpoints.append(checkpoint_id)
            time.sleep(0.01)
        
        # Create 3 reward checkpoints
        for i in range(3):
            checkpoint_id = self.manager.save_checkpoint(
                model=self.mock_model,
                optimizer=self.mock_optimizer,
                epoch=i + 1,
                step=(i + 1) * 50,
                stage="reward"
            )
            reward_checkpoints.append(checkpoint_id)
            time.sleep(0.01)
        
        # Verify initial state
        assert len(self.manager.list_checkpoints("sft")) == 4
        assert len(self.manager.list_checkpoints("reward")) == 3
        
        # Apply cleanup (keep 2 per stage)
        self.manager.cleanup_old_checkpoints(keep_last=2)
        
        # Verify cleanup results
        remaining_sft = self.manager.list_checkpoints("sft")
        remaining_reward = self.manager.list_checkpoints("reward")
        
        assert len(remaining_sft) == 2
        assert len(remaining_reward) == 2
        
        # Verify correct checkpoints are kept (most recent)
        assert sft_checkpoints[-2] in remaining_sft
        assert sft_checkpoints[-1] in remaining_sft
        assert reward_checkpoints[-2] in remaining_reward
        assert reward_checkpoints[-1] in remaining_reward
    
    def test_checkpoint_list_and_latest_retrieval(self):
        """Test checkpoint listing and latest checkpoint retrieval."""
        # Initially no checkpoints
        assert len(self.manager.list_checkpoints()) == 0
        assert len(self.manager.list_checkpoints("sft")) == 0
        assert self.manager.get_latest_checkpoint("sft") is None
        
        # Create checkpoints for different stages
        sft_id1 = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            epoch=1, step=100, stage="sft"
        )
        time.sleep(0.01)
        
        reward_id1 = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            epoch=1, step=50, stage="reward"
        )
        time.sleep(0.01)
        
        sft_id2 = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            epoch=2, step=200, stage="sft"
        )
        
        # Test listing all checkpoints
        all_checkpoints = self.manager.list_checkpoints()
        assert len(all_checkpoints) == 3
        assert sft_id1 in all_checkpoints
        assert sft_id2 in all_checkpoints
        assert reward_id1 in all_checkpoints
        
        # Test listing by stage
        sft_checkpoints = self.manager.list_checkpoints("sft")
        assert len(sft_checkpoints) == 2
        assert sft_id1 in sft_checkpoints
        assert sft_id2 in sft_checkpoints
        assert reward_id1 not in sft_checkpoints
        
        reward_checkpoints = self.manager.list_checkpoints("reward")
        assert len(reward_checkpoints) == 1
        assert reward_id1 in reward_checkpoints
        
        # Test latest checkpoint retrieval
        latest_sft = self.manager.get_latest_checkpoint("sft")
        assert latest_sft == sft_id2  # Most recent SFT checkpoint
        
        latest_reward = self.manager.get_latest_checkpoint("reward")
        assert latest_reward == reward_id1
        
        latest_ppo = self.manager.get_latest_checkpoint("ppo")
        assert latest_ppo is None  # No PPO checkpoints
    
    def test_checkpoint_info_retrieval(self):
        """Test detailed checkpoint information retrieval."""
        # Create checkpoint
        checkpoint_id = self.manager.save_checkpoint(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            epoch=3,
            step=300,
            stage="ppo",
            metrics={"policy_loss": 0.25, "value_loss": 0.12}
        )
        
        # Get checkpoint info
        info = self.manager.get_checkpoint_info(checkpoint_id)
        
        assert info is not None
        assert info["id"] == checkpoint_id
        assert info["stage"] == "ppo"
        assert info["epoch"] == 3
        assert info["step"] == 300
        assert info["metrics"]["policy_loss"] == 0.25
        assert info["metrics"]["value_loss"] == 0.12
        assert info["local_exists"] is True
        assert info["integrity_valid"] is True
        assert info["size_mb"] > 0
        
        # Test info for non-existent checkpoint
        info_none = self.manager.get_checkpoint_info("nonexistent")
        assert info_none is None
    
    def test_checkpoint_validation_all(self):
        """Test validation of all checkpoints."""
        # Create multiple checkpoints
        checkpoint_ids = []
        for i in range(3):
            checkpoint_id = self.manager.save_checkpoint(
                model=self.mock_model,
                optimizer=self.mock_optimizer,
                epoch=i + 1,
                step=(i + 1) * 100,
                stage="sft"
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Validate all checkpoints
        validation_results = self.manager.validate_all_checkpoints()
        
        assert len(validation_results) == 3
        for checkpoint_id in checkpoint_ids:
            assert checkpoint_id in validation_results
            assert validation_results[checkpoint_id] is True
        
        # Corrupt one checkpoint
        checkpoint_dir = Path(self.temp_dir) / checkpoint_ids[1]
        optimizer_file = checkpoint_dir / "optimizer.pt"
        optimizer_file.write_bytes(b"corrupted")
        
        # Re-validate
        validation_results = self.manager.validate_all_checkpoints()
        assert validation_results[checkpoint_ids[0]] is True
        assert validation_results[checkpoint_ids[1]] is False  # Corrupted
        assert validation_results[checkpoint_ids[2]] is True
    
    def test_checkpoint_load_nonexistent(self):
        """Test loading a checkpoint that doesn't exist."""
        result = self.manager.load_checkpoint("nonexistent_checkpoint_id")
        assert result == (None, None, None)
    
    def test_checkpoint_save_error_handling(self):
        """Test error handling during checkpoint save operations."""
        # Test with invalid stage
        with pytest.raises(Exception):
            # Create a scenario that would cause save to fail
            # Mock the save_pretrained to raise an exception
            mock_model = Mock()
            mock_model.save_pretrained.side_effect = Exception("Save failed")
            
            self.manager.save_checkpoint(
                model=mock_model,
                optimizer=self.mock_optimizer,
                epoch=1,
                step=100,
                stage="sft"
            )
    
    def test_checkpoint_metadata_serialization(self):
        """Test checkpoint metadata serialization and deserialization."""
        # Create metadata
        metadata = CheckpointMetadata(
            stage="sft",
            epoch=2,
            step=250,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_path="model",
            optimizer_path="optimizer.pt",
            config_hash="test_hash_123",
            metrics={"loss": 1.5, "accuracy": 0.88},
            file_hashes={"model": "abc123", "optimizer": "def456"}
        )
        
        # Test to_dict
        metadata_dict = metadata.to_dict()
        assert metadata_dict["stage"] == "sft"
        assert metadata_dict["epoch"] == 2
        assert metadata_dict["step"] == 250
        assert metadata_dict["metrics"]["loss"] == 1.5
        
        # Test from_dict
        restored_metadata = CheckpointMetadata.from_dict(metadata_dict)
        assert restored_metadata.stage == metadata.stage
        assert restored_metadata.epoch == metadata.epoch
        assert restored_metadata.step == metadata.step
        assert restored_metadata.metrics == metadata.metrics
        assert restored_metadata.file_hashes == metadata.file_hashes


class TestGoogleDriveManager:
    """Unit tests for GoogleDriveManager class."""
    
    @patch('rlhf_phi3.checkpoints.checkpoint_manager.GOOGLE_DRIVE_AVAILABLE', False)
    def test_google_drive_manager_unavailable(self):
        """Test GoogleDriveManager when Google Drive dependencies are unavailable."""
        manager = GoogleDriveManager()
        assert manager.service is None
        assert manager.authenticated is False
        
        # All operations should return None/False
        assert manager.upload_file(Path("test.txt"), "remote.txt") is None
        assert manager.download_file("file_id", Path("local.txt")) is False
        assert manager.create_folder("test_folder") is None
        assert manager.list_files() == []
    
    @patch('rlhf_phi3.checkpoints.checkpoint_manager.GOOGLE_DRIVE_AVAILABLE', True)
    @patch('rlhf_phi3.checkpoints.checkpoint_manager.build')
    @patch('rlhf_phi3.checkpoints.checkpoint_manager.Credentials')
    def test_google_drive_manager_authentication_success(self, mock_credentials, mock_build):
        """Test successful Google Drive authentication."""
        # Mock credentials
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        
        # Mock service
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Create temporary token file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            token_path = f.name
            json.dump({"token": "test_token"}, f)
        
        try:
            manager = GoogleDriveManager(token_path=token_path)
            assert manager.authenticated is True
            assert manager.service == mock_service
        finally:
            Path(token_path).unlink(missing_ok=True)
    
    @patch('rlhf_phi3.checkpoints.checkpoint_manager.GOOGLE_DRIVE_AVAILABLE', True)
    @patch('rlhf_phi3.checkpoints.checkpoint_manager.os.path.exists')
    def test_google_drive_manager_authentication_failure(self, mock_exists):
        """Test Google Drive authentication failure scenarios."""
        # Mock missing credentials file
        mock_exists.return_value = False
        
        manager = GoogleDriveManager(credentials_path="nonexistent.json")
        assert manager.authenticated is False
        assert manager.service is None


class TestCheckpointManagerGoogleDriveIntegration:
    """Unit tests for CheckpointManager Google Drive integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('rlhf_phi3.checkpoints.checkpoint_manager.GOOGLE_DRIVE_AVAILABLE', False)
    def test_checkpoint_manager_google_drive_disabled(self):
        """
        Test CheckpointManager when Google Drive is disabled.
        Validates fallback to local storage (Requirement 4.1).
        """
        manager = CheckpointManager(
            base_path=self.temp_dir,
            enable_drive_sync=True  # Request sync but should be disabled
        )
        
        # Should fall back to local storage
        assert manager.enable_drive_sync is False
        assert manager.drive_manager is None
        
        # Should still work for local operations
        model = MockPeftModel()
        optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-4)
        
        checkpoint_id = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            stage="sft"
        )
        
        assert checkpoint_id is not None
        
        # Should be able to load
        model_path, optimizer_state, metadata = manager.load_checkpoint(checkpoint_id)
        assert model_path is not None
        assert optimizer_state is not None
        assert metadata is not None
    
    @patch('rlhf_phi3.checkpoints.checkpoint_manager.GOOGLE_DRIVE_AVAILABLE', True)
    def test_checkpoint_manager_google_drive_sync_methods(self):
        """Test Google Drive sync method interfaces."""
        # Create manager with mocked Google Drive
        with patch.object(GoogleDriveManager, '__init__', return_value=None):
            with patch.object(GoogleDriveManager, 'authenticated', True):
                manager = CheckpointManager(
                    base_path=self.temp_dir,
                    enable_drive_sync=True
                )
                
                # Mock the drive manager
                manager.drive_manager = Mock()
                manager.drive_manager.authenticated = True
                manager.drive_folder_id = "test_folder_id"
                
                # Test sync_to_drive method exists and can be called
                result = manager.sync_to_drive()
                # Should return False since no checkpoints exist
                assert result is True  # Empty sync is considered successful
    
    def test_checkpoint_manager_initialization_with_drive_config(self):
        """Test CheckpointManager initialization with Google Drive configuration."""
        manager = CheckpointManager(
            base_path=self.temp_dir,
            google_drive_folder="custom-checkpoint-folder",
            credentials_path="custom_credentials.json",
            enable_drive_sync=False  # Explicitly disable for testing
        )
        
        assert manager.google_drive_folder == "custom-checkpoint-folder"
        assert manager.enable_drive_sync is False
        assert manager.drive_manager is None
    
    def test_checkpoint_manager_metadata_cache_persistence(self):
        """Test that checkpoint metadata cache is persisted and loaded correctly."""
        # Create manager and save checkpoint
        manager1 = CheckpointManager(
            base_path=self.temp_dir,
            enable_drive_sync=False
        )
        
        model = MockPeftModel()
        optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-4)
        
        checkpoint_id = manager1.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            stage="sft",
            metrics={"loss": 2.0}
        )
        
        # Verify cache file exists
        cache_file = Path(self.temp_dir) / "checkpoint_cache.json"
        assert cache_file.exists()
        
        # Create new manager instance (simulating restart)
        manager2 = CheckpointManager(
            base_path=self.temp_dir,
            enable_drive_sync=False
        )
        
        # Verify cache was loaded
        assert checkpoint_id in manager2.metadata_cache
        assert len(manager2.list_checkpoints()) == 1
        
        # Verify checkpoint can be loaded
        model_path, optimizer_state, metadata = manager2.load_checkpoint(checkpoint_id)
        assert model_path is not None
        assert metadata.metrics["loss"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__])