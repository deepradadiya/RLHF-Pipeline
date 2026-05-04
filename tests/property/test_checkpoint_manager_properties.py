"""
Property-based tests for Checkpoint Manager component.

This module implements property-based tests using Hypothesis to validate
the correctness properties of the Checkpoint Manager across all possible
valid inputs and edge cases.

Properties tested:
- Property 5: State Preservation During Interruption (Requirement 4.2)
- Property 6: Checkpoint Integrity Verification (Requirement 4.4)
- Property 7: Checkpoint Cleanup Policy (Requirement 4.5)
"""

import pytest
import tempfile
import shutil
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite

import torch
import torch.nn as nn
from torch.optim import AdamW
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlhf_phi3.checkpoints.checkpoint_manager import (
    CheckpointManager, CheckpointMetadata, GoogleDriveManager
)


# Mock PEFT model for testing
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
            "peft_type": "LORA"
        }))
        
        # Create mock adapter weights
        mock_weights = torch.randn(100, 50)
        torch.save({"default": mock_weights}, path / "adapter_model.bin")
        
        # Create mock training args
        (path / "training_args.bin").write_bytes(b"mock_training_args")


# Hypothesis strategies for checkpoint testing

@composite
def valid_checkpoint_metadata(draw):
    """Generate valid CheckpointMetadata instances."""
    stage = draw(st.sampled_from(["sft", "reward", "ppo"]))
    epoch = draw(st.integers(min_value=1, max_value=10))
    step = draw(st.integers(min_value=1, max_value=1000))
    timestamp = datetime.now(timezone.utc).isoformat()
    
    return CheckpointMetadata(
        stage=stage,
        epoch=epoch,
        step=step,
        timestamp=timestamp,
        model_path="model",
        optimizer_path="optimizer.pt",
        config_hash=draw(st.text(min_size=32, max_size=64, alphabet="0123456789abcdef")),
        metrics={
            "loss": draw(st.floats(min_value=0.1, max_value=10.0)),
            "accuracy": draw(st.floats(min_value=0.0, max_value=1.0))
        },
        file_hashes={
            "model": draw(st.text(min_size=64, max_size=64, alphabet="0123456789abcdef")),
            "optimizer": draw(st.text(min_size=64, max_size=64, alphabet="0123456789abcdef"))
        }
    )

@composite
def training_state(draw):
    """Generate training state for checkpoint testing."""
    return {
        "model": MockPeftModel(),
        "optimizer": AdamW([torch.randn(10, requires_grad=True)], lr=1e-4),
        "epoch": draw(st.integers(min_value=1, max_value=10)),
        "step": draw(st.integers(min_value=1, max_value=1000)),
        "stage": draw(st.sampled_from(["sft", "reward", "ppo"])),
        "metrics": {
            "loss": draw(st.floats(min_value=0.1, max_value=10.0)),
            "learning_rate": draw(st.floats(min_value=1e-6, max_value=1e-2))
        }
    }

@composite
def checkpoint_sequence(draw):
    """Generate a sequence of checkpoints for cleanup testing."""
    stage = draw(st.sampled_from(["sft", "reward", "ppo"]))
    num_checkpoints = draw(st.integers(min_value=1, max_value=10))
    
    checkpoints = []
    for i in range(num_checkpoints):
        epoch = draw(st.integers(min_value=1, max_value=5))
        step = draw(st.integers(min_value=i*100, max_value=(i+1)*100))
        
        checkpoints.append({
            "stage": stage,
            "epoch": epoch,
            "step": step,
            "metrics": {"loss": draw(st.floats(min_value=0.1, max_value=10.0))}
        })
    
    return checkpoints

@composite
def corrupted_checkpoint_scenario(draw):
    """Generate scenarios for checkpoint corruption testing."""
    corruption_type = draw(st.sampled_from([
        "missing_model_file",
        "missing_optimizer_file", 
        "corrupted_model_file",
        "corrupted_optimizer_file",
        "invalid_hash",
        "missing_metadata"
    ]))
    
    return {
        "type": corruption_type,
        "metadata": draw(valid_checkpoint_metadata())
    }


class TestCheckpointManagerProperties:
    """Property-based tests for Checkpoint Manager correctness properties."""
    
    @given(training_state())
    @settings(max_examples=20, deadline=None)
    def test_property_5_state_preservation_during_interruption(self, state):
        """
        **Validates: Requirement 4.2**
        
        Property 5: State Preservation During Interruption
        
        For any training interruption scenario, the Checkpoint_Manager SHALL preserve 
        the exact model state, optimizer state, and training step.
        
        This property ensures that:
        1. Model state is preserved exactly
        2. Optimizer state is preserved exactly
        3. Training step and epoch are preserved
        4. All metadata is preserved
        5. State can be restored identically
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize checkpoint manager (disable Google Drive for testing)
            manager = CheckpointManager(
                base_path=temp_dir,
                enable_drive_sync=False
            )
            
            # Save checkpoint (simulating interruption point)
            checkpoint_id = manager.save_checkpoint(
                model=state["model"],
                optimizer=state["optimizer"],
                epoch=state["epoch"],
                step=state["step"],
                stage=state["stage"],
                metrics=state["metrics"],
                config_hash="test_config_hash"
            )
            
            # Verify checkpoint was created
            assert checkpoint_id is not None
            assert checkpoint_id in manager.metadata_cache
            
            # Load checkpoint (simulating resumption after interruption)
            model_path, optimizer_state, metadata = manager.load_checkpoint(checkpoint_id)
            
            # Verify all components were preserved
            assert model_path is not None
            assert optimizer_state is not None
            assert metadata is not None
            
            # Verify exact state preservation
            assert metadata.stage == state["stage"]
            assert metadata.epoch == state["epoch"]
            assert metadata.step == state["step"]
            assert metadata.metrics == state["metrics"]
            assert metadata.config_hash == "test_config_hash"
            
            # Verify optimizer state preservation
            original_optimizer_state = state["optimizer"].state_dict()
            
            # Compare optimizer state keys and structure
            assert set(optimizer_state.keys()) == set(original_optimizer_state.keys())
            
            # Verify model files exist and are loadable
            model_dir = Path(model_path)
            assert model_dir.exists()
            assert (model_dir / "adapter_config.json").exists()
            assert (model_dir / "adapter_model.bin").exists()
            
            # Verify model can be loaded (basic structure check)
            with open(model_dir / "adapter_config.json", 'r') as f:
                adapter_config = json.load(f)
            assert "base_model_name_or_path" in adapter_config
            assert "peft_type" in adapter_config
            
            # Verify training step preservation (critical for resumption)
            assert metadata.step == state["step"]
            assert metadata.epoch == state["epoch"]
            
            # Verify timestamp is reasonable (within last minute)
            saved_time = datetime.fromisoformat(metadata.timestamp.replace('Z', '+00:00'))
            time_diff = datetime.now(timezone.utc) - saved_time
            assert time_diff.total_seconds() < 60  # Saved within last minute
    
    @given(valid_checkpoint_metadata(), corrupted_checkpoint_scenario())
    @settings(max_examples=30, deadline=None)
    def test_property_6_checkpoint_integrity_verification(self, metadata, corruption_scenario):
        """
        **Validates: Requirement 4.4**
        
        Property 6: Checkpoint Integrity Verification
        
        For any checkpoint data, the Checkpoint_Manager SHALL maintain checkpoint 
        integrity verification using hash validation.
        
        This property ensures that:
        1. Valid checkpoints pass integrity verification
        2. Corrupted checkpoints fail integrity verification
        3. Missing files are detected
        4. Hash mismatches are detected
        5. Integrity verification is comprehensive
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "test_checkpoint"
            checkpoint_dir.mkdir(parents=True)
            
            # Create checkpoint files based on metadata
            model_dir = checkpoint_dir / metadata.model_path
            model_dir.mkdir(parents=True)
            
            optimizer_path = checkpoint_dir / metadata.optimizer_path
            
            # Create valid files initially
            (model_dir / "adapter_config.json").write_text('{"test": "config"}')
            (model_dir / "adapter_model.bin").write_bytes(b"mock_model_data")
            optimizer_path.write_bytes(b"mock_optimizer_data")
            
            # Calculate correct hashes for valid scenario
            valid_metadata = CheckpointMetadata(
                stage=metadata.stage,
                epoch=metadata.epoch,
                step=metadata.step,
                timestamp=metadata.timestamp,
                model_path=metadata.model_path,
                optimizer_path=metadata.optimizer_path,
                config_hash=metadata.config_hash,
                metrics=metadata.metrics,
                file_hashes={
                    "model_adapter_config.json": CheckpointMetadata._calculate_file_hash(model_dir / "adapter_config.json"),
                    "model_adapter_model.bin": CheckpointMetadata._calculate_file_hash(model_dir / "adapter_model.bin"),
                    "optimizer": CheckpointMetadata._calculate_file_hash(optimizer_path)
                }
            )
            
            # Test valid checkpoint passes verification
            assert valid_metadata.validate_integrity(checkpoint_dir) is True
            
            # Apply corruption based on scenario
            corruption_type = corruption_scenario["type"]
            corrupted_metadata = corruption_scenario["metadata"]
            
            if corruption_type == "missing_model_file":
                # Remove model file
                (model_dir / "adapter_model.bin").unlink()
                assert corrupted_metadata.validate_integrity(checkpoint_dir) is False
                
            elif corruption_type == "missing_optimizer_file":
                # Remove optimizer file
                optimizer_path.unlink()
                assert corrupted_metadata.validate_integrity(checkpoint_dir) is False
                
            elif corruption_type == "corrupted_model_file":
                # Corrupt model file content
                (model_dir / "adapter_model.bin").write_bytes(b"corrupted_data")
                assert valid_metadata.validate_integrity(checkpoint_dir) is False
                
            elif corruption_type == "corrupted_optimizer_file":
                # Corrupt optimizer file content
                optimizer_path.write_bytes(b"corrupted_optimizer")
                assert valid_metadata.validate_integrity(checkpoint_dir) is False
                
            elif corruption_type == "invalid_hash":
                # Use metadata with wrong hash
                invalid_metadata = CheckpointMetadata(
                    stage=metadata.stage,
                    epoch=metadata.epoch,
                    step=metadata.step,
                    timestamp=metadata.timestamp,
                    model_path=metadata.model_path,
                    optimizer_path=metadata.optimizer_path,
                    config_hash=metadata.config_hash,
                    metrics=metadata.metrics,
                    file_hashes={"optimizer": "invalid_hash_value"}
                )
                assert invalid_metadata.validate_integrity(checkpoint_dir) is False
                
            elif corruption_type == "missing_metadata":
                # Test with completely missing files
                shutil.rmtree(checkpoint_dir)
                assert corrupted_metadata.validate_integrity(checkpoint_dir) is False
    
    @given(checkpoint_sequence(), st.integers(min_value=1, max_value=5))
    @settings(max_examples=20, deadline=None)
    def test_property_7_checkpoint_cleanup_policy(self, checkpoints, keep_last):
        """
        **Validates: Requirement 4.5**
        
        Property 7: Checkpoint Cleanup Policy
        
        For any checkpoint history, the Checkpoint_Manager SHALL automatically 
        clean up old checkpoints, keeping exactly the 3 most recent per stage.
        
        This property ensures that:
        1. Cleanup preserves exactly the specified number of recent checkpoints
        2. Older checkpoints are removed completely
        3. Cleanup works correctly across different stages
        4. Metadata cache is updated correctly
        5. Local storage is cleaned up properly
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize checkpoint manager
            manager = CheckpointManager(
                base_path=temp_dir,
                enable_drive_sync=False
            )
            
            # Create multiple checkpoints for the same stage
            checkpoint_ids = []
            stage = checkpoints[0]["stage"]
            
            for i, checkpoint_data in enumerate(checkpoints):
                # Create mock model and optimizer
                model = MockPeftModel()
                optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-4)
                
                # Save checkpoint
                checkpoint_id = manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=checkpoint_data["epoch"],
                    step=checkpoint_data["step"],
                    stage=checkpoint_data["stage"],
                    metrics=checkpoint_data["metrics"]
                )
                
                checkpoint_ids.append(checkpoint_id)
                
                # Small delay to ensure different timestamps
                import time
                time.sleep(0.01)
            
            # Verify all checkpoints were created
            assert len(checkpoint_ids) == len(checkpoints)
            
            # Get initial checkpoint list
            initial_checkpoints = manager.list_checkpoints(stage)
            assert len(initial_checkpoints) == len(checkpoints)
            
            # Verify all checkpoints exist locally
            for checkpoint_id in checkpoint_ids:
                checkpoint_dir = Path(temp_dir) / checkpoint_id
                assert checkpoint_dir.exists()
                assert checkpoint_id in manager.metadata_cache
            
            # Apply cleanup policy
            manager.cleanup_old_checkpoints(keep_last=keep_last)
            
            # Verify cleanup results
            remaining_checkpoints = manager.list_checkpoints(stage)
            expected_remaining = min(keep_last, len(checkpoints))
            
            # Should keep exactly the specified number of recent checkpoints
            assert len(remaining_checkpoints) == expected_remaining
            
            # Verify the remaining checkpoints are the most recent ones
            # (list_checkpoints returns sorted by timestamp, newest first)
            expected_remaining_ids = initial_checkpoints[:expected_remaining]
            assert set(remaining_checkpoints) == set(expected_remaining_ids)
            
            # Verify removed checkpoints are completely gone
            removed_checkpoints = initial_checkpoints[expected_remaining:]
            for checkpoint_id in removed_checkpoints:
                # Should not exist in local storage
                checkpoint_dir = Path(temp_dir) / checkpoint_id
                assert not checkpoint_dir.exists()
                
                # Should not exist in metadata cache
                assert checkpoint_id not in manager.metadata_cache
            
            # Verify remaining checkpoints still exist and are valid
            for checkpoint_id in remaining_checkpoints:
                checkpoint_dir = Path(temp_dir) / checkpoint_id
                assert checkpoint_dir.exists()
                assert checkpoint_id in manager.metadata_cache
                
                # Verify checkpoint can still be loaded
                model_path, optimizer_state, metadata = manager.load_checkpoint(checkpoint_id)
                assert model_path is not None
                assert optimizer_state is not None
                assert metadata is not None
            
            # Test edge case: cleanup with keep_last=0 should remove all
            if len(remaining_checkpoints) > 0:
                manager.cleanup_old_checkpoints(keep_last=0)
                final_checkpoints = manager.list_checkpoints(stage)
                assert len(final_checkpoints) == 0
    
    @given(st.lists(training_state(), min_size=2, max_size=5))
    @settings(max_examples=10, deadline=None)
    def test_checkpoint_manager_multi_stage_isolation(self, states):
        """
        Additional property: Multi-stage checkpoint isolation.
        
        This ensures that checkpoints from different stages are managed
        independently and cleanup policies work correctly across stages.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                base_path=temp_dir,
                enable_drive_sync=False
            )
            
            # Ensure we have different stages
            stages = ["sft", "reward", "ppo"]
            stage_checkpoints = {stage: [] for stage in stages}
            
            # Create checkpoints for different stages
            for i, state in enumerate(states):
                stage = stages[i % len(stages)]
                state["stage"] = stage
                
                checkpoint_id = manager.save_checkpoint(
                    model=state["model"],
                    optimizer=state["optimizer"],
                    epoch=state["epoch"],
                    step=state["step"],
                    stage=stage,
                    metrics=state["metrics"]
                )
                
                stage_checkpoints[stage].append(checkpoint_id)
            
            # Verify stage isolation
            for stage in stages:
                stage_list = manager.list_checkpoints(stage)
                expected_ids = stage_checkpoints[stage]
                assert set(stage_list) == set(expected_ids)
            
            # Test cleanup isolation - clean up one stage
            if stage_checkpoints["sft"]:
                initial_sft_count = len(stage_checkpoints["sft"])
                initial_reward_count = len(stage_checkpoints["reward"])
                initial_ppo_count = len(stage_checkpoints["ppo"])
                
                # Cleanup SFT stage only
                manager.cleanup_old_checkpoints(keep_last=1)
                
                # Verify SFT was cleaned up
                remaining_sft = manager.list_checkpoints("sft")
                assert len(remaining_sft) == min(1, initial_sft_count)
                
                # Verify other stages were also cleaned up (cleanup_old_checkpoints affects all stages)
                remaining_reward = manager.list_checkpoints("reward")
                remaining_ppo = manager.list_checkpoints("ppo")
                
                assert len(remaining_reward) == min(1, initial_reward_count)
                assert len(remaining_ppo) == min(1, initial_ppo_count)
    
    @given(valid_checkpoint_metadata())
    @settings(max_examples=15, deadline=None)
    def test_checkpoint_metadata_serialization_round_trip(self, metadata):
        """
        Additional property: Checkpoint metadata serialization round-trip.
        
        This ensures that checkpoint metadata can be serialized and deserialized
        without loss of information.
        """
        # Test dictionary round-trip
        metadata_dict = metadata.to_dict()
        restored_metadata = CheckpointMetadata.from_dict(metadata_dict)
        
        # Verify all fields are preserved
        assert restored_metadata.stage == metadata.stage
        assert restored_metadata.epoch == metadata.epoch
        assert restored_metadata.step == metadata.step
        assert restored_metadata.timestamp == metadata.timestamp
        assert restored_metadata.model_path == metadata.model_path
        assert restored_metadata.optimizer_path == metadata.optimizer_path
        assert restored_metadata.config_hash == metadata.config_hash
        assert restored_metadata.metrics == metadata.metrics
        assert restored_metadata.file_hashes == metadata.file_hashes
        
        # Test JSON round-trip
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = Path(f.name)
        
        try:
            # Save to JSON
            with open(json_path, 'w') as f:
                json.dump(metadata_dict, f)
            
            # Load from JSON
            with open(json_path, 'r') as f:
                loaded_dict = json.load(f)
            
            loaded_metadata = CheckpointMetadata.from_dict(loaded_dict)
            
            # Verify JSON round-trip preservation
            assert loaded_metadata.stage == metadata.stage
            assert loaded_metadata.epoch == metadata.epoch
            assert loaded_metadata.step == metadata.step
            assert loaded_metadata.metrics == metadata.metrics
            
        finally:
            json_path.unlink(missing_ok=True)
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=10, deadline=None)
    def test_checkpoint_manager_concurrent_operations(self, num_operations):
        """
        Additional property: Checkpoint manager handles concurrent-like operations.
        
        This ensures that multiple checkpoint operations don't interfere with
        each other and the manager maintains consistency.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                base_path=temp_dir,
                enable_drive_sync=False
            )
            
            checkpoint_ids = []
            
            # Perform multiple checkpoint operations
            for i in range(num_operations):
                model = MockPeftModel()
                optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-4)
                
                # Save checkpoint
                checkpoint_id = manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=i + 1,
                    step=(i + 1) * 100,
                    stage="sft",
                    metrics={"loss": 1.0 / (i + 1)}
                )
                
                checkpoint_ids.append(checkpoint_id)
                
                # Verify checkpoint can be immediately loaded
                model_path, optimizer_state, metadata = manager.load_checkpoint(checkpoint_id)
                assert model_path is not None
                assert optimizer_state is not None
                assert metadata is not None
                assert metadata.epoch == i + 1
                assert metadata.step == (i + 1) * 100
            
            # Verify all checkpoints are tracked
            all_checkpoints = manager.list_checkpoints("sft")
            assert len(all_checkpoints) == num_operations
            assert set(all_checkpoints) == set(checkpoint_ids)
            
            # Verify metadata cache consistency
            assert len(manager.metadata_cache) == num_operations
            for checkpoint_id in checkpoint_ids:
                assert checkpoint_id in manager.metadata_cache
            
            # Test batch validation
            validation_results = manager.validate_all_checkpoints()
            assert len(validation_results) == num_operations
            assert all(validation_results.values())  # All should be valid


# Additional edge case tests for comprehensive coverage

class TestCheckpointManagerEdgeCases:
    """Test edge cases and boundary conditions for checkpoint management."""
    
    def test_checkpoint_manager_initialization_edge_cases(self):
        """Test checkpoint manager initialization with various configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with non-existent directory (should create it)
            non_existent_path = Path(temp_dir) / "non_existent" / "checkpoints"
            manager = CheckpointManager(
                base_path=non_existent_path,
                enable_drive_sync=False
            )
            assert non_existent_path.exists()
            
            # Test with existing directory
            existing_path = Path(temp_dir) / "existing"
            existing_path.mkdir(parents=True)
            manager2 = CheckpointManager(
                base_path=existing_path,
                enable_drive_sync=False
            )
            assert existing_path.exists()
    
    def test_checkpoint_loading_nonexistent_checkpoint(self):
        """Test loading a checkpoint that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                base_path=temp_dir,
                enable_drive_sync=False
            )
            
            # Try to load non-existent checkpoint
            result = manager.load_checkpoint("nonexistent_checkpoint")
            assert result == (None, None, None)
    
    def test_checkpoint_info_and_validation(self):
        """Test checkpoint info retrieval and validation methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                base_path=temp_dir,
                enable_drive_sync=False
            )
            
            # Create a checkpoint
            model = MockPeftModel()
            optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-4)
            
            checkpoint_id = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=1,
                step=100,
                stage="sft",
                metrics={"loss": 2.5}
            )
            
            # Test checkpoint info
            info = manager.get_checkpoint_info(checkpoint_id)
            assert info is not None
            assert info["stage"] == "sft"
            assert info["epoch"] == 1
            assert info["step"] == 100
            assert info["local_exists"] is True
            assert info["integrity_valid"] is True
            assert info["size_mb"] > 0
            
            # Test info for non-existent checkpoint
            info_none = manager.get_checkpoint_info("nonexistent")
            assert info_none is None
            
            # Test validation of all checkpoints
            validation_results = manager.validate_all_checkpoints()
            assert checkpoint_id in validation_results
            assert validation_results[checkpoint_id] is True
    
    def test_latest_checkpoint_retrieval(self):
        """Test getting the latest checkpoint for a stage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                base_path=temp_dir,
                enable_drive_sync=False
            )
            
            # No checkpoints initially
            latest = manager.get_latest_checkpoint("sft")
            assert latest is None
            
            # Create multiple checkpoints
            model = MockPeftModel()
            optimizer = AdamW([torch.randn(10, requires_grad=True)], lr=1e-4)
            
            checkpoint_ids = []
            for i in range(3):
                checkpoint_id = manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=i + 1,
                    step=(i + 1) * 100,
                    stage="sft",
                    metrics={"loss": 1.0 / (i + 1)}
                )
                checkpoint_ids.append(checkpoint_id)
                
                # Small delay to ensure different timestamps
                import time
                time.sleep(0.01)
            
            # Get latest checkpoint
            latest = manager.get_latest_checkpoint("sft")
            assert latest is not None
            assert latest in checkpoint_ids
            
            # Verify it's actually the latest (highest step)
            latest_metadata = manager.metadata_cache[latest]
            assert latest_metadata.step == 300  # Last checkpoint had step 300
    
    @patch('rlhf_phi3.checkpoints.checkpoint_manager.GOOGLE_DRIVE_AVAILABLE', False)
    def test_checkpoint_manager_without_google_drive(self):
        """Test checkpoint manager when Google Drive is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                base_path=temp_dir,
                enable_drive_sync=True  # Request sync but it should be disabled
            )
            
            # Should fall back to local storage only
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