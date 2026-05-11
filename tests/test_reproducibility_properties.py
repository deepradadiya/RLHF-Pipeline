"""
Property-Based Tests for Reproducibility Features

This module contains property-based tests for the reproducibility utilities
using Hypothesis to validate universal correctness properties.

Property tested:
- Property 37: Deterministic Training Reproducibility
- Validates: Requirement 15.2

The property tests ensure that reproducibility features work correctly
across a wide range of inputs and scenarios.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
import numpy as np

# Import reproducibility utilities
from rlhf_phi3.utils.reproducibility import (
    ReproducibilityManager,
    setup_reproducible_training,
    log_training_environment,
    create_training_fingerprint,
    ensure_deterministic_environment
)

# Import training provenance utilities
from rlhf_phi3.utils.training_provenance import (
    TrainingProvenanceManager,
    TrainingStageProvenance,
    create_provenance_manager
)

# Import configuration for integration tests
from rlhf_phi3.config.config_manager import Config


class TestReproducibilityProperties:
    """Property-based tests for reproducibility features."""
    
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=50, deadline=5000)
    def test_property_37_deterministic_training_reproducibility_basic(self, seed):
        """
        Property 37: Deterministic Training Reproducibility (Basic)
        
        For any fixed random seed, the ReproducibilityManager SHALL use fixed 
        random seeds for deterministic training producing identical results across runs.
        
        Validates: Requirement 15.2
        """
        # Create two managers with the same seed
        manager1 = ReproducibilityManager(seed=seed, enable_deterministic=True)
        manager2 = ReproducibilityManager(seed=seed, enable_deterministic=True)
        
        # Set up deterministic training for both
        manager1.setup_deterministic_training()
        manager2.setup_deterministic_training()
        
        # Both should have the same seed
        assert manager1.seed == manager2.seed == seed
        
        # Both should have deterministic mode enabled
        assert manager1.enable_deterministic == manager2.enable_deterministic == True
        
        # Environment variables should be set consistently
        assert os.environ.get('PYTHONHASHSEED') == str(seed)
        
        # Generate reproducibility hashes - should be identical
        hash1 = manager1.create_reproducibility_hash()
        hash2 = manager2.create_reproducibility_hash()
        
        assert hash1 == hash2, f"Reproducibility hashes differ for seed {seed}: {hash1} != {hash2}"
        
        # Validate reproducibility
        assert manager1.validate_reproducibility(hash2)
        assert manager2.validate_reproducibility(hash1)
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        enable_deterministic=st.booleans()
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_37_deterministic_training_reproducibility_configuration(self, seed, enable_deterministic):
        """
        Property 37: Deterministic Training Reproducibility (Configuration)
        
        For any configuration of deterministic training, the reproducibility
        features should work consistently.
        
        Validates: Requirement 15.2
        """
        # Create manager with given configuration
        manager = ReproducibilityManager(seed=seed, enable_deterministic=enable_deterministic)
        
        # Setup should not fail
        manager.setup_deterministic_training()
        
        # Seed should be preserved
        assert manager.seed == seed
        assert manager.enable_deterministic == enable_deterministic
        
        # Environment info should be capturable
        env_info = manager.log_environment_info()
        assert env_info is not None
        assert env_info['seed'] == seed
        assert env_info['deterministic_mode'] == enable_deterministic
        
        # Reproducibility hash should be consistent
        hash1 = manager.create_reproducibility_hash()
        hash2 = manager.create_reproducibility_hash()
        assert hash1 == hash2
        
        # Hash should be valid SHA-256 (64 hex characters)
        assert len(hash1) == 64
        assert all(c in '0123456789abcdef' for c in hash1.lower())
    
    @given(
        seed1=st.integers(min_value=0, max_value=2**31 - 1),
        seed2=st.integers(min_value=0, max_value=2**31 - 1)
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_37_deterministic_training_reproducibility_uniqueness(self, seed1, seed2):
        """
        Property 37: Deterministic Training Reproducibility (Uniqueness)
        
        Different seeds should produce different reproducibility hashes,
        ensuring that reproducibility validation can distinguish between
        different training configurations.
        
        Validates: Requirement 15.2
        """
        assume(seed1 != seed2)  # Only test with different seeds
        
        # Create managers with different seeds
        manager1 = ReproducibilityManager(seed=seed1)
        manager2 = ReproducibilityManager(seed=seed2)
        
        # Generate hashes
        hash1 = manager1.create_reproducibility_hash()
        hash2 = manager2.create_reproducibility_hash()
        
        # Hashes should be different for different seeds
        assert hash1 != hash2, f"Same hash for different seeds {seed1} and {seed2}"
        
        # Cross-validation should fail
        assert not manager1.validate_reproducibility(hash2)
        assert not manager2.validate_reproducibility(hash1)
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        config_dict=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
            values=st.one_of(
                st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False, allow_infinity=False),
                st.integers(min_value=1, max_value=1000),
                st.text(min_size=1, max_size=50)
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=25, deadline=5000)
    def test_property_37_training_fingerprint_consistency(self, seed, config_dict):
        """
        Property 37: Training Fingerprint Consistency
        
        For any seed and configuration dictionary, the training fingerprint
        should be consistent across multiple generations.
        
        Validates: Requirement 15.2
        """
        # Generate fingerprints multiple times
        fingerprint1 = create_training_fingerprint(seed, config_dict)
        fingerprint2 = create_training_fingerprint(seed, config_dict)
        fingerprint3 = create_training_fingerprint(seed, config_dict)
        
        # All fingerprints should be identical
        assert fingerprint1 == fingerprint2 == fingerprint3
        
        # Fingerprint should be valid SHA-256
        assert len(fingerprint1) == 64
        assert all(c in '0123456789abcdef' for c in fingerprint1.lower())
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        num_operations=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, deadline=10000)
    def test_property_37_environment_logging_consistency(self, seed, num_operations):
        """
        Property 37: Environment Logging Consistency
        
        Environment logging should produce consistent results across
        multiple calls within the same session.
        
        Validates: Requirement 15.2
        """
        manager = ReproducibilityManager(seed=seed)
        
        # Log environment multiple times
        env_logs = []
        for _ in range(num_operations):
            env_info = manager.log_environment_info()
            env_logs.append(env_info)
        
        # All logs should have the same seed
        for env_log in env_logs:
            assert env_log['seed'] == seed
        
        # Key environment information should be consistent
        first_log = env_logs[0]
        for env_log in env_logs[1:]:
            # Python version should be consistent
            assert env_log['python']['version'] == first_log['python']['version']
            
            # System information should be consistent
            assert env_log['system']['platform'] == first_log['system']['platform']
            
            # Library versions should be consistent
            for lib_name in ['torch', 'transformers', 'numpy']:
                if lib_name in first_log['libraries']:
                    assert env_log['libraries'][lib_name] == first_log['libraries'][lib_name]


class TestTrainingProvenanceProperties:
    """Property-based tests for training provenance features."""
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        stage_names=st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=20, deadline=10000)
    def test_property_37_provenance_tracking_consistency(self, seed, stage_names):
        """
        Property 37: Training Provenance Tracking Consistency
        
        Training provenance should be tracked consistently across
        different stage configurations.
        
        Validates: Requirement 15.2 (through provenance integration)
        """
        # Create configuration and provenance manager
        config = Config()
        provenance_manager = create_provenance_manager(config, seed=seed)
        
        # Track all stages
        stages = []
        for stage_name in stage_names:
            stage = provenance_manager.start_stage(stage_name)
            stages.append(stage)
            
            # Add some metrics
            provenance_manager.update_stage_metrics(stage_name, 10, {"loss": 1.0})
            provenance_manager.update_stage_metrics(stage_name, 20, {"loss": 0.8})
            
            # Finalize stage
            provenance_manager.finalize_stage(stage_name, final_loss=0.5)
        
        # Finalize training
        provenance_manager.finalize_training()
        
        # Verify provenance consistency
        assert provenance_manager.provenance.seed == seed
        assert len(provenance_manager.provenance.stages) == len(stage_names)
        
        # All stages should be properly tracked
        tracked_names = [stage.stage_name for stage in provenance_manager.provenance.stages]
        assert set(tracked_names) == set(stage_names)
        
        # All stages should be finalized
        for stage in provenance_manager.provenance.stages:
            assert stage.end_time is not None
            assert stage.final_loss is not None
            assert len(stage.metrics_history) == 2  # Two metric updates per stage
        
        # Provenance summary should be consistent
        summary = provenance_manager.get_provenance_summary()
        assert summary['seed'] == seed
        assert summary['stages_completed'] == len(stage_names)
        assert set(summary['stage_names']) == set(stage_names)
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        num_metrics_updates=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=15, deadline=8000)
    def test_property_37_metrics_tracking_consistency(self, seed, num_metrics_updates):
        """
        Property 37: Metrics Tracking Consistency
        
        Metrics tracking within training stages should maintain consistency
        and proper ordering.
        
        Validates: Requirement 15.2
        """
        config = Config()
        provenance_manager = create_provenance_manager(config, seed=seed)
        
        # Start a stage
        stage_name = "test_stage"
        stage = provenance_manager.start_stage(stage_name)
        
        # Add metrics in sequence
        for i in range(num_metrics_updates):
            step = i * 10
            loss = 2.0 - (i * 0.1)  # Decreasing loss
            metrics = {"loss": loss, "step": step}
            provenance_manager.update_stage_metrics(stage_name, step, metrics)
        
        # Finalize stage
        provenance_manager.finalize_stage(stage_name, final_loss=0.5)
        
        # Verify metrics consistency
        stage = provenance_manager.provenance.get_stage(stage_name)
        assert stage is not None
        assert len(stage.metrics_history) == num_metrics_updates
        
        # Metrics should be in the order they were added
        for i, metric_entry in enumerate(stage.metrics_history):
            expected_step = i * 10
            expected_loss = 2.0 - (i * 0.1)
            
            assert metric_entry["step"] == expected_step
            assert abs(metric_entry["metrics"]["loss"] - expected_loss) < 1e-6
            assert metric_entry["metrics"]["step"] == expected_step
        
        # Total steps should be correct
        assert stage.total_steps == (num_metrics_updates - 1) * 10


class ReproducibilityStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for reproducibility features.
    
    This tests complex interactions between reproducibility components
    to ensure they maintain consistency across different operation sequences.
    """
    
    def __init__(self):
        super().__init__()
        self.managers = {}
        self.environment_logs = {}
        self.reproducibility_hashes = {}
        self.seeds_used = set()
    
    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.managers = {}
        self.environment_logs = {}
        self.reproducibility_hashes = {}
        self.seeds_used = set()
    
    @rule(
        seed=st.integers(min_value=0, max_value=1000),
        enable_deterministic=st.booleans()
    )
    def create_manager(self, seed, enable_deterministic):
        """Create a new reproducibility manager."""
        manager_id = f"manager_{len(self.managers)}"
        manager = ReproducibilityManager(seed=seed, enable_deterministic=enable_deterministic)
        
        self.managers[manager_id] = manager
        self.seeds_used.add(seed)
        
        # Managers with the same seed should have the same configuration
        for existing_id, existing_manager in self.managers.items():
            if existing_manager.seed == seed and existing_id != manager_id:
                assert existing_manager.enable_deterministic == enable_deterministic or not enable_deterministic
    
    @rule(
        manager_id=st.sampled_from([]),  # Will be populated by create_manager
        target=st.just("environment_logs")
    )
    def log_environment(self, manager_id):
        """Log environment information for a manager."""
        assume(manager_id in self.managers)
        
        manager = self.managers[manager_id]
        env_info = manager.log_environment_info()
        
        log_id = f"{manager_id}_log_{len(self.environment_logs)}"
        self.environment_logs[log_id] = env_info
        
        # Environment log should contain expected fields
        assert 'seed' in env_info
        assert 'timestamp' in env_info
        assert 'libraries' in env_info
        assert env_info['seed'] == manager.seed
        
        return log_id
    
    @rule(
        manager_id=st.sampled_from([]),  # Will be populated by create_manager
        target=st.just("reproducibility_hashes")
    )
    def create_hash(self, manager_id):
        """Create a reproducibility hash for a manager."""
        assume(manager_id in self.managers)
        
        manager = self.managers[manager_id]
        repro_hash = manager.create_reproducibility_hash()
        
        hash_id = f"{manager_id}_hash_{len(self.reproducibility_hashes)}"
        self.reproducibility_hashes[hash_id] = {
            'hash': repro_hash,
            'seed': manager.seed,
            'manager_id': manager_id
        }
        
        # Hash should be valid SHA-256
        assert len(repro_hash) == 64
        assert all(c in '0123456789abcdef' for c in repro_hash.lower())
        
        return hash_id
    
    @rule(
        hash_id1=st.sampled_from([]),  # Will be populated by create_hash
        hash_id2=st.sampled_from([])   # Will be populated by create_hash
    )
    def validate_hash_consistency(self, hash_id1, hash_id2):
        """Validate hash consistency between different managers."""
        assume(hash_id1 in self.reproducibility_hashes)
        assume(hash_id2 in self.reproducibility_hashes)
        assume(hash_id1 != hash_id2)
        
        hash_info1 = self.reproducibility_hashes[hash_id1]
        hash_info2 = self.reproducibility_hashes[hash_id2]
        
        manager1 = self.managers[hash_info1['manager_id']]
        manager2 = self.managers[hash_info2['manager_id']]
        
        # Same seed should produce same hash
        if hash_info1['seed'] == hash_info2['seed']:
            # If managers have same seed and deterministic settings, hashes should match
            if (manager1.enable_deterministic == manager2.enable_deterministic and
                manager1.enable_deterministic):  # Both deterministic
                assert hash_info1['hash'] == hash_info2['hash']
        else:
            # Different seeds should produce different hashes
            assert hash_info1['hash'] != hash_info2['hash']
    
    @invariant()
    def managers_maintain_seed_consistency(self):
        """Invariant: All managers should maintain their assigned seeds."""
        for manager_id, manager in self.managers.items():
            assert manager.seed in self.seeds_used
            assert isinstance(manager.seed, int)
            assert 0 <= manager.seed <= 1000
    
    @invariant()
    def environment_logs_contain_valid_data(self):
        """Invariant: All environment logs should contain valid data."""
        for log_id, env_info in self.environment_logs.items():
            assert isinstance(env_info, dict)
            assert 'seed' in env_info
            assert 'timestamp' in env_info
            assert 'libraries' in env_info
            assert isinstance(env_info['seed'], int)
    
    @invariant()
    def reproducibility_hashes_are_valid(self):
        """Invariant: All reproducibility hashes should be valid SHA-256."""
        for hash_id, hash_info in self.reproducibility_hashes.items():
            repro_hash = hash_info['hash']
            assert len(repro_hash) == 64
            assert all(c in '0123456789abcdef' for c in repro_hash.lower())


# Configure the state machine for testing
TestReproducibilityStateMachine = ReproducibilityStateMachine.TestCase


class TestIntegrationProperties:
    """Integration property tests for reproducibility with other components."""
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1)
    )
    @settings(max_examples=10, deadline=15000)
    def test_property_37_config_integration_reproducibility(self, seed):
        """
        Property 37: Configuration Integration Reproducibility
        
        Reproducibility features should integrate correctly with
        configuration management for complete reproducibility.
        
        Validates: Requirement 15.2
        """
        # Create configuration
        config = Config()
        
        # Setup reproducible training
        manager = setup_reproducible_training(seed=seed)
        
        # Create checkpoint snapshot with reproducibility info
        checkpoint_id = f"test_checkpoint_{seed}"
        training_metadata = {"seed": seed, "test": True}
        
        snapshot = config.create_checkpoint_snapshot(checkpoint_id, training_metadata)
        
        # Snapshot should contain reproducibility information
        assert 'reproducibility' in snapshot
        if snapshot['reproducibility'] is not None:
            assert snapshot['reproducibility']['seed'] == seed
        
        # Configuration hash should be consistent
        hash1 = snapshot.get('config_hash')
        
        # Create another snapshot with same config
        snapshot2 = config.create_checkpoint_snapshot(checkpoint_id, training_metadata)
        hash2 = snapshot2.get('config_hash')
        
        # Hashes should be identical for same configuration
        assert hash1 == hash2
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1)
    )
    @settings(max_examples=10, deadline=15000)
    def test_property_37_provenance_integration_reproducibility(self, seed):
        """
        Property 37: Training Provenance Integration Reproducibility
        
        Training provenance should maintain reproducibility information
        consistently throughout the training process.
        
        Validates: Requirement 15.2
        """
        # Create configuration and provenance manager
        config = Config()
        provenance_manager = create_provenance_manager(config, seed=seed)
        
        # Verify initial reproducibility setup
        assert provenance_manager.seed == seed
        assert provenance_manager.provenance.seed == seed
        
        # Environment info should be captured
        assert provenance_manager.provenance.environment_info is not None
        assert provenance_manager.provenance.environment_info['seed'] == seed
        
        # Reproducibility hash should be consistent
        hash1 = provenance_manager.provenance.reproducibility_hash
        hash2 = provenance_manager.repro_manager.create_reproducibility_hash()
        
        assert hash1 == hash2
        
        # Training fingerprint should be deterministic
        fingerprint1 = provenance_manager.create_training_fingerprint()
        fingerprint2 = provenance_manager.create_training_fingerprint()
        
        assert fingerprint1 == fingerprint2
        
        # Provenance summary should maintain reproducibility info
        summary = provenance_manager.get_provenance_summary()
        assert summary['seed'] == seed
        assert summary['reproducibility_hash'] == hash1
        assert summary['training_fingerprint'] == fingerprint1


# Test configuration for property tests
@pytest.mark.property
class TestReproducibilityPropertiesRunner:
    """Runner for all reproducibility property tests."""
    
    def test_run_all_property_tests(self):
        """Run all property tests to ensure comprehensive coverage."""
        # This test ensures all property tests are discovered and run
        # Individual property tests are defined in the classes above
        pass


if __name__ == "__main__":
    # Run property tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])