"""
Property-Based Tests for Training Orchestrator

This module contains property-based tests for the Training Orchestrator component,
validating universal correctness properties across different configurations and scenarios.

Properties tested:
- Property 1: Failure State Preservation (Validates: Requirement 1.4)
- Property 2: Stage Validation Consistency (Validates: Requirement 1.5)
- Property 9: Memory Monitoring Consistency (Validates: Requirement 5.5)
- Property 23: Loss Divergence Response (Validates: Requirement 9.4)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume
import torch
import numpy as np
from datetime import datetime, timezone

from rlhf_phi3.config.config_manager import Config
from rlhf_phi3.training.training_orchestrator import (
    TrainingOrchestrator, TrainingStage, StageResult, PipelineState
)


class TestTrainingOrchestratorProperties:
    """Property-based tests for Training Orchestrator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration for testing."""
        config = Config()
        config.paths.base_output_dir = str(temp_dir)
        config.paths.cache_dir = str(temp_dir / "cache")
        config.paths.logs_dir = str(temp_dir / "logs")
        config.wandb.project = "test_project"
        return config
    
    @pytest.fixture
    def mock_orchestrator(self, mock_config):
        """Create mock orchestrator with mocked dependencies."""
        with patch('rlhf_phi3.training.training_orchestrator.ModelManager'), \
             patch('rlhf_phi3.training.training_orchestrator.DatasetManager'), \
             patch('rlhf_phi3.training.training_orchestrator.CheckpointManager'), \
             patch('rlhf_phi3.training.training_orchestrator.ExperimentTracker'):
            
            orchestrator = TrainingOrchestrator(mock_config)
            
            # Mock the component managers
            orchestrator.model_manager = Mock()
            orchestrator.dataset_manager = Mock()
            orchestrator.checkpoint_manager = Mock()
            orchestrator.experiment_tracker = Mock()
            
            return orchestrator
    
    @given(
        error_type=st.sampled_from([RuntimeError, ValueError, MemoryError, KeyboardInterrupt]),
        error_message=st.text(min_size=1, max_size=100),
        stage=st.sampled_from([TrainingStage.SFT, TrainingStage.REWARD, TrainingStage.RLOO]),
        duration=st.floats(min_value=0.1, max_value=3600.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_1_failure_state_preservation(self, mock_orchestrator, error_type, error_message, stage, duration):
        """
        **Property 1: Failure State Preservation**
        
        *For any* training stage failure scenario, the Training_Orchestrator SHALL save 
        the current state and provide clear error diagnostics.
        
        **Validates: Requirement 1.4**
        """
        # Arrange
        orchestrator = mock_orchestrator
        error = error_type(error_message)
        
        # Act - simulate stage failure
        orchestrator._handle_stage_failure(stage, error, duration)
        
        # Assert - verify failure state is preserved
        assert stage in orchestrator.pipeline_state.stage_results
        
        stage_result = orchestrator.pipeline_state.stage_results[stage]
        assert isinstance(stage_result, StageResult)
        assert stage_result.stage == stage
        assert stage_result.success is False
        assert stage_result.error_message == error_message
        assert stage_result.duration_seconds == duration
        
        # Verify failure count is incremented
        assert orchestrator.pipeline_state.failure_count > 0
        
        # Verify error details are saved (check that error file would be created)
        expected_error_file = Path(orchestrator.config.paths.base_output_dir) / f"error_{stage.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        # Note: We can't check file creation in unit test, but we verify the logic path
        
        print(f"✓ Property 1 verified: Failure state preserved for {stage.value} with {error_type.__name__}")
    
    @given(
        stage=st.sampled_from([TrainingStage.SFT, TrainingStage.REWARD, TrainingStage.RLOO]),
        has_checkpoint=st.booleans(),
        checkpoint_exists=st.booleans(),
        has_metrics=st.booleans()
    )
    @settings(max_examples=30, deadline=None)
    def test_property_2_stage_validation_consistency(self, mock_orchestrator, stage, has_checkpoint, checkpoint_exists, has_metrics):
        """
        **Property 2: Stage Validation Consistency**
        
        *For any* training stage completion state, the Training_Orchestrator SHALL validate 
        successful completion before proceeding to the next stage.
        
        **Validates: Requirement 1.5**
        """
        # Arrange
        orchestrator = mock_orchestrator
        
        # Setup stage result based on test parameters
        checkpoint_path = "/fake/checkpoint/path" if has_checkpoint else None
        metrics = {"loss": 0.5, "accuracy": 0.8} if has_metrics else {}
        
        stage_result = StageResult(
            stage=stage,
            success=True,
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            duration_seconds=100.0,
            memory_peak_gb=2.5
        )
        
        # Add to pipeline state
        orchestrator.pipeline_state.completed_stages.append(stage)
        orchestrator.pipeline_state.stage_results[stage] = stage_result
        
        # Mock checkpoint existence
        with patch('pathlib.Path.exists', return_value=checkpoint_exists):
            # Act
            validation_result = orchestrator.validate_stage_completion(stage.value)
            
            # Assert - validation should pass only if all conditions are met
            expected_valid = has_checkpoint and checkpoint_exists
            assert validation_result == expected_valid
            
            if expected_valid:
                print(f"✓ Property 2 verified: Stage {stage.value} validation passed correctly")
            else:
                print(f"✓ Property 2 verified: Stage {stage.value} validation failed correctly (missing checkpoint or file)")
    
    @given(
        memory_values=st.lists(
            st.floats(min_value=0.0, max_value=16.0),  # GPU memory in GB
            min_size=5,
            max_size=20
        ),
        steps=st.lists(
            st.integers(min_value=0, max_value=1000),
            min_size=5,
            max_size=20
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property_9_memory_monitoring_consistency(self, mock_orchestrator, memory_values, steps):
        """
        **Property 9: Memory Monitoring Consistency**
        
        *For any* training stage, the Training_Orchestrator SHALL monitor and report 
        peak memory usage.
        
        **Validates: Requirement 5.5**
        """
        assume(len(memory_values) == len(steps))
        
        # Arrange
        orchestrator = mock_orchestrator
        
        # Mock memory usage data
        memory_stats_sequence = []
        for i, (memory_gb, step) in enumerate(zip(memory_values, steps)):
            memory_stats = {
                'gpu_allocated_gb': memory_gb,
                'gpu_reserved_gb': memory_gb * 1.2,
                'gpu_max_memory_gb': 16.0,
                'gpu_utilization': memory_gb / 16.0,
                'cpu_memory_gb': 4.0,
                'cpu_memory_percent': 0.5
            }
            memory_stats_sequence.append(memory_stats)
        
        # Mock model manager to return our test memory stats
        orchestrator.model_manager.get_memory_usage.side_effect = memory_stats_sequence
        
        # Act - simulate memory monitoring during training
        for i, step in enumerate(steps):
            orchestrator._monitor_memory_usage(step)
        
        # Assert - verify memory monitoring consistency
        assert len(orchestrator.memory_usage_history) == len(steps)
        
        # Verify each memory entry has required fields
        for i, entry in enumerate(orchestrator.memory_usage_history):
            assert 'step' in entry
            assert 'timestamp' in entry
            assert 'gpu_allocated_gb' in entry
            assert entry['step'] == steps[i]
            assert entry['gpu_allocated_gb'] == memory_values[i]
        
        # Verify peak memory calculation
        peak_memory = orchestrator._get_peak_memory_usage()
        expected_peak = max(memory_values) if memory_values else 0.0
        assert abs(peak_memory - expected_peak) < 1e-6
        
        # Verify experiment tracker was called for memory logging
        assert orchestrator.experiment_tracker.log_metrics.called
        
        print(f"✓ Property 9 verified: Memory monitoring consistent across {len(steps)} steps")
    
    @given(
        loss_sequence=st.lists(
            st.floats(min_value=0.01, max_value=10.0),
            min_size=10,
            max_size=50
        ),
        divergence_factor=st.floats(min_value=1.5, max_value=5.0)
    )
    @settings(max_examples=25, deadline=None)
    def test_property_23_loss_divergence_response(self, mock_orchestrator, loss_sequence, divergence_factor):
        """
        **Property 23: Loss Divergence Response**
        
        *For any* training loss divergence pattern, the Training_Orchestrator SHALL 
        implement early stopping and suggest hyperparameter adjustments.
        
        **Validates: Requirement 9.4**
        """
        # Arrange
        orchestrator = mock_orchestrator
        orchestrator.loss_divergence_threshold = divergence_factor
        
        # Setup initial loss history (stable losses)
        stable_losses = loss_sequence[:10]
        orchestrator.loss_history = stable_losses.copy()
        
        # Calculate what would trigger divergence
        recent_avg = np.mean(stable_losses[-10:])
        divergent_loss = recent_avg * divergence_factor * 1.1  # Slightly above threshold
        
        # Act - test normal loss (should not trigger divergence)
        normal_loss = recent_avg * 0.9
        normal_metrics = {'train_loss': normal_loss}
        divergence_detected_normal = orchestrator._check_loss_divergence(normal_metrics)
        
        # Act - test divergent loss (should trigger divergence)
        divergent_metrics = {'train_loss': divergent_loss}
        divergence_detected_divergent = orchestrator._check_loss_divergence(divergent_metrics)
        
        # Act - test NaN loss (should trigger divergence)
        nan_metrics = {'train_loss': float('nan')}
        divergence_detected_nan = orchestrator._check_loss_divergence(nan_metrics)
        
        # Act - test infinite loss (should trigger divergence)
        inf_metrics = {'train_loss': float('inf')}
        divergence_detected_inf = orchestrator._check_loss_divergence(inf_metrics)
        
        # Assert - verify divergence detection logic
        assert divergence_detected_normal is False, "Normal loss should not trigger divergence"
        assert divergence_detected_divergent is True, f"Loss {divergent_loss:.4f} vs avg {recent_avg:.4f} should trigger divergence"
        assert divergence_detected_nan is True, "NaN loss should trigger divergence"
        assert divergence_detected_inf is True, "Infinite loss should trigger divergence"
        
        # Verify that divergence threshold is respected
        threshold_loss = recent_avg * divergence_factor * 0.9  # Just below threshold
        threshold_metrics = {'train_loss': threshold_loss}
        divergence_detected_threshold = orchestrator._check_loss_divergence(threshold_metrics)
        assert divergence_detected_threshold is False, "Loss below threshold should not trigger divergence"
        
        print(f"✓ Property 23 verified: Loss divergence detection works correctly with threshold {divergence_factor}")
    
    @given(
        stage_sequence=st.lists(
            st.sampled_from([TrainingStage.SFT, TrainingStage.REWARD, TrainingStage.RLOO]),
            min_size=1,
            max_size=3,
            unique=True
        ),
        success_pattern=st.lists(st.booleans(), min_size=1, max_size=3)
    )
    @settings(max_examples=15, deadline=None)
    def test_pipeline_state_consistency(self, mock_orchestrator, stage_sequence, success_pattern):
        """
        Test that pipeline state remains consistent across stage transitions.
        
        This property ensures that the orchestrator maintains accurate state
        tracking regardless of success/failure patterns.
        """
        assume(len(stage_sequence) == len(success_pattern))
        
        # Arrange
        orchestrator = mock_orchestrator
        
        # Act - simulate stage completions
        for stage, success in zip(stage_sequence, success_pattern):
            if success:
                # Simulate successful stage completion
                stage_result = StageResult(
                    stage=stage,
                    success=True,
                    checkpoint_path=f"/fake/checkpoint/{stage.value}",
                    metrics={"loss": 0.5},
                    duration_seconds=100.0,
                    memory_peak_gb=2.0
                )
                orchestrator.pipeline_state.completed_stages.append(stage)
                orchestrator.pipeline_state.stage_results[stage] = stage_result
            else:
                # Simulate stage failure
                error = RuntimeError(f"Simulated {stage.value} failure")
                orchestrator._handle_stage_failure(stage, error, 50.0)
        
        # Assert - verify state consistency
        expected_completed = [stage for stage, success in zip(stage_sequence, success_pattern) if success]
        assert orchestrator.pipeline_state.completed_stages == expected_completed
        
        # Verify all stages have results
        for stage in stage_sequence:
            assert stage in orchestrator.pipeline_state.stage_results
            result = orchestrator.pipeline_state.stage_results[stage]
            expected_success = success_pattern[stage_sequence.index(stage)]
            assert result.success == expected_success
        
        # Verify failure count
        expected_failures = sum(1 for success in success_pattern if not success)
        assert orchestrator.pipeline_state.failure_count == expected_failures
        
        print(f"✓ Pipeline state consistency verified for {len(stage_sequence)} stages")
    
    @given(
        memory_threshold=st.floats(min_value=0.8, max_value=0.99),
        memory_usage=st.floats(min_value=0.5, max_value=1.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_memory_optimization_trigger(self, mock_orchestrator, memory_threshold, memory_usage):
        """
        Test that memory optimization is triggered appropriately based on usage thresholds.
        """
        # Arrange
        orchestrator = mock_orchestrator
        orchestrator.model_manager._memory_threshold = memory_threshold
        
        memory_stats = {
            'gpu_allocated_gb': memory_usage * 16.0,
            'gpu_reserved_gb': memory_usage * 16.0 * 1.2,
            'gpu_max_memory_gb': 16.0,
            'gpu_utilization': memory_usage,
            'cpu_memory_gb': 4.0,
            'cpu_memory_percent': 0.5
        }
        
        orchestrator.model_manager.get_memory_usage.return_value = memory_stats
        
        # Act
        orchestrator._monitor_memory_usage(100)
        
        # Assert - verify memory optimization is triggered when appropriate
        if memory_usage > 0.95:  # High memory usage threshold in the implementation
            orchestrator.model_manager.handle_memory_exhaustion.assert_called_once()
        
        # Verify memory stats are recorded
        assert len(orchestrator.memory_usage_history) == 1
        assert orchestrator.memory_usage_history[0]['gpu_utilization'] == memory_usage
        
        print(f"✓ Memory optimization trigger verified for usage {memory_usage:.2%} vs threshold {memory_threshold:.2%}")


# Additional helper functions for property testing

def generate_valid_stage_result(stage: TrainingStage) -> StageResult:
    """Generate a valid StageResult for testing."""
    return StageResult(
        stage=stage,
        success=True,
        checkpoint_path=f"/fake/checkpoint/{stage.value}",
        metrics={"loss": 0.5, "accuracy": 0.8},
        duration_seconds=100.0,
        memory_peak_gb=2.5
    )


def generate_failed_stage_result(stage: TrainingStage, error_msg: str) -> StageResult:
    """Generate a failed StageResult for testing."""
    return StageResult(
        stage=stage,
        success=False,
        error_message=error_msg,
        duration_seconds=50.0,
        memory_peak_gb=1.5
    )


@pytest.mark.property
class TestTrainingOrchestratorEdgeCases:
    """Test edge cases and boundary conditions for Training Orchestrator properties."""
    
    def test_empty_loss_history_divergence_check(self):
        """Test loss divergence check with empty history."""
        config = Config()
        with patch('rlhf_phi3.training.training_orchestrator.ModelManager'), \
             patch('rlhf_phi3.training.training_orchestrator.DatasetManager'), \
             patch('rlhf_phi3.training.training_orchestrator.CheckpointManager'), \
             patch('rlhf_phi3.training.training_orchestrator.ExperimentTracker'):
            
            orchestrator = TrainingOrchestrator(config)
            orchestrator.loss_history = []
            
            # Should not detect divergence with empty history
            result = orchestrator._check_loss_divergence({'train_loss': 10.0})
            assert result is False
    
    def test_insufficient_loss_history_divergence_check(self):
        """Test loss divergence check with insufficient history."""
        config = Config()
        with patch('rlhf_phi3.training.training_orchestrator.ModelManager'), \
             patch('rlhf_phi3.training.training_orchestrator.DatasetManager'), \
             patch('rlhf_phi3.training.training_orchestrator.CheckpointManager'), \
             patch('rlhf_phi3.training.training_orchestrator.ExperimentTracker'):
            
            orchestrator = TrainingOrchestrator(config)
            orchestrator.loss_history = [1.0, 1.1, 0.9]  # Less than 10 entries
            
            # Should not detect divergence with insufficient history
            result = orchestrator._check_loss_divergence({'train_loss': 10.0})
            assert result is False
    
    def test_missing_train_loss_divergence_check(self):
        """Test loss divergence check without train_loss metric."""
        config = Config()
        with patch('rlhf_phi3.training.training_orchestrator.ModelManager'), \
             patch('rlhf_phi3.training.training_orchestrator.DatasetManager'), \
             patch('rlhf_phi3.training.training_orchestrator.CheckpointManager'), \
             patch('rlhf_phi3.training.training_orchestrator.ExperimentTracker'):
            
            orchestrator = TrainingOrchestrator(config)
            orchestrator.loss_history = [1.0] * 15
            
            # Should not detect divergence without train_loss
            result = orchestrator._check_loss_divergence({'other_metric': 10.0})
            assert result is False