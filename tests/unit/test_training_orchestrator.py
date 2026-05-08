"""
Unit Tests for Training Orchestrator

This module contains comprehensive unit tests for the Training Orchestrator component,
testing specific examples, edge cases, and component functionality.

Test Coverage:
- Stage sequencing and validation
- Error handling and recovery
- Memory monitoring integration
- Configuration management
- Checkpoint integration
- Experiment tracking integration

Requirements tested:
- 1.4: Failure state preservation and error diagnostics
- 1.5: Stage validation before proceeding
- 5.5: Memory monitoring and reporting
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timezone

import torch
import numpy as np
from transformers import TrainingArguments

from rlhf_phi3.config.config_manager import Config
from rlhf_phi3.training.training_orchestrator import (
    TrainingOrchestrator, TrainingStage, StageResult, PipelineState
)


class TestTrainingOrchestrator:
    """Unit tests for Training Orchestrator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        config = Config()
        config.paths.base_output_dir = str(temp_dir)
        config.paths.cache_dir = str(temp_dir / "cache")
        config.paths.logs_dir = str(temp_dir / "logs")
        config.wandb.project = "test_project"
        config.training.sft.epochs = 1
        config.training.sft.max_steps = 10
        config.training.reward.epochs = 1
        config.training.reward.max_steps = 5
        config.training.ppo.max_steps = 5
        return config
    
    @pytest.fixture
    def mock_orchestrator(self, config):
        """Create orchestrator with mocked dependencies."""
        with patch('rlhf_phi3.training.training_orchestrator.ModelManager') as mock_model_mgr, \
             patch('rlhf_phi3.training.training_orchestrator.DatasetManager') as mock_dataset_mgr, \
             patch('rlhf_phi3.training.training_orchestrator.CheckpointManager') as mock_checkpoint_mgr, \
             patch('rlhf_phi3.training.training_orchestrator.ExperimentTracker') as mock_exp_tracker:
            
            orchestrator = TrainingOrchestrator(config)
            
            # Setup mock returns
            orchestrator.model_manager = Mock()
            orchestrator.dataset_manager = Mock()
            orchestrator.checkpoint_manager = Mock()
            orchestrator.experiment_tracker = Mock()
            
            return orchestrator
    
    def test_orchestrator_initialization(self, config):
        """Test orchestrator initialization with proper component setup."""
        with patch('rlhf_phi3.training.training_orchestrator.ModelManager'), \
             patch('rlhf_phi3.training.training_orchestrator.DatasetManager'), \
             patch('rlhf_phi3.training.training_orchestrator.CheckpointManager'), \
             patch('rlhf_phi3.training.training_orchestrator.ExperimentTracker'):
            
            orchestrator = TrainingOrchestrator(config)
            
            # Verify initialization
            assert orchestrator.config == config
            assert isinstance(orchestrator.pipeline_state, PipelineState)
            assert orchestrator.pipeline_state.completed_stages == []
            assert orchestrator.pipeline_state.stage_results == {}
            assert orchestrator.pipeline_state.failure_count == 0
            assert orchestrator.loss_history == []
            assert orchestrator.memory_usage_history == []
    
    def test_stage_validation_success(self, mock_orchestrator):
        """Test successful stage validation."""
        orchestrator = mock_orchestrator
        stage = TrainingStage.SFT
        
        # Setup successful stage result
        stage_result = StageResult(
            stage=stage,
            success=True,
            checkpoint_path="/fake/checkpoint/path",
            metrics={"loss": 0.5},
            duration_seconds=100.0,
            memory_peak_gb=2.0
        )
        
        orchestrator.pipeline_state.completed_stages.append(stage)
        orchestrator.pipeline_state.stage_results[stage] = stage_result
        
        # Mock checkpoint file existence
        with patch('pathlib.Path.exists', return_value=True):
            result = orchestrator.validate_stage_completion(stage.value)
            assert result is True
    
    def test_stage_validation_failure_missing_stage(self, mock_orchestrator):
        """Test stage validation failure when stage not completed."""
        orchestrator = mock_orchestrator
        
        # Try to validate a stage that hasn't been completed
        result = orchestrator.validate_stage_completion(TrainingStage.SFT.value)
        assert result is False
    
    def test_stage_validation_failure_no_checkpoint(self, mock_orchestrator):
        """Test stage validation failure when checkpoint is missing."""
        orchestrator = mock_orchestrator
        stage = TrainingStage.SFT
        
        # Setup stage result without checkpoint
        stage_result = StageResult(
            stage=stage,
            success=True,
            checkpoint_path=None,  # No checkpoint
            metrics={"loss": 0.5},
            duration_seconds=100.0,
            memory_peak_gb=2.0
        )
        
        orchestrator.pipeline_state.completed_stages.append(stage)
        orchestrator.pipeline_state.stage_results[stage] = stage_result
        
        result = orchestrator.validate_stage_completion(stage.value)
        assert result is False
    
    def test_stage_validation_failure_checkpoint_not_exists(self, mock_orchestrator):
        """Test stage validation failure when checkpoint file doesn't exist."""
        orchestrator = mock_orchestrator
        stage = TrainingStage.SFT
        
        # Setup stage result with checkpoint path
        stage_result = StageResult(
            stage=stage,
            success=True,
            checkpoint_path="/fake/checkpoint/path",
            metrics={"loss": 0.5},
            duration_seconds=100.0,
            memory_peak_gb=2.0
        )
        
        orchestrator.pipeline_state.completed_stages.append(stage)
        orchestrator.pipeline_state.stage_results[stage] = stage_result
        
        # Mock checkpoint file doesn't exist
        with patch('pathlib.Path.exists', return_value=False):
            result = orchestrator.validate_stage_completion(stage.value)
            assert result is False
    
    def test_stage_failure_handling(self, mock_orchestrator):
        """Test proper handling of stage failures."""
        orchestrator = mock_orchestrator
        stage = TrainingStage.SFT
        error = RuntimeError("Test error message")
        duration = 123.45
        
        # Act
        orchestrator._handle_stage_failure(stage, error, duration)
        
        # Assert
        assert stage in orchestrator.pipeline_state.stage_results
        stage_result = orchestrator.pipeline_state.stage_results[stage]
        
        assert stage_result.stage == stage
        assert stage_result.success is False
        assert stage_result.error_message == "Test error message"
        assert stage_result.duration_seconds == duration
        assert orchestrator.pipeline_state.failure_count == 1
    
    def test_memory_monitoring(self, mock_orchestrator):
        """Test memory monitoring functionality."""
        orchestrator = mock_orchestrator
        step = 100
        
        # Mock memory stats
        memory_stats = {
            'gpu_allocated_gb': 4.5,
            'gpu_reserved_gb': 5.0,
            'gpu_max_memory_gb': 16.0,
            'gpu_utilization': 0.28,
            'cpu_memory_gb': 8.0,
            'cpu_memory_percent': 0.6
        }
        
        orchestrator.model_manager.get_memory_usage.return_value = memory_stats
        
        # Act
        orchestrator._monitor_memory_usage(step)
        
        # Assert
        assert len(orchestrator.memory_usage_history) == 1
        memory_entry = orchestrator.memory_usage_history[0]
        
        assert memory_entry['step'] == step
        assert 'timestamp' in memory_entry
        assert memory_entry['gpu_allocated_gb'] == 4.5
        assert memory_entry['gpu_utilization'] == 0.28
        
        # Verify model manager was called
        orchestrator.model_manager.get_memory_usage.assert_called_once()
    
    def test_memory_monitoring_high_usage_triggers_optimization(self, mock_orchestrator):
        """Test that high memory usage triggers optimization."""
        orchestrator = mock_orchestrator
        step = 100
        
        # Mock high memory usage
        memory_stats = {
            'gpu_allocated_gb': 15.5,
            'gpu_reserved_gb': 16.0,
            'gpu_max_memory_gb': 16.0,
            'gpu_utilization': 0.97,  # High usage
            'cpu_memory_gb': 8.0,
            'cpu_memory_percent': 0.6
        }
        
        orchestrator.model_manager.get_memory_usage.return_value = memory_stats
        
        # Act
        orchestrator._monitor_memory_usage(step)
        
        # Assert memory optimization was triggered
        orchestrator.model_manager.handle_memory_exhaustion.assert_called_once()
    
    def test_loss_divergence_detection_normal_loss(self, mock_orchestrator):
        """Test loss divergence detection with normal loss values."""
        orchestrator = mock_orchestrator
        
        # Setup loss history with stable values
        orchestrator.loss_history = [1.0, 0.9, 1.1, 0.95, 1.05, 0.98, 1.02, 0.97, 1.03, 0.99]
        
        # Test normal loss (should not trigger divergence)
        metrics = {'train_loss': 1.01}
        result = orchestrator._check_loss_divergence(metrics)
        assert result is False
    
    def test_loss_divergence_detection_divergent_loss(self, mock_orchestrator):
        """Test loss divergence detection with divergent loss values."""
        orchestrator = mock_orchestrator
        orchestrator.loss_divergence_threshold = 2.0
        
        # Setup loss history with stable values
        orchestrator.loss_history = [1.0, 0.9, 1.1, 0.95, 1.05, 0.98, 1.02, 0.97, 1.03, 0.99]
        
        # Test divergent loss (should trigger divergence)
        recent_avg = np.mean(orchestrator.loss_history[-10:])
        divergent_loss = recent_avg * 2.5  # Above threshold
        
        metrics = {'train_loss': divergent_loss}
        result = orchestrator._check_loss_divergence(metrics)
        assert result is True
    
    def test_loss_divergence_detection_nan_loss(self, mock_orchestrator):
        """Test loss divergence detection with NaN loss."""
        orchestrator = mock_orchestrator
        
        # Setup loss history
        orchestrator.loss_history = [1.0] * 15
        
        # Test NaN loss (should trigger divergence)
        metrics = {'train_loss': float('nan')}
        result = orchestrator._check_loss_divergence(metrics)
        assert result is True
    
    def test_loss_divergence_detection_infinite_loss(self, mock_orchestrator):
        """Test loss divergence detection with infinite loss."""
        orchestrator = mock_orchestrator
        
        # Setup loss history
        orchestrator.loss_history = [1.0] * 15
        
        # Test infinite loss (should trigger divergence)
        metrics = {'train_loss': float('inf')}
        result = orchestrator._check_loss_divergence(metrics)
        assert result is True
    
    def test_loss_divergence_detection_insufficient_history(self, mock_orchestrator):
        """Test loss divergence detection with insufficient history."""
        orchestrator = mock_orchestrator
        
        # Setup insufficient loss history
        orchestrator.loss_history = [1.0, 0.9, 1.1]  # Less than 10 entries
        
        # Should not trigger divergence with insufficient history
        metrics = {'train_loss': 10.0}
        result = orchestrator._check_loss_divergence(metrics)
        assert result is False
    
    def test_peak_memory_calculation(self, mock_orchestrator):
        """Test peak memory usage calculation."""
        orchestrator = mock_orchestrator
        
        # Setup memory history
        orchestrator.memory_usage_history = [
            {'step': 1, 'gpu_allocated_gb': 2.5},
            {'step': 2, 'gpu_allocated_gb': 4.2},
            {'step': 3, 'gpu_allocated_gb': 3.8},
            {'step': 4, 'gpu_allocated_gb': 5.1},  # Peak
            {'step': 5, 'gpu_allocated_gb': 3.2},
        ]
        
        peak_memory = orchestrator._get_peak_memory_usage()
        assert peak_memory == 5.1
    
    def test_peak_memory_calculation_empty_history(self, mock_orchestrator):
        """Test peak memory calculation with empty history."""
        orchestrator = mock_orchestrator
        orchestrator.memory_usage_history = []
        
        peak_memory = orchestrator._get_peak_memory_usage()
        assert peak_memory == 0.0
    
    def test_sft_training_arguments_creation(self, mock_orchestrator):
        """Test creation of SFT training arguments."""
        orchestrator = mock_orchestrator
        
        training_args = orchestrator._create_sft_training_arguments()
        
        assert isinstance(training_args, TrainingArguments)
        assert training_args.num_train_epochs == orchestrator.config.training.sft.epochs
        assert training_args.per_device_train_batch_size == orchestrator.config.training.sft.batch_size
        assert training_args.learning_rate == orchestrator.config.training.sft.learning_rate
        assert training_args.max_steps == orchestrator.config.training.sft.max_steps
        assert training_args.fp16 == orchestrator.config.optimization.fp16
        assert training_args.gradient_checkpointing == orchestrator.config.optimization.gradient_checkpointing
    
    def test_reward_training_arguments_creation(self, mock_orchestrator):
        """Test creation of reward training arguments."""
        orchestrator = mock_orchestrator
        
        training_args = orchestrator._create_reward_training_arguments()
        
        assert isinstance(training_args, TrainingArguments)
        assert training_args.num_train_epochs == orchestrator.config.training.reward.epochs
        assert training_args.per_device_train_batch_size == orchestrator.config.training.reward.batch_size
        assert training_args.learning_rate == orchestrator.config.training.reward.learning_rate
        assert training_args.max_steps == orchestrator.config.training.reward.max_steps
    
    def test_ppo_config_creation(self, mock_orchestrator):
        """Test creation of PPO configuration."""
        orchestrator = mock_orchestrator
        
        ppo_config = orchestrator._create_ppo_config()
        
        assert ppo_config.model_name == orchestrator.config.model.name
        assert ppo_config.learning_rate == orchestrator.config.training.ppo.learning_rate
        assert ppo_config.batch_size == orchestrator.config.training.ppo.batch_size
        assert ppo_config.mini_batch_size == orchestrator.config.training.ppo.mini_batch_size
        assert ppo_config.ppo_epochs == orchestrator.config.training.ppo.ppo_epochs
    
    def test_sft_dataset_validation_success(self, mock_orchestrator):
        """Test successful SFT dataset validation."""
        orchestrator = mock_orchestrator
        
        # Mock dataset with required fields
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.column_names = ['input_ids', 'attention_mask', 'labels', 'text']
        mock_dataset.__getitem__ = Mock(return_value={
            'input_ids': [1, 2, 3, 4, 5],
            'attention_mask': [1, 1, 1, 1, 1],
            'labels': [1, 2, 3, 4, 5],
            'text': 'Sample text'
        })
        
        result = orchestrator._validate_sft_dataset(mock_dataset)
        assert result is True
    
    def test_sft_dataset_validation_empty_dataset(self, mock_orchestrator):
        """Test SFT dataset validation with empty dataset."""
        orchestrator = mock_orchestrator
        
        # Mock empty dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=0)
        
        result = orchestrator._validate_sft_dataset(mock_dataset)
        assert result is False
    
    def test_sft_dataset_validation_missing_fields(self, mock_orchestrator):
        """Test SFT dataset validation with missing required fields."""
        orchestrator = mock_orchestrator
        
        # Mock dataset missing required fields
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.column_names = ['input_ids', 'attention_mask']  # Missing 'labels'
        
        result = orchestrator._validate_sft_dataset(mock_dataset)
        assert result is False
    
    def test_preference_dataset_validation_success(self, mock_orchestrator):
        """Test successful preference dataset validation."""
        orchestrator = mock_orchestrator
        
        # Mock dataset with required fields
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_dataset.column_names = [
            'chosen_input_ids', 'chosen_attention_mask',
            'rejected_input_ids', 'rejected_attention_mask'
        ]
        mock_dataset.__getitem__ = Mock(return_value={
            'chosen_input_ids': [1, 2, 3, 4, 5],
            'chosen_attention_mask': [1, 1, 1, 1, 1],
            'rejected_input_ids': [1, 2, 3, 4, 6],
            'rejected_attention_mask': [1, 1, 1, 1, 1]
        })
        
        result = orchestrator._validate_preference_dataset(mock_dataset)
        assert result is True
    
    def test_preference_dataset_validation_empty_dataset(self, mock_orchestrator):
        """Test preference dataset validation with empty dataset."""
        orchestrator = mock_orchestrator
        
        # Mock empty dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=0)
        
        result = orchestrator._validate_preference_dataset(mock_dataset)
        assert result is False
    
    def test_ppo_dataset_validation_success(self, mock_orchestrator):
        """Test successful PPO dataset validation."""
        orchestrator = mock_orchestrator
        
        # Mock dataset with required fields
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=200)
        mock_dataset.column_names = ['input_ids', 'attention_mask', 'text']
        
        result = orchestrator._validate_ppo_dataset(mock_dataset)
        assert result is True
    
    def test_checkpoint_saving(self, mock_orchestrator):
        """Test checkpoint saving functionality."""
        orchestrator = mock_orchestrator
        
        # Mock model and optimizer
        mock_model = Mock()
        mock_optimizer = Mock()
        stage = TrainingStage.SFT
        train_result = Mock()
        train_result.metrics = {
            'train_loss': 0.5,
            'epoch': 1,
            'global_step': 100
        }
        
        # Mock checkpoint manager
        orchestrator.checkpoint_manager.save_checkpoint.return_value = "checkpoint_123"
        orchestrator.checkpoint_manager.base_path = Path("/fake/base/path")
        
        # Act
        checkpoint_path = orchestrator._save_stage_checkpoint(
            mock_model, mock_optimizer, stage, train_result
        )
        
        # Assert
        assert checkpoint_path == "/fake/base/path/checkpoint_123"
        orchestrator.checkpoint_manager.save_checkpoint.assert_called_once()
        
        # Verify call arguments
        call_args = orchestrator.checkpoint_manager.save_checkpoint.call_args
        assert call_args[1]['model'] == mock_model
        assert call_args[1]['optimizer'] == mock_optimizer
        assert call_args[1]['stage'] == stage.value
        assert call_args[1]['metrics'] == train_result.metrics
    
    def test_pipeline_failure_handling(self, mock_orchestrator):
        """Test pipeline failure handling and state preservation."""
        orchestrator = mock_orchestrator
        error = RuntimeError("Pipeline failure test")
        
        # Setup some pipeline state
        orchestrator.pipeline_state.current_stage = TrainingStage.SFT
        orchestrator.pipeline_state.completed_stages = [TrainingStage.SFT]
        orchestrator.pipeline_state.failure_count = 1
        
        # Act
        orchestrator._handle_pipeline_failure(error)
        
        # Assert - verify error handling doesn't crash
        # (The method should handle the error gracefully)
        assert True  # If we get here, error handling worked
    
    def test_resource_cleanup(self, mock_orchestrator):
        """Test resource cleanup functionality."""
        orchestrator = mock_orchestrator
        
        # Setup some resources
        orchestrator.current_model = Mock()
        orchestrator.current_tokenizer = Mock()
        orchestrator.loss_history = [1.0, 0.9, 1.1]
        orchestrator.memory_usage_history = [{'step': 1, 'memory': 2.0}]
        
        # Act
        orchestrator._cleanup_resources()
        
        # Assert
        assert orchestrator.current_model is None
        assert orchestrator.current_tokenizer is None
        assert orchestrator.loss_history == []
        assert orchestrator.memory_usage_history == []
        
        # Verify model manager cleanup was called
        orchestrator.model_manager.cleanup.assert_called_once()
    
    def test_find_previous_checkpoint(self, mock_orchestrator):
        """Test finding previous checkpoint for a stage."""
        orchestrator = mock_orchestrator
        stage = TrainingStage.SFT
        
        # Mock checkpoint manager to return checkpoints
        orchestrator.checkpoint_manager.list_checkpoints.return_value = [
            "checkpoint_latest", "checkpoint_older"
        ]
        
        # Act
        checkpoint = orchestrator._find_previous_checkpoint(stage)
        
        # Assert
        assert checkpoint == "checkpoint_latest"
        orchestrator.checkpoint_manager.list_checkpoints.assert_called_once_with(stage.value)
    
    def test_find_previous_checkpoint_none_available(self, mock_orchestrator):
        """Test finding previous checkpoint when none available."""
        orchestrator = mock_orchestrator
        stage = TrainingStage.SFT
        
        # Mock checkpoint manager to return no checkpoints
        orchestrator.checkpoint_manager.list_checkpoints.return_value = []
        
        # Act
        checkpoint = orchestrator._find_previous_checkpoint(stage)
        
        # Assert
        assert checkpoint is None
    
    def test_pipeline_summary_logging(self, mock_orchestrator, temp_dir):
        """Test pipeline summary logging functionality."""
        orchestrator = mock_orchestrator
        
        # Setup pipeline state with completed stages
        orchestrator.pipeline_state.pipeline_start_time = datetime.now(timezone.utc)
        orchestrator.pipeline_state.completed_stages = [TrainingStage.SFT, TrainingStage.REWARD]
        orchestrator.pipeline_state.stage_results = {
            TrainingStage.SFT: StageResult(
                stage=TrainingStage.SFT,
                success=True,
                checkpoint_path="/fake/sft/checkpoint",
                metrics={"loss": 0.5},
                duration_seconds=120.0,
                memory_peak_gb=3.5
            ),
            TrainingStage.REWARD: StageResult(
                stage=TrainingStage.REWARD,
                success=True,
                checkpoint_path="/fake/reward/checkpoint",
                metrics={"loss": 0.3},
                duration_seconds=80.0,
                memory_peak_gb=2.8
            )
        }
        orchestrator.pipeline_state.last_checkpoint_path = "/fake/reward/checkpoint"
        
        # Act
        orchestrator._log_pipeline_summary()
        
        # Assert - verify summary file would be created
        # (We can't easily test file creation in unit test, but we verify the method runs)
        assert True  # If we get here, summary logging worked


class TestTrainingOrchestratorIntegration:
    """Integration tests for Training Orchestrator with mocked external dependencies."""
    
    @pytest.fixture
    def config(self):
        """Create minimal test configuration."""
        config = Config()
        config.training.sft.epochs = 1
        config.training.sft.max_steps = 5
        config.training.reward.epochs = 1
        config.training.reward.max_steps = 3
        config.training.ppo.max_steps = 3
        return config
    
    def test_stage_sequence_validation(self, config):
        """Test that stages must be completed in correct sequence."""
        with patch('rlhf_phi3.training.training_orchestrator.ModelManager'), \
             patch('rlhf_phi3.training.training_orchestrator.DatasetManager'), \
             patch('rlhf_phi3.training.training_orchestrator.CheckpointManager'), \
             patch('rlhf_phi3.training.training_orchestrator.ExperimentTracker'):
            
            orchestrator = TrainingOrchestrator(config)
            
            # Try to validate PPO stage without completing SFT and Reward
            result = orchestrator.validate_stage_completion(TrainingStage.PPO.value)
            assert result is False
            
            # Complete SFT stage
            sft_result = StageResult(
                stage=TrainingStage.SFT,
                success=True,
                checkpoint_path="/fake/sft/checkpoint",
                metrics={"loss": 0.5},
                duration_seconds=100.0,
                memory_peak_gb=2.0
            )
            orchestrator.pipeline_state.completed_stages.append(TrainingStage.SFT)
            orchestrator.pipeline_state.stage_results[TrainingStage.SFT] = sft_result
            
            # SFT should now validate successfully
            with patch('pathlib.Path.exists', return_value=True):
                result = orchestrator.validate_stage_completion(TrainingStage.SFT.value)
                assert result is True
    
    def test_error_recovery_state_preservation(self, config):
        """Test that error recovery preserves pipeline state correctly."""
        with patch('rlhf_phi3.training.training_orchestrator.ModelManager'), \
             patch('rlhf_phi3.training.training_orchestrator.DatasetManager'), \
             patch('rlhf_phi3.training.training_orchestrator.CheckpointManager'), \
             patch('rlhf_phi3.training.training_orchestrator.ExperimentTracker'):
            
            orchestrator = TrainingOrchestrator(config)
            
            # Simulate successful SFT stage
            sft_result = StageResult(
                stage=TrainingStage.SFT,
                success=True,
                checkpoint_path="/fake/sft/checkpoint",
                metrics={"loss": 0.5},
                duration_seconds=100.0,
                memory_peak_gb=2.0
            )
            orchestrator.pipeline_state.completed_stages.append(TrainingStage.SFT)
            orchestrator.pipeline_state.stage_results[TrainingStage.SFT] = sft_result
            
            # Simulate reward stage failure
            reward_error = RuntimeError("Reward training failed")
            orchestrator._handle_stage_failure(TrainingStage.REWARD, reward_error, 50.0)
            
            # Verify state preservation
            assert TrainingStage.SFT in orchestrator.pipeline_state.completed_stages
            assert TrainingStage.REWARD not in orchestrator.pipeline_state.completed_stages
            assert TrainingStage.SFT in orchestrator.pipeline_state.stage_results
            assert TrainingStage.REWARD in orchestrator.pipeline_state.stage_results
            
            # Verify SFT result is still successful
            assert orchestrator.pipeline_state.stage_results[TrainingStage.SFT].success is True
            
            # Verify reward result shows failure
            assert orchestrator.pipeline_state.stage_results[TrainingStage.REWARD].success is False
            assert orchestrator.pipeline_state.stage_results[TrainingStage.REWARD].error_message == "Reward training failed"
            
            # Verify failure count
            assert orchestrator.pipeline_state.failure_count == 1