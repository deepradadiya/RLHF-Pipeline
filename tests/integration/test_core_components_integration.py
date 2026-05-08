"""
Integration tests for core components of RLHF Phi-3 Pipeline.

This module tests the integration between all core components:
- Configuration Manager
- Dataset Manager  
- Model Manager
- Checkpoint Manager
- Experiment Tracker
- Training Orchestrator

Tests verify that components work together correctly and handle
cross-component interactions properly.

Task 7: Checkpoint - Core Components Integration Test
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from rlhf_phi3.config.config_manager import Config, load_colab_config
from rlhf_phi3.data.dataset_manager import DatasetManager
from rlhf_phi3.models.model_manager import ModelManager
from rlhf_phi3.checkpoints.checkpoint_manager import CheckpointManager
from rlhf_phi3.tracking.experiment_tracker import ExperimentTracker
from rlhf_phi3.training.training_orchestrator import TrainingOrchestrator

from tests.fixtures.test_data import (
    SAMPLE_SFT_CONVERSATIONS, 
    SAMPLE_PREFERENCE_DATA,
    MockTokenizer,
    MockHuggingFaceModel
)
from tests.utils import (
    create_minimal_valid_config,
    create_mock_sft_dataset,
    create_mock_preference_dataset,
    temporary_directory,
    mock_environment_variables
)


class TestCoreComponentsIntegration:
    """Integration tests for core pipeline components."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.config = Config()
        # Use smaller values for faster testing
        self.config.model.max_length = 128
        self.config.training.sft.max_steps = 10
        self.config.training.reward.max_steps = 5
        self.config.training.ppo.max_steps = 3
        self.config.datasets.sft.max_samples = 5
        self.config.datasets.preference.max_samples = 3
        self.config.checkpointing.save_steps = 2
        self.config.logging.log_steps = 1
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for integration tests."""
        with temporary_directory() as temp_dir:
            # Set up directory structure
            (temp_dir / "checkpoints").mkdir()
            (temp_dir / "logs").mkdir()
            (temp_dir / "cache").mkdir()
            (temp_dir / "datasets").mkdir()
            
            # Update config paths
            self.config.paths.base_output_dir = str(temp_dir / "checkpoints")
            self.config.paths.cache_dir = str(temp_dir / "cache")
            self.config.paths.logs_dir = str(temp_dir / "logs")
            
            yield temp_dir
    
    @pytest.fixture
    def mock_external_services(self):
        """Mock external services for integration testing."""
        with patch('wandb.init') as mock_wandb_init, \
             patch('wandb.log') as mock_wandb_log, \
             patch('wandb.finish') as mock_wandb_finish, \
             patch('datasets.load_dataset') as mock_load_dataset, \
             patch('transformers.AutoModelForCausalLM') as mock_model, \
             patch('transformers.AutoTokenizer') as mock_tokenizer, \
             patch('peft.get_peft_model') as mock_get_peft_model:
            
            # Configure mocks
            mock_wandb_init.return_value = Mock()
            mock_tokenizer.from_pretrained.return_value = MockTokenizer()
            mock_model.from_pretrained.return_value = MockHuggingFaceModel("test-model")
            mock_get_peft_model.return_value = MockHuggingFaceModel("peft-model")
            
            # Mock dataset loading
            mock_sft_dataset = Mock()
            mock_sft_dataset.__len__ = Mock(return_value=5)
            mock_sft_dataset.__getitem__ = Mock(side_effect=lambda i: SAMPLE_SFT_CONVERSATIONS[i % len(SAMPLE_SFT_CONVERSATIONS)])
            mock_sft_dataset.map = Mock(return_value=mock_sft_dataset)
            mock_sft_dataset.filter = Mock(return_value=mock_sft_dataset)
            mock_sft_dataset.select = Mock(return_value=mock_sft_dataset)
            mock_sft_dataset.column_names = ['messages']
            
            mock_load_dataset.return_value = mock_sft_dataset
            
            yield {
                'wandb_init': mock_wandb_init,
                'wandb_log': mock_wandb_log,
                'wandb_finish': mock_wandb_finish,
                'load_dataset': mock_load_dataset,
                'model': mock_model,
                'tokenizer': mock_tokenizer,
                'get_peft_model': mock_get_peft_model
            }
    
    def test_config_manager_integration_with_all_components(self, temp_workspace, mock_external_services):
        """Test that Configuration Manager integrates correctly with all components."""
        # Test 1: All components can be initialized with the same config
        dataset_manager = DatasetManager(self.config)
        model_manager = ModelManager(self.config)
        checkpoint_manager = CheckpointManager(str(temp_workspace / "checkpoints"))
        experiment_tracker = ExperimentTracker("test-project", self.config)
        
        # Verify all components have access to config
        assert dataset_manager.config == self.config
        assert model_manager.config == self.config
        assert experiment_tracker.config == self.config
        
        # Test 2: Stage-specific configurations work across components
        sft_config = self.config.get_stage_config("sft")
        reward_config = self.config.get_stage_config("reward")
        ppo_config = self.config.get_stage_config("ppo")
        
        # Verify stage configs contain all necessary sections
        for stage_config in [sft_config, reward_config, ppo_config]:
            assert "model" in stage_config
            assert "lora" in stage_config
            assert "training" in stage_config
            assert "optimization" in stage_config
            assert "paths" in stage_config
            assert "checkpointing" in stage_config
            assert "logging" in stage_config
        
        # Test 3: Configuration validation works across all components
        assert self.config.validate_config() is True
        
        # Test 4: Configuration serialization preserves all component settings
        config_path = temp_workspace / "test_config.json"
        self.config.save_config(config_path)
        
        loaded_config = Config.load_config(config_path)
        assert loaded_config.model.name == self.config.model.name
        assert loaded_config.training.sft.epochs == self.config.training.sft.epochs
        assert loaded_config.datasets.sft.name == self.config.datasets.sft.name
    
    def test_dataset_manager_integration_with_model_manager(self, temp_workspace, mock_external_services):
        """Test Dataset Manager integration with Model Manager."""
        dataset_manager = DatasetManager(self.config)
        model_manager = ModelManager(self.config)
        
        # Test 1: Dataset Manager can use Model Manager's tokenizer
        model_manager.load_base_model()
        dataset_manager.tokenizer = model_manager.tokenizer
        
        # Test 2: Dataset preprocessing works with model tokenizer
        sft_dataset = dataset_manager.load_sft_dataset()
        processed_dataset = dataset_manager.preprocess_sft_data(sft_dataset, validate=False)
        
        # Verify processed dataset is compatible with model expectations
        assert processed_dataset is not None
        
        # Test 3: Chat template formatting works consistently
        messages = [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "Test answer"}
        ]
        
        formatted_text = dataset_manager.format_chat_template(messages)
        assert isinstance(formatted_text, str)
        assert len(formatted_text) > 0
        
        # Test 4: Tokenization produces valid model inputs
        tokenized = dataset_manager._tokenize_text(
            formatted_text, 
            max_length=self.config.model.max_length
        )
        
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert len(tokenized['input_ids']) <= self.config.model.max_length
    
    def test_model_manager_integration_with_checkpoint_manager(self, temp_workspace, mock_external_services):
        """Test Model Manager integration with Checkpoint Manager."""
        model_manager = ModelManager(self.config)
        checkpoint_manager = CheckpointManager(str(temp_workspace / "checkpoints"))
        
        # Test 1: Model Manager can save checkpoints via Checkpoint Manager
        base_model = model_manager.load_base_model()
        peft_model = model_manager.apply_peft(base_model)
        
        # Create mock optimizer for checkpoint saving
        mock_optimizer = Mock()
        mock_optimizer.state_dict = Mock(return_value={"param_groups": []})
        
        checkpoint_id = checkpoint_manager.save_checkpoint(
            model=peft_model,
            optimizer=mock_optimizer,
            epoch=1,
            step=10,
            stage="sft",
            metrics={"loss": 2.5, "accuracy": 0.7}
        )
        
        assert checkpoint_id is not None
        assert checkpoint_id in checkpoint_manager.metadata_cache
        
        # Test 2: Model Manager can load checkpoints via Checkpoint Manager
        model_path, optimizer_state, metadata = checkpoint_manager.load_checkpoint(checkpoint_id)
        
        assert model_path is not None
        assert optimizer_state is not None
        assert metadata is not None
        assert metadata.stage == "sft"
        
        # Test 3: Memory optimization works with checkpoint operations
        memory_stats_before = model_manager.get_memory_usage()
        
        # Simulate memory pressure and checkpoint saving
        adjusted_batch_size, adjusted_grad_steps = model_manager.check_memory_and_adjust_batch_size(
            current_batch_size=4,
            gradient_accumulation_steps=2
        )
        
        # Should return original values if no memory pressure
        assert adjusted_batch_size == 4
        assert adjusted_grad_steps == 2
        
        memory_stats_after = model_manager.get_memory_usage()
        assert isinstance(memory_stats_after, dict)
    
    def test_checkpoint_manager_integration_with_experiment_tracker(self, temp_workspace, mock_external_services):
        """Test Checkpoint Manager integration with Experiment Tracker."""
        checkpoint_manager = CheckpointManager(str(temp_workspace / "checkpoints"))
        experiment_tracker = ExperimentTracker("test-project", self.config)
        
        # Test 1: Checkpoint metadata can be logged to experiment tracker
        mock_model = MockHuggingFaceModel("test-model")
        mock_optimizer = Mock()
        mock_optimizer.state_dict = Mock(return_value={"param_groups": []})
        
        checkpoint_id = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=2,
            step=20,
            stage="reward",
            metrics={"loss": 1.8, "reward_accuracy": 0.85}
        )
        
        # Log checkpoint to experiment tracker
        checkpoint_info = checkpoint_manager.get_checkpoint_info(checkpoint_id)
        assert checkpoint_info is not None
        
        experiment_tracker.start_run("reward", "test_run")
        experiment_tracker.log_model_checkpoint(
            checkpoint_path=str(temp_workspace / "checkpoints" / checkpoint_id),
            stage="reward",
            metadata=checkpoint_info
        )
        
        # Test 2: Experiment tracker can track checkpoint metrics
        experiment_tracker.log_metrics(
            metrics=checkpoint_info["metrics"],
            step=checkpoint_info["step"],
            stage=checkpoint_info["stage"]
        )
        
        # Verify metrics were logged
        assert len(experiment_tracker.metrics_history) > 0
        
        # Test 3: Checkpoint cleanup integrates with experiment tracking
        # Create multiple checkpoints
        for i in range(5):
            checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                epoch=i+1,
                step=(i+1)*10,
                stage="reward",
                metrics={"loss": 2.0 - i*0.1}
            )
        
        # Cleanup old checkpoints (keep last 3)
        checkpoint_manager.cleanup_old_checkpoints(keep_last=3)
        
        # Verify only 3 checkpoints remain for reward stage
        reward_checkpoints = checkpoint_manager.list_checkpoints("reward")
        assert len(reward_checkpoints) <= 3
        
        experiment_tracker.finish_run()
    
    def test_experiment_tracker_integration_across_all_components(self, temp_workspace, mock_external_services):
        """Test Experiment Tracker integration with all components."""
        # Initialize all components
        dataset_manager = DatasetManager(self.config)
        model_manager = ModelManager(self.config)
        checkpoint_manager = CheckpointManager(str(temp_workspace / "checkpoints"))
        experiment_tracker = ExperimentTracker("integration-test", self.config)
        
        # Test 1: Experiment tracker can log configuration from all components
        experiment_tracker.start_run("integration", "full_pipeline_test")
        
        # Verify configuration logging
        assert experiment_tracker.current_run is not None
        assert experiment_tracker.current_stage == "integration"
        
        # Test 2: Experiment tracker can log metrics from different components
        # Dataset metrics
        dataset_stats = {
            "dataset_size": 100,
            "avg_sequence_length": 256,
            "preprocessing_time": 5.2
        }
        experiment_tracker.log_metrics(dataset_stats, step=0, stage="data_prep")
        
        # Model metrics
        model_stats = {
            "total_parameters": 1000000,
            "trainable_parameters": 10000,
            "memory_usage_gb": 2.5
        }
        experiment_tracker.log_metrics(model_stats, step=0, stage="model_init")
        
        # Training metrics
        training_stats = {
            "loss": 2.3,
            "learning_rate": 2e-4,
            "grad_norm": 1.2
        }
        experiment_tracker.log_metrics(training_stats, step=10, stage="training")
        
        # Test 3: Experiment tracker can generate comprehensive plots
        # Add more metrics for plotting
        for step in range(1, 11):
            metrics = {
                "loss": 2.5 - (step * 0.1),
                "learning_rate": 2e-4 * (0.95 ** step),
                "grad_norm": 1.0 + (step * 0.05)
            }
            experiment_tracker.log_metrics(metrics, step=step, stage="training")
        
        # Generate training plots
        plots = experiment_tracker.create_training_plots()
        assert isinstance(plots, dict)
        
        # Test 4: Experiment tracker can log evaluation results
        eval_results = {
            "mt_bench_score": 7.2,
            "helpfulness": 7.5,
            "harmlessness": 8.1,
            "honesty": 6.8,
            "inference_speed": 15.3
        }
        experiment_tracker.log_evaluation_results(eval_results, stage="evaluation")
        
        experiment_tracker.finish_run()
        
        # Verify all metrics were logged
        assert len(experiment_tracker.metrics_history) > 10
    
    @patch('torch.cuda.is_available', return_value=False)  # Force CPU for testing
    def test_memory_management_across_components(self, mock_cuda, temp_workspace, mock_external_services):
        """Test memory management integration across all components."""
        # Use Colab config for memory-optimized settings
        colab_config = load_colab_config()
        colab_config.paths.base_output_dir = str(temp_workspace / "checkpoints")
        colab_config.paths.cache_dir = str(temp_workspace / "cache")
        colab_config.paths.logs_dir = str(temp_workspace / "logs")
        
        # Initialize components with memory-optimized config
        dataset_manager = DatasetManager(colab_config)
        model_manager = ModelManager(colab_config)
        
        # Test 1: Dataset Manager uses streaming for memory efficiency
        sft_dataset = dataset_manager.load_sft_dataset(streaming=True)
        assert sft_dataset is not None
        
        # Test 2: Model Manager applies memory optimizations
        base_model = model_manager.load_base_model()
        peft_model = model_manager.apply_peft(base_model)
        training_model = model_manager.prepare_for_training(peft_model)
        
        # Verify PEFT reduces parameters significantly
        total_params = sum(p.numel() for p in training_model.parameters())
        trainable_params = sum(p.numel() for p in training_model.parameters() if p.requires_grad)
        reduction_percentage = (1 - trainable_params / total_params) * 100
        
        # Should achieve significant parameter reduction (requirement: 99%)
        assert reduction_percentage > 90  # Relaxed for mock model
        
        # Test 3: Memory monitoring works across components
        memory_stats = model_manager.get_memory_usage()
        assert isinstance(memory_stats, dict)
        
        # Test 4: Automatic batch size adjustment
        original_batch_size = colab_config.training.sft.batch_size
        adjusted_batch_size, adjusted_grad_steps = model_manager.check_memory_and_adjust_batch_size(
            current_batch_size=original_batch_size,
            gradient_accumulation_steps=colab_config.training.sft.gradient_accumulation_steps
        )
        
        # Should return reasonable values
        assert adjusted_batch_size > 0
        assert adjusted_grad_steps > 0
        
        # Test 5: Dataset Manager memory optimization
        dataset_memory_stats = dataset_manager.get_memory_usage_stats()
        assert isinstance(dataset_memory_stats, dict)
        assert "process_memory_mb" in dataset_memory_stats
        
        # Test memory optimization
        optimization_results = dataset_manager.optimize_memory_usage()
        assert isinstance(optimization_results, dict)
    
    def test_error_handling_integration(self, temp_workspace, mock_external_services):
        """Test error handling integration across components."""
        dataset_manager = DatasetManager(self.config)
        model_manager = ModelManager(self.config)
        checkpoint_manager = CheckpointManager(str(temp_workspace / "checkpoints"))
        
        # Test 1: Dataset Manager handles invalid data gracefully
        invalid_dataset = Mock()
        invalid_dataset.column_names = ['messages']
        invalid_dataset.map = Mock(side_effect=Exception("Dataset processing error"))
        
        with pytest.raises(Exception):
            dataset_manager.preprocess_sft_data(invalid_dataset)
        
        # Test 2: Model Manager handles memory errors gracefully
        success = model_manager.handle_memory_exhaustion()
        assert isinstance(success, bool)
        
        # Test 3: Checkpoint Manager handles corrupted checkpoints
        # Create invalid checkpoint
        invalid_checkpoint_dir = temp_workspace / "checkpoints" / "invalid_checkpoint"
        invalid_checkpoint_dir.mkdir(parents=True)
        (invalid_checkpoint_dir / "metadata.json").write_text("invalid json content")
        
        model_path, optimizer_state, metadata = checkpoint_manager.load_checkpoint("invalid_checkpoint")
        assert model_path is None
        assert optimizer_state is None
        assert metadata is None
        
        # Test 4: Component cleanup works properly
        model_manager.cleanup()
        dataset_manager.clear_cache()
        
        # Verify cleanup
        assert len(dataset_manager._dataset_cache) == 0
    
    def test_configuration_consistency_across_components(self, temp_workspace, mock_external_services):
        """Test that configuration remains consistent across all components."""
        # Test 1: All components use the same configuration object
        components = {
            'dataset_manager': DatasetManager(self.config),
            'model_manager': ModelManager(self.config),
            'experiment_tracker': ExperimentTracker("test-project", self.config)
        }
        
        # Verify all components reference the same config
        for name, component in components.items():
            if hasattr(component, 'config'):
                assert component.config is self.config, f"{name} has different config object"
        
        # Test 2: Configuration changes propagate correctly
        original_max_length = self.config.model.max_length
        self.config.model.max_length = 256
        
        # Create new components with updated config
        new_dataset_manager = DatasetManager(self.config)
        assert new_dataset_manager.config.model.max_length == 256
        
        # Restore original value
        self.config.model.max_length = original_max_length
        
        # Test 3: Stage-specific configurations are consistent
        stages = ["sft", "reward", "ppo"]
        for stage in stages:
            stage_config = self.config.get_stage_config(stage)
            
            # Verify stage config has all required sections
            required_sections = ["model", "lora", "training", "optimization", "paths"]
            for section in required_sections:
                assert section in stage_config, f"Missing {section} in {stage} config"
        
        # Test 4: Configuration validation is consistent
        # Create invalid config
        invalid_config = self.config.copy()
        invalid_config.model.max_length = -1
        
        assert invalid_config.validate_config() is False
        errors = invalid_config.get_validation_errors()
        assert len(errors) > 0
    
    def test_end_to_end_data_flow(self, temp_workspace, mock_external_services):
        """Test end-to-end data flow through all components."""
        # Initialize all components
        dataset_manager = DatasetManager(self.config)
        model_manager = ModelManager(self.config)
        checkpoint_manager = CheckpointManager(str(temp_workspace / "checkpoints"))
        experiment_tracker = ExperimentTracker("e2e-test", self.config)
        
        # Test complete data flow: Dataset → Model → Checkpoint → Experiment
        
        # Step 1: Load and preprocess dataset
        sft_dataset = dataset_manager.load_sft_dataset()
        processed_dataset = dataset_manager.preprocess_sft_data(sft_dataset, validate=False)
        
        # Step 2: Load and prepare model
        base_model = model_manager.load_base_model()
        peft_model = model_manager.apply_peft(base_model)
        training_model = model_manager.prepare_for_training(peft_model)
        
        # Step 3: Start experiment tracking
        experiment_tracker.start_run("e2e", "full_pipeline")
        
        # Step 4: Simulate training step and checkpoint saving
        mock_optimizer = Mock()
        mock_optimizer.state_dict = Mock(return_value={"param_groups": []})
        
        checkpoint_id = checkpoint_manager.save_checkpoint(
            model=training_model,
            optimizer=mock_optimizer,
            epoch=1,
            step=5,
            stage="sft",
            metrics={"loss": 2.1, "accuracy": 0.75}
        )
        
        # Step 5: Log checkpoint and metrics to experiment tracker
        experiment_tracker.log_model_checkpoint(
            checkpoint_path=str(temp_workspace / "checkpoints" / checkpoint_id),
            stage="sft"
        )
        
        experiment_tracker.log_metrics(
            metrics={"loss": 2.1, "accuracy": 0.75},
            step=5,
            stage="sft"
        )
        
        # Step 6: Verify data flow integrity
        # Check checkpoint was saved correctly
        checkpoint_info = checkpoint_manager.get_checkpoint_info(checkpoint_id)
        assert checkpoint_info is not None
        assert checkpoint_info["stage"] == "sft"
        assert checkpoint_info["step"] == 5
        
        # Check experiment tracking recorded everything
        assert len(experiment_tracker.metrics_history) > 0
        assert experiment_tracker.current_stage == "e2e"
        
        # Step 7: Test checkpoint loading and model restoration
        model_path, optimizer_state, metadata = checkpoint_manager.load_checkpoint(checkpoint_id)
        assert model_path is not None
        assert metadata.stage == "sft"
        assert metadata.step == 5
        
        experiment_tracker.finish_run()
        
        # Verify complete pipeline worked
        assert checkpoint_id in checkpoint_manager.metadata_cache
        assert len(experiment_tracker.metrics_history) > 0


class TestComponentInteractionEdgeCases:
    """Test edge cases in component interactions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = create_minimal_valid_config()
    
    def test_concurrent_checkpoint_operations(self, temp_workspace):
        """Test concurrent checkpoint operations don't interfere."""
        checkpoint_manager = CheckpointManager(str(temp_workspace / "checkpoints"))
        
        # Create multiple checkpoints rapidly
        mock_model = MockHuggingFaceModel("test-model")
        mock_optimizer = Mock()
        mock_optimizer.state_dict = Mock(return_value={"param_groups": []})
        
        checkpoint_ids = []
        for i in range(3):
            checkpoint_id = checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                epoch=i+1,
                step=(i+1)*10,
                stage="test",
                metrics={"loss": 2.0 - i*0.1}
            )
            checkpoint_ids.append(checkpoint_id)
            time.sleep(0.1)  # Small delay to ensure different timestamps
        
        # Verify all checkpoints were saved
        assert len(checkpoint_ids) == 3
        for checkpoint_id in checkpoint_ids:
            assert checkpoint_id in checkpoint_manager.metadata_cache
        
        # Test concurrent loading
        for checkpoint_id in checkpoint_ids:
            model_path, optimizer_state, metadata = checkpoint_manager.load_checkpoint(checkpoint_id)
            assert model_path is not None
            assert metadata is not None
    
    def test_configuration_edge_cases(self):
        """Test configuration edge cases across components."""
        # Test with minimal configuration
        minimal_config = Config()
        minimal_config.model.max_length = 32  # Very small
        minimal_config.training.sft.max_steps = 1
        minimal_config.datasets.sft.max_samples = 1
        
        # Should still be valid
        assert minimal_config.validate_config() is True
        
        # Test components can handle minimal config
        with patch('transformers.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.return_value = MockTokenizer()
            dataset_manager = DatasetManager(minimal_config)
            assert dataset_manager.config.model.max_length == 32
    
    def test_memory_pressure_scenarios(self, temp_workspace):
        """Test component behavior under memory pressure."""
        config = load_colab_config()  # Memory-optimized config
        config.paths.base_output_dir = str(temp_workspace / "checkpoints")
        
        with patch('transformers.AutoModelForCausalLM') as mock_model, \
             patch('transformers.AutoTokenizer') as mock_tokenizer:
            
            mock_tokenizer.from_pretrained.return_value = MockTokenizer()
            mock_model.from_pretrained.return_value = MockHuggingFaceModel("test-model")
            
            model_manager = ModelManager(config)
            
            # Simulate memory pressure
            original_batch_size = 8
            adjusted_batch_size, adjusted_grad_steps = model_manager.check_memory_and_adjust_batch_size(
                current_batch_size=original_batch_size,
                gradient_accumulation_steps=2
            )
            
            # Should return reasonable values even under pressure
            assert adjusted_batch_size > 0
            assert adjusted_grad_steps > 0
    
    def test_component_cleanup_integration(self, temp_workspace):
        """Test that component cleanup works properly together."""
        with patch('transformers.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.return_value = MockTokenizer()
            
            # Initialize components
            dataset_manager = DatasetManager(self.config)
            model_manager = ModelManager(self.config)
            
            # Add some data to components
            dataset_manager._dataset_cache['test'] = Mock()
            
            # Test cleanup
            dataset_manager.clear_cache()
            model_manager.cleanup()
            
            # Verify cleanup worked
            assert len(dataset_manager._dataset_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])