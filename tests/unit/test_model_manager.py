"""
Unit tests for Model Manager component.

This module contains unit tests for specific functionality of the Model Manager,
including PEFT configuration, checkpoint operations, and model merging.

Tests cover:
- PEFT configuration and application
- Checkpoint save/load round-trip consistency  
- Model merging functionality
- Memory optimization utilities
- Training preparation utilities

Requirements tested:
- 5.1: PEFT/LoRA implementation
- 10.1: Checkpoint operations and model merging
- 5.2, 5.3: Memory optimization features
"""

import pytest
import torch
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import asdict

# Import components to test
from rlhf_phi3.models.model_manager import ModelManager, MemoryOptimizer, TrainingPreparationUtils
from rlhf_phi3.config.config_manager import Config, load_colab_config
from transformers import TrainingArguments


class TestModelManager:
    """Unit tests for ModelManager class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = load_colab_config()
        self.temp_dir = tempfile.mkdtemp()
        self.config.paths.base_output_dir = self.temp_dir
        self.config.paths.cache_dir = str(Path(self.temp_dir) / "cache")
        self.model_manager = ModelManager(self.config)
        
    def teardown_method(self):
        """Cleanup test environment."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_peft_configuration_creation(self):
        """
        Test PEFT/LoRA configuration creation with various task types.
        
        Requirements: 5.1 - PEFT configuration and application
        """
        # Test default CAUSAL_LM configuration
        lora_config = self.model_manager.setup_peft_config()
        
        assert lora_config is not None
        assert lora_config.r == self.config.lora.r
        assert lora_config.lora_alpha == self.config.lora.alpha
        assert lora_config.lora_dropout == self.config.lora.dropout
        assert lora_config.target_modules == self.config.lora.target_modules
        assert lora_config.bias == self.config.lora.bias
        
        # Test different task types
        for task_type in ["CAUSAL_LM", "SEQ_CLS", "TOKEN_CLS"]:
            lora_config = self.model_manager.setup_peft_config(task_type)
            assert lora_config.task_type.name == task_type
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    def test_base_model_loading(self, mock_config, mock_tokenizer, mock_model):
        """
        Test base model loading with proper configuration.
        
        Requirements: 5.2 - Mixed precision training setup
        """
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = [Mock(numel=lambda: 1000000) for _ in range(10)]
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        # Test model loading
        with patch('torch.cuda.is_available', return_value=True):
            model = self.model_manager.load_base_model()
        
        # Verify model loading was called with correct parameters
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        
        assert call_kwargs['trust_remote_code'] is True
        assert call_kwargs['low_cpu_mem_usage'] is True
        assert call_kwargs['use_cache'] is False
        assert call_kwargs['cache_dir'] == self.config.paths.cache_dir
        
        # Verify tokenizer setup
        assert self.model_manager.tokenizer.pad_token == "</s>"
        assert self.model_manager.tokenizer.pad_token_id == 2
        
        # Verify model is stored
        assert self.model_manager.base_model == mock_model_instance
    
    @patch('peft.get_peft_model')
    @patch('peft.prepare_model_for_kbit_training')
    def test_peft_application(self, mock_prepare_kbit, mock_get_peft):
        """
        Test PEFT application to base model.
        
        Requirements: 5.1 - PEFT/LoRA to reduce trainable parameters by at least 99%
        """
        # Setup mock base model
        mock_base_model = Mock()
        mock_base_model.config = Mock()
        mock_base_model.config.quantization_config = Mock()  # Simulate quantized model
        
        # Setup mock PEFT model with parameter reduction
        mock_peft_model = Mock()
        
        # Mock parameters to simulate 99%+ reduction
        total_params = [Mock(numel=lambda: 100000) for _ in range(100)]  # 10M total params
        trainable_params = [Mock(numel=lambda: 1000, requires_grad=True) for _ in range(10)]  # 10K trainable
        frozen_params = [Mock(numel=lambda: 99000, requires_grad=False) for _ in range(100)]  # Rest frozen
        
        mock_peft_model.parameters.return_value = trainable_params + frozen_params
        mock_get_peft.return_value = mock_peft_model
        mock_prepare_kbit.return_value = mock_base_model
        
        # Test PEFT application
        lora_config = self.model_manager.setup_peft_config()
        peft_model = self.model_manager.apply_peft(mock_base_model, lora_config)
        
        # Verify PEFT was applied
        mock_prepare_kbit.assert_called_once_with(mock_base_model)
        mock_get_peft.assert_called_once_with(mock_base_model, lora_config)
        
        # Verify model is stored
        assert self.model_manager.peft_model == mock_peft_model
        assert peft_model == mock_peft_model
    
    def test_training_preparation(self):
        """
        Test model preparation for training with memory optimizations.
        
        Requirements: 5.2, 5.3 - Mixed precision and gradient checkpointing
        """
        # Setup mock PEFT model
        mock_peft_model = Mock()
        mock_peft_model.enable_input_require_grads = Mock()
        mock_peft_model.gradient_checkpointing_enable = Mock()
        mock_peft_model.train = Mock()
        
        # Test training preparation
        prepared_model = self.model_manager.prepare_for_training(mock_peft_model)
        
        # Verify gradient checkpointing was enabled
        if self.config.optimization.gradient_checkpointing:
            mock_peft_model.enable_input_require_grads.assert_called_once()
            mock_peft_model.gradient_checkpointing_enable.assert_called_once()
        
        # Verify model was set to training mode
        mock_peft_model.train.assert_called_once()
        
        assert prepared_model == mock_peft_model
    
    def test_checkpoint_save_and_load_round_trip(self):
        """
        Test checkpoint save/load round-trip consistency.
        
        Requirements: 10.1 - Checkpoint saving and loading capabilities
        """
        # Setup mock PEFT model
        mock_peft_model = Mock()
        mock_peft_model.save_pretrained = Mock()
        
        # Setup mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()
        self.model_manager.tokenizer = mock_tokenizer
        
        # Test checkpoint saving
        checkpoint_path = Path(self.temp_dir) / "test_checkpoint"
        metadata = {"epoch": 1, "step": 100, "loss": 0.5}
        
        self.model_manager.save_checkpoint(
            mock_peft_model, 
            checkpoint_path, 
            metadata=metadata
        )
        
        # Verify checkpoint directory was created
        assert checkpoint_path.exists()
        
        # Verify model and tokenizer save_pretrained were called
        mock_peft_model.save_pretrained.assert_called_once_with(checkpoint_path)
        mock_tokenizer.save_pretrained.assert_called_once_with(checkpoint_path)
        
        # Verify metadata was saved
        metadata_path = checkpoint_path / "checkpoint_metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata["model_name"] == self.config.model.name
        assert saved_metadata["epoch"] == 1
        assert saved_metadata["step"] == 100
        assert saved_metadata["loss"] == 0.5
        
        # Test checkpoint loading
        with patch('peft.PeftModel.from_pretrained') as mock_load_peft, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_load_tokenizer:
            
            mock_loaded_peft = Mock()
            mock_load_peft.return_value = mock_loaded_peft
            
            mock_loaded_tokenizer = Mock()
            mock_load_tokenizer.return_value = mock_loaded_tokenizer
            
            # Setup base model for loading
            self.model_manager.base_model = Mock()
            
            loaded_model, loaded_metadata = self.model_manager.load_checkpoint(checkpoint_path, load_base_model=False)
            
            # Verify loading was successful
            assert loaded_model == mock_loaded_peft
            assert loaded_metadata["epoch"] == 1
            assert loaded_metadata["step"] == 100
            assert loaded_metadata["loss"] == 0.5
            
            # Verify model and tokenizer were loaded
            mock_load_peft.assert_called_once()
            assert self.model_manager.peft_model == mock_loaded_peft
    
    @patch('peft.PeftModel.merge_and_unload')
    def test_model_merging_functionality(self, mock_merge):
        """
        Test model merging functionality for deployment.
        
        Requirements: 10.1 - PEFT adapter merging with base model for deployment
        """
        # Setup mock PEFT model
        mock_peft_model = Mock()
        
        # Setup mock merged model
        mock_merged_model = Mock()
        mock_merged_model.save_pretrained = Mock()
        mock_merge.return_value = mock_merged_model
        
        # Setup mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()
        self.model_manager.tokenizer = mock_tokenizer
        
        # Test model merging
        output_path = Path(self.temp_dir) / "merged_model"
        merged_model = self.model_manager.merge_and_save(mock_peft_model, output_path)
        
        # Verify merging was called
        mock_merge.assert_called_once()
        
        # Verify merged model was saved
        mock_merged_model.save_pretrained.assert_called_once()
        call_kwargs = mock_merged_model.save_pretrained.call_args[1]
        assert call_kwargs['safe_serialization'] is True
        
        # Verify tokenizer was saved
        mock_tokenizer.save_pretrained.assert_called_once()
        
        # Verify model card was created
        model_card_path = output_path / "README.md"
        assert model_card_path.exists()
        
        with open(model_card_path, 'r') as f:
            content = f.read()
        
        assert "RLHF Fine-tuned Phi-3 Model" in content
        assert self.config.model.name in content
        assert str(self.config.lora.r) in content
        assert str(self.config.lora.alpha) in content
        
        assert merged_model == mock_merged_model
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.get_device_properties')
    @patch('psutil.virtual_memory')
    def test_memory_usage_monitoring(self, mock_virtual_memory, mock_device_props, 
                                   mock_memory_reserved, mock_memory_allocated, mock_cuda_available):
        """
        Test memory usage monitoring functionality.
        
        Requirements: 2.3 - Memory monitoring for adaptive behavior
        """
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 8 * 1024**3  # 8GB
        mock_memory_reserved.return_value = 10 * 1024**3  # 10GB
        
        mock_device = Mock()
        mock_device.total_memory = 15 * 1024**3  # 15GB total
        mock_device_props.return_value = mock_device
        
        mock_vm = Mock()
        mock_vm.used = 16 * 1024**3  # 16GB CPU memory used
        mock_vm.percent = 80.0
        mock_virtual_memory.return_value = mock_vm
        
        # Test memory usage retrieval
        memory_stats = self.model_manager.get_memory_usage()
        
        # Verify GPU memory stats
        assert memory_stats["gpu_allocated_gb"] == 8.0
        assert memory_stats["gpu_reserved_gb"] == 10.0
        assert memory_stats["gpu_max_memory_gb"] == 15.0
        assert abs(memory_stats["gpu_utilization"] - (8.0/15.0)) < 0.01
        
        # Verify CPU memory stats
        assert memory_stats["cpu_memory_gb"] == 16.0
        assert memory_stats["cpu_memory_percent"] == 0.8
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.empty_cache')
    def test_memory_exhaustion_handling(self, mock_empty_cache, mock_device_props, 
                                      mock_memory_allocated, mock_cuda_available):
        """
        Test memory exhaustion detection and handling.
        
        Requirements: 9.1 - Automatic batch size reduction and training continuation on GPU memory exhaustion
        """
        # Setup mocks for high memory usage scenario
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 14 * 1024**3  # 14GB out of 15GB (93% usage)
        
        mock_device = Mock()
        mock_device.total_memory = 15 * 1024**3
        mock_device_props.return_value = mock_device
        
        # Test batch size adjustment for high memory usage
        initial_batch_size = 8
        initial_grad_accum = 2
        
        new_batch_size, new_grad_accum = self.model_manager.check_memory_and_adjust_batch_size(
            initial_batch_size, initial_grad_accum
        )
        
        # Should reduce batch size and increase gradient accumulation
        assert new_batch_size < initial_batch_size
        assert new_grad_accum > initial_grad_accum
        
        # Test memory exhaustion recovery
        recovery_success = self.model_manager.handle_memory_exhaustion()
        
        assert recovery_success is True
        mock_empty_cache.assert_called()
    
    def test_cleanup_functionality(self):
        """Test resource cleanup functionality."""
        # Setup mock objects
        self.model_manager.peft_model = Mock()
        self.model_manager.base_model = Mock()
        self.model_manager.tokenizer = Mock()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('gc.collect') as mock_gc_collect:
            
            # Test cleanup
            self.model_manager.cleanup()
            
            # Verify objects were cleared
            assert self.model_manager.peft_model is None
            assert self.model_manager.base_model is None
            assert self.model_manager.tokenizer is None
            
            # Verify cache clearing and garbage collection
            mock_empty_cache.assert_called_once()
            mock_gc_collect.assert_called_once()


class TestMemoryOptimizer:
    """Unit tests for MemoryOptimizer class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = load_colab_config()
        self.memory_optimizer = MemoryOptimizer(self.config)
    
    @patch('accelerate.Accelerator')
    def test_mixed_precision_setup(self, mock_accelerator_class):
        """
        Test mixed precision training setup.
        
        Requirements: 5.2 - Mixed precision training (fp16) to reduce memory usage by 50%
        """
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator
        
        # Test with fp16 enabled
        self.config.optimization.fp16 = True
        accelerator = self.memory_optimizer.setup_mixed_precision()
        
        # Verify accelerator was created with correct parameters
        mock_accelerator_class.assert_called_once()
        call_kwargs = mock_accelerator_class.call_args[1]
        assert call_kwargs['mixed_precision'] == "fp16"
        
        # Test with fp16 disabled
        self.config.optimization.fp16 = False
        mock_accelerator_class.reset_mock()
        
        accelerator = self.memory_optimizer.setup_mixed_precision()
        
        call_kwargs = mock_accelerator_class.call_args[1]
        assert call_kwargs['mixed_precision'] == "no"
    
    def test_gradient_checkpointing_setup(self):
        """
        Test gradient checkpointing setup.
        
        Requirements: 5.3 - Gradient checkpointing to trade compute for memory efficiency
        """
        # Test with gradient checkpointing enabled
        self.config.optimization.gradient_checkpointing = True
        
        # Mock model with gradient checkpointing support
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        mock_model.enable_input_require_grads = Mock()
        
        result_model = self.memory_optimizer.setup_gradient_checkpointing(mock_model)
        
        # Verify gradient checkpointing was enabled
        mock_model.gradient_checkpointing_enable.assert_called_once()
        mock_model.enable_input_require_grads.assert_called_once()
        assert result_model == mock_model
        
        # Test with gradient checkpointing disabled
        self.config.optimization.gradient_checkpointing = False
        mock_model.reset_mock()
        
        result_model = self.memory_optimizer.setup_gradient_checkpointing(mock_model)
        
        # Should not call gradient checkpointing methods
        mock_model.gradient_checkpointing_enable.assert_not_called()
        assert result_model == mock_model
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_properties')
    def test_batch_size_optimization(self, mock_device_props, mock_memory_allocated, mock_cuda_available):
        """
        Test automatic batch size optimization for memory constraints.
        
        Requirements: 2.3 - Automatic batch size adjustment for memory constraints
        """
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 8 * 1024**3  # 8GB
        
        mock_device = Mock()
        mock_device.total_memory = 15 * 1024**3  # 15GB
        mock_device_props.return_value = mock_device
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.eval = Mock()
        
        mock_tokenizer = Mock()
        mock_inputs = {
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'attention_mask': torch.ones(1, 512)
        }
        mock_tokenizer.return_value = mock_inputs
        
        with patch('torch.cuda.empty_cache'):
            # Test successful optimization
            optimal_batch_size, grad_accum_steps = self.memory_optimizer.optimize_batch_size_for_memory(
                mock_model, mock_tokenizer, initial_batch_size=4
            )
            
            # Verify results are reasonable
            assert optimal_batch_size >= 1
            assert optimal_batch_size <= 4  # Should not exceed initial
            assert grad_accum_steps >= 1
            
            # Verify effective batch size is maintained
            effective_batch_size = optimal_batch_size * grad_accum_steps
            assert effective_batch_size >= 4  # Should maintain or exceed initial


class TestTrainingPreparationUtils:
    """Unit tests for TrainingPreparationUtils class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = load_colab_config()
        self.training_utils = TrainingPreparationUtils(self.config)
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test environment."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_training_arguments_creation(self):
        """Test creation of training arguments for different stages."""
        output_dir = str(Path(self.temp_dir) / "training_output")
        
        # Test SFT stage
        sft_args = self.training_utils.create_training_arguments("sft", output_dir)
        
        assert isinstance(sft_args, TrainingArguments)
        assert sft_args.output_dir == output_dir
        assert sft_args.per_device_train_batch_size == self.config.training.sft.batch_size
        assert sft_args.learning_rate == self.config.training.sft.learning_rate
        assert sft_args.gradient_accumulation_steps == self.config.training.sft.gradient_accumulation_steps
        assert sft_args.fp16 == self.config.optimization.fp16
        assert sft_args.gradient_checkpointing == self.config.optimization.gradient_checkpointing
        
        # Test reward stage
        reward_args = self.training_utils.create_training_arguments("reward", output_dir)
        
        assert reward_args.per_device_train_batch_size == self.config.training.reward.batch_size
        assert reward_args.learning_rate == self.config.training.reward.learning_rate
        
        # Test PPO stage
        ppo_args = self.training_utils.create_training_arguments("ppo", output_dir)
        
        assert ppo_args.per_device_train_batch_size == self.config.training.ppo.batch_size
        assert ppo_args.learning_rate == self.config.training.ppo.learning_rate
        
        # Test with overrides
        custom_args = self.training_utils.create_training_arguments(
            "sft", output_dir, batch_size=2, gradient_accumulation_steps=8
        )
        
        assert custom_args.per_device_train_batch_size == 2
        assert custom_args.gradient_accumulation_steps == 8
    
    def test_invalid_stage_handling(self):
        """Test handling of invalid training stage."""
        output_dir = str(Path(self.temp_dir) / "training_output")
        
        with pytest.raises(ValueError, match="Invalid stage"):
            self.training_utils.create_training_arguments("invalid_stage", output_dir)
    
    @patch('transformers.get_scheduler')
    def test_optimizer_and_scheduler_preparation(self, mock_get_scheduler):
        """Test optimizer and scheduler preparation."""
        # Mock model with trainable parameters
        mock_model = Mock()
        mock_params = [Mock(requires_grad=True, numel=lambda: 1000) for _ in range(5)]
        mock_model.parameters.return_value = mock_params
        
        # Mock scheduler
        mock_scheduler = Mock()
        mock_get_scheduler.return_value = mock_scheduler
        
        # Create training arguments
        output_dir = str(Path(self.temp_dir) / "training_output")
        training_args = self.training_utils.create_training_arguments("sft", output_dir)
        
        # Test optimizer and scheduler creation
        optimizer, scheduler = self.training_utils.prepare_optimizer_and_scheduler(
            mock_model, training_args, num_training_steps=1000
        )
        
        # Verify optimizer was created
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.AdamW)
        
        # Verify scheduler was created
        mock_get_scheduler.assert_called_once()
        call_args = mock_get_scheduler.call_args
        assert call_args[0][0] == training_args.lr_scheduler_type
        assert call_args[1]['optimizer'] == optimizer
        assert call_args[1]['num_training_steps'] == 1000
        
        assert scheduler == mock_scheduler


if __name__ == "__main__":
    pytest.main([__file__])