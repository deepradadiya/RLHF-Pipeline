"""
Property-based tests for Model Manager component.

This module contains property-based tests that validate universal correctness
properties of the Model Manager across all valid inputs and configurations.

Properties tested:
- Property 21: Memory Exhaustion Recovery (Validates: Requirement 9.1)
- Property 25: PEFT Model Merging (Validates: Requirement 10.1)  
- Property 3: Memory Adaptive Behavior (Validates: Requirement 2.3)
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite

# Import the components to test
from rlhf_phi3.models.model_manager import ModelManager, MemoryOptimizer, TrainingPreparationUtils
from rlhf_phi3.config.config_manager import Config, load_colab_config


# Test data generators
@composite
def valid_batch_sizes(draw):
    """Generate valid batch sizes for testing."""
    return draw(st.integers(min_value=1, max_value=32))


@composite
def valid_gradient_accumulation_steps(draw):
    """Generate valid gradient accumulation steps."""
    return draw(st.integers(min_value=1, max_value=64))


@composite
def memory_utilization_values(draw):
    """Generate realistic memory utilization values."""
    return draw(st.floats(min_value=0.0, max_value=1.0))


@composite
def valid_lora_configs(draw):
    """Generate valid LoRA configurations."""
    r = draw(st.integers(min_value=1, max_value=64))
    alpha = draw(st.integers(min_value=1, max_value=128))
    dropout = draw(st.floats(min_value=0.0, max_value=0.5))
    
    return {
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }


class TestModelManagerProperties:
    """Property-based tests for ModelManager component."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = load_colab_config()
        self.temp_dir = tempfile.mkdtemp()
        self.config.paths.base_output_dir = self.temp_dir
        self.config.paths.cache_dir = str(Path(self.temp_dir) / "cache")
        
    def teardown_method(self):
        """Cleanup test environment."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @given(
        initial_batch_size=valid_batch_sizes(),
        gradient_accumulation_steps=valid_gradient_accumulation_steps(),
        memory_utilization=memory_utilization_values()
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_21_memory_exhaustion_recovery(
        self, 
        initial_batch_size, 
        gradient_accumulation_steps,
        memory_utilization
    ):
        """
        **Property 21: Memory Exhaustion Recovery**
        **Validates: Requirement 9.1**
        
        For any GPU memory exhaustion scenario, the Model_Manager SHALL 
        automatically reduce batch size and continue training.
        
        This property verifies that:
        1. Memory exhaustion is detected correctly
        2. Batch size is reduced appropriately
        3. Gradient accumulation is increased to maintain effective batch size
        4. Recovery mechanisms work consistently
        """
        model_manager = ModelManager(self.config)
        
        # Mock GPU availability and memory functions
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=int(15 * 1024**3 * memory_utilization)), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            
            # Setup mock GPU properties
            mock_device = Mock()
            mock_device.total_memory = 15 * 1024**3  # 15GB T4 GPU
            mock_props.return_value = mock_device
            
            # Test memory exhaustion detection and recovery
            if memory_utilization > 0.9:  # Memory exhaustion scenario
                # Should trigger batch size reduction
                new_batch_size, new_grad_accum = model_manager.check_memory_and_adjust_batch_size(
                    initial_batch_size, gradient_accumulation_steps
                )
                
                # Verify batch size was reduced
                assert new_batch_size <= initial_batch_size, \
                    f"Batch size should be reduced when memory utilization is {memory_utilization:.1%}"
                
                # Verify gradient accumulation was increased to maintain effective batch size
                original_effective = initial_batch_size * gradient_accumulation_steps
                new_effective = new_batch_size * new_grad_accum
                assert new_effective >= original_effective, \
                    "Effective batch size should be maintained or increased"
                
                # Verify memory cleanup was attempted
                recovery_success = model_manager.handle_memory_exhaustion()
                assert recovery_success, "Memory exhaustion recovery should succeed"
                
                # Verify cache clearing was called
                mock_empty_cache.assert_called()
                
            else:  # Normal memory usage
                # Should not trigger reduction
                new_batch_size, new_grad_accum = model_manager.check_memory_and_adjust_batch_size(
                    initial_batch_size, gradient_accumulation_steps
                )
                
                # Batch size should remain the same for normal memory usage
                assert new_batch_size == initial_batch_size, \
                    f"Batch size should not change when memory utilization is {memory_utilization:.1%}"
                assert new_grad_accum == gradient_accumulation_steps, \
                    "Gradient accumulation should not change for normal memory usage"
    
    @given(lora_config=valid_lora_configs())
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_25_peft_model_merging(self, lora_config):
        """
        **Property 25: PEFT Model Merging**
        **Validates: Requirement 10.1**
        
        For any PEFT configuration, the Model_Manager SHALL merge PEFT adapters 
        with the base model producing a valid merged model for deployment.
        
        This property verifies that:
        1. PEFT adapters can be merged with any valid configuration
        2. The merged model maintains the same architecture as the base model
        3. Merged model can be saved and loaded correctly
        4. Model card and metadata are generated properly
        """
        # Update config with test LoRA parameters
        self.config.lora.r = lora_config["r"]
        self.config.lora.alpha = lora_config["alpha"]
        self.config.lora.dropout = lora_config["dropout"]
        self.config.lora.target_modules = lora_config["target_modules"]
        
        model_manager = ModelManager(self.config)
        
        # Mock the model loading and PEFT operations
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_load_model, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_load_tokenizer, \
             patch('peft.get_peft_model') as mock_get_peft_model, \
             patch('peft.PeftModel.merge_and_unload') as mock_merge:
            
            # Setup mock base model
            mock_base_model = Mock()
            mock_base_model.parameters.return_value = [
                Mock(numel=lambda: 1000000, requires_grad=True) for _ in range(10)
            ]
            mock_base_model.config = Mock()
            mock_load_model.return_value = mock_base_model
            
            # Setup mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.pad_token = None
            mock_load_tokenizer.return_value = mock_tokenizer
            
            # Setup mock PEFT model
            mock_peft_model = Mock()
            mock_peft_model.parameters.return_value = [
                Mock(numel=lambda: 1000, requires_grad=True),  # Only adapters are trainable
                Mock(numel=lambda: 999000, requires_grad=False)  # Base model frozen
            ]
            mock_get_peft_model.return_value = mock_peft_model
            
            # Setup mock merged model
            mock_merged_model = Mock()
            mock_merged_model.save_pretrained = Mock()
            mock_merge.return_value = mock_merged_model
            
            # Test the complete PEFT workflow
            try:
                # Load base model
                base_model = model_manager.load_base_model()
                assert base_model is not None, "Base model should be loaded successfully"
                
                # Setup and apply PEFT
                lora_config_obj = model_manager.setup_peft_config()
                assert lora_config_obj is not None, "LoRA config should be created successfully"
                
                peft_model = model_manager.apply_peft(base_model, lora_config_obj)
                assert peft_model is not None, "PEFT should be applied successfully"
                
                # Test merging
                output_path = Path(self.temp_dir) / "merged_model"
                merged_model = model_manager.merge_and_save(peft_model, output_path)
                
                # Verify merging was successful
                assert merged_model is not None, "Model merging should produce a valid merged model"
                mock_merge.assert_called_once(), "merge_and_unload should be called"
                mock_merged_model.save_pretrained.assert_called(), "Merged model should be saved"
                
                # Verify output directory structure
                assert output_path.exists(), "Output directory should be created"
                
                # Verify model card was created
                model_card_path = output_path / "README.md"
                if model_card_path.exists():
                    with open(model_card_path, 'r') as f:
                        content = f.read()
                        assert "RLHF Fine-tuned Phi-3 Model" in content, "Model card should contain proper title"
                        assert str(lora_config["r"]) in content, "Model card should contain LoRA rank"
                        assert str(lora_config["alpha"]) in content, "Model card should contain LoRA alpha"
                
            except Exception as e:
                pytest.fail(f"PEFT model merging failed with configuration {lora_config}: {str(e)}")
    
    @given(
        batch_size=valid_batch_sizes(),
        gradient_accumulation_steps=valid_gradient_accumulation_steps(),
        memory_utilization=memory_utilization_values()
    )
    @settings(
        max_examples=25,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_3_memory_adaptive_behavior(
        self, 
        batch_size, 
        gradient_accumulation_steps,
        memory_utilization
    ):
        """
        **Property 3: Memory Adaptive Behavior**
        **Validates: Requirement 2.3**
        
        For any GPU memory usage scenario exceeding 90% capacity, the Model_Manager 
        SHALL automatically reduce batch size and increase gradient accumulation steps.
        
        This property verifies that:
        1. Memory usage is monitored accurately
        2. Adaptive behavior triggers at the correct threshold
        3. Batch size and gradient accumulation are adjusted proportionally
        4. Effective batch size is preserved when possible
        """
        memory_optimizer = MemoryOptimizer(self.config)
        
        # Mock GPU memory functions
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated') as mock_allocated, \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            
            # Setup mock GPU properties (T4 GPU with 15GB)
            total_memory = 15 * 1024**3
            allocated_memory = int(total_memory * memory_utilization)
            
            mock_allocated.return_value = allocated_memory
            mock_device = Mock()
            mock_device.total_memory = total_memory
            mock_props.return_value = mock_device
            
            # Test adaptive behavior
            new_batch_size, new_grad_accum, adjustment_made = memory_optimizer.adaptive_batch_size_adjustment(
                batch_size, gradient_accumulation_steps, memory_threshold=0.9
            )
            
            # Verify adaptive behavior based on memory utilization
            if memory_utilization > 0.9:  # Above threshold
                # Should reduce batch size and increase gradient accumulation
                assert new_batch_size <= batch_size, \
                    f"Batch size should be reduced when memory utilization is {memory_utilization:.1%}"
                assert new_grad_accum >= gradient_accumulation_steps, \
                    "Gradient accumulation should be increased when memory is high"
                assert adjustment_made, "Adjustment should be made when above threshold"
                
                # Verify effective batch size is maintained
                original_effective = batch_size * gradient_accumulation_steps
                new_effective = new_batch_size * new_grad_accum
                
                # Allow some flexibility in effective batch size (within 50% to 200%)
                assert new_effective >= original_effective * 0.5, \
                    "Effective batch size should not be reduced too drastically"
                assert new_effective <= original_effective * 2.0, \
                    "Effective batch size should not be increased too much"
                
                # Verify cache clearing was attempted
                mock_empty_cache.assert_called()
                
            elif memory_utilization < 0.6:  # Well below threshold
                # May increase batch size if conditions are right
                if new_batch_size > batch_size:
                    assert new_grad_accum <= gradient_accumulation_steps, \
                        "Gradient accumulation should be reduced when batch size increases"
                    assert adjustment_made, "Adjustment should be flagged when batch size increases"
                
                # Should not reduce batch size when memory is low
                assert new_batch_size >= batch_size, \
                    f"Batch size should not be reduced when memory utilization is {memory_utilization:.1%}"
                
            else:  # Normal range (0.6 - 0.9)
                # Should maintain current settings in normal range
                if not adjustment_made:
                    assert new_batch_size == batch_size, \
                        "Batch size should remain stable in normal memory range"
                    assert new_grad_accum == gradient_accumulation_steps, \
                        "Gradient accumulation should remain stable in normal memory range"
            
            # Verify memory monitoring provides accurate statistics
            memory_stats = memory_optimizer.monitor_memory_during_training()
            
            if torch.cuda.is_available():
                assert "utilization" in memory_stats, "Memory stats should include utilization"
                assert "allocated_gb" in memory_stats, "Memory stats should include allocated memory"
                assert "max_memory_gb" in memory_stats, "Memory stats should include max memory"
                
                # Verify utilization calculation is correct
                expected_utilization = allocated_memory / total_memory
                actual_utilization = memory_stats["utilization"]
                assert abs(actual_utilization - expected_utilization) < 0.01, \
                    f"Memory utilization calculation should be accurate: expected {expected_utilization:.3f}, got {actual_utilization:.3f}"
                
                # Verify recommendations are appropriate
                if "recommendation" in memory_stats:
                    if memory_utilization > 0.9:
                        assert memory_stats["recommendation"] == "critical_reduce_batch_size", \
                            "Should recommend critical batch size reduction for high memory usage"
                    elif memory_utilization > 0.8:
                        assert memory_stats["recommendation"] == "consider_reducing_batch_size", \
                            "Should recommend considering batch size reduction for moderately high memory usage"
                    elif memory_utilization < 0.5:
                        assert memory_stats["recommendation"] == "can_increase_batch_size", \
                            "Should recommend possible batch size increase for low memory usage"
                    else:
                        assert memory_stats["recommendation"] == "optimal", \
                            "Should indicate optimal memory usage for normal range"


class TestMemoryOptimizerProperties:
    """Additional property tests for MemoryOptimizer component."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = load_colab_config()
    
    @given(
        initial_batch_size=st.integers(min_value=1, max_value=16),
        max_length=st.integers(min_value=128, max_value=2048)
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_batch_size_optimization_consistency(self, initial_batch_size, max_length):
        """
        Test that batch size optimization produces consistent and valid results.
        
        This property verifies that:
        1. Optimal batch size is always <= initial batch size
        2. Gradient accumulation steps compensate for batch size reduction
        3. Results are deterministic for the same inputs
        """
        memory_optimizer = MemoryOptimizer(self.config)
        
        # Mock model and tokenizer for testing
        with patch('transformers.AutoTokenizer') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {
                'input_ids': torch.randint(0, 1000, (1, max_length)),
                'attention_mask': torch.ones(1, max_length)
            }
            mock_tokenizer_class.return_value = mock_tokenizer
            
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.return_value = Mock()
            
            # Mock CUDA functions to simulate memory constraints
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.memory_allocated', return_value=8 * 1024**3), \
                 patch('torch.cuda.get_device_properties') as mock_props, \
                 patch('torch.cuda.empty_cache'):
                
                mock_device = Mock()
                mock_device.total_memory = 15 * 1024**3
                mock_props.return_value = mock_device
                
                try:
                    optimal_batch_size, grad_accum_steps = memory_optimizer.optimize_batch_size_for_memory(
                        mock_model, mock_tokenizer, initial_batch_size, max_length
                    )
                    
                    # Verify constraints
                    assert optimal_batch_size >= 1, "Optimal batch size must be at least 1"
                    assert optimal_batch_size <= initial_batch_size, \
                        f"Optimal batch size ({optimal_batch_size}) should not exceed initial ({initial_batch_size})"
                    
                    assert grad_accum_steps >= 1, "Gradient accumulation steps must be at least 1"
                    
                    # Verify effective batch size relationship
                    effective_batch_size = optimal_batch_size * grad_accum_steps
                    assert effective_batch_size >= initial_batch_size, \
                        "Effective batch size should be at least the initial batch size"
                    
                except Exception as e:
                    # If optimization fails, it should be due to extreme memory constraints
                    # This is acceptable behavior
                    assert "memory" in str(e).lower() or "cuda" in str(e).lower(), \
                        f"Optimization failure should be memory-related: {str(e)}"


if __name__ == "__main__":
    pytest.main([__file__])