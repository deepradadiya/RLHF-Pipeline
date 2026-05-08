"""
Property-Based Tests for Error Handling and Recovery Systems

This module contains property-based tests for the error handling framework,
validating that error recovery mechanisms work correctly across all scenarios.

Properties tested:
- Property 22: Dataset Loading Fallback (validates Requirement 9.2)
- Property 24: Comprehensive Error Handling (validates Requirement 9.5)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import torch
import gc

from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from rlhf_phi3.config.config_manager import Config
from rlhf_phi3.utils.error_handler import (
    ErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext,
    MemoryRecoveryStrategy, DatasetRecoveryStrategy, 
    AuthenticationRecoveryStrategy, TrainingRecoveryStrategy
)
from rlhf_phi3.data.dataset_manager import DatasetManager
from rlhf_phi3.checkpoints.checkpoint_manager import CheckpointManager


class TestDatasetLoadingFallback:
    """
    **Property 22: Dataset Loading Fallback**
    **Validates: Requirement 9.2**
    
    For any dataset loading failure scenario, the Dataset_Manager SHALL attempt 
    alternative sources and provide clear guidance.
    """
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.paths.cache_dir = temp_dir
            config.paths.base_output_dir = temp_dir
            config.paths.logs_dir = temp_dir
            yield config
    
    @pytest.fixture
    def dataset_manager(self, temp_config):
        """Create dataset manager with mocked tokenizer."""
        with patch('rlhf_phi3.data.dataset_manager.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_tokenizer.from_pretrained.return_value.pad_token = "[PAD]"
            mock_tokenizer.from_pretrained.return_value.eos_token = "[EOS]"
            mock_tokenizer.from_pretrained.return_value.apply_chat_template = Mock(return_value="formatted_text")
            
            manager = DatasetManager(temp_config)
            manager.tokenizer = mock_tokenizer.from_pretrained.return_value
            yield manager
    
    @given(
        dataset_name=st.text(min_size=1, max_size=50).filter(lambda x: '/' in x),
        dataset_type=st.sampled_from(['sft', 'preference']),
        max_retries=st.integers(min_value=1, max_value=5),
        streaming=st.booleans()
    )
    @settings(max_examples=20, deadline=30000)
    def test_dataset_fallback_property(self, dataset_manager, dataset_name, dataset_type, max_retries, streaming):
        """
        Property: For any dataset loading failure, fallback mechanisms are attempted.
        
        This test verifies that when a dataset fails to load, the system:
        1. Attempts alternative datasets
        2. Provides clear error guidance
        3. Returns None only after all alternatives fail
        4. Logs appropriate recovery instructions
        """
        assume(dataset_name.count('/') == 1)  # Valid HuggingFace format
        assume(not dataset_name.startswith('/') and not dataset_name.endswith('/'))
        
        # Mock dataset loading to always fail for primary dataset
        with patch('rlhf_phi3.data.dataset_manager.load_dataset') as mock_load:
            # Make primary dataset fail
            mock_load.side_effect = Exception("Dataset loading failed")
            
            # Mock logger to capture recovery instructions
            with patch('rlhf_phi3.data.dataset_manager.logger') as mock_logger:
                result = dataset_manager.load_dataset_with_fallback(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    streaming=streaming,
                    max_retries=max_retries
                )
                
                # Property: Result should be None when all alternatives fail
                assert result is None
                
                # Property: Should attempt to load multiple datasets (primary + alternatives)
                expected_calls = max_retries * (1 + len(dataset_manager.alternative_datasets.get(dataset_type, [])))
                assert mock_load.call_count >= max_retries  # At least tried primary dataset
                
                # Property: Should log recovery instructions
                error_calls = [call for call in mock_logger.error.call_args_list if call[0]]
                assert len(error_calls) > 0, "Should log recovery instructions when all datasets fail"
                
                # Property: Should log warning about fallback attempts
                warning_calls = [call for call in mock_logger.warning.call_args_list if call[0]]
                assert len(warning_calls) > 0, "Should log warnings about failed attempts"
    
    @given(
        dataset_type=st.sampled_from(['sft', 'preference']),
        alternative_index=st.integers(min_value=0, max_value=3)
    )
    @settings(max_examples=10, deadline=20000)
    def test_successful_fallback_property(self, dataset_manager, dataset_type, alternative_index):
        """
        Property: When an alternative dataset succeeds, it should be returned with appropriate logging.
        """
        alternatives = dataset_manager.alternative_datasets.get(dataset_type, [])
        assume(alternative_index < len(alternatives))
        
        primary_dataset = "fake/primary-dataset"
        successful_alternative = alternatives[alternative_index]
        
        # Mock successful loading for the alternative
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        with patch('rlhf_phi3.data.dataset_manager.load_dataset') as mock_load:
            def load_side_effect(dataset_name, **kwargs):
                if dataset_name == primary_dataset:
                    raise Exception("Primary dataset failed")
                elif dataset_name == successful_alternative:
                    return mock_dataset
                else:
                    raise Exception("Other datasets failed")
            
            mock_load.side_effect = load_side_effect
            
            with patch('rlhf_phi3.data.dataset_manager.logger') as mock_logger:
                result = dataset_manager.load_dataset_with_fallback(
                    dataset_name=primary_dataset,
                    dataset_type=dataset_type,
                    streaming=False,
                    max_retries=1
                )
                
                # Property: Should return the successful alternative dataset
                assert result is not None
                assert result == mock_dataset
                
                # Property: Should log warning about using fallback
                warning_messages = [str(call[0][0]) for call in mock_logger.warning.call_args_list]
                fallback_logged = any("fallback" in msg.lower() for msg in warning_messages)
                assert fallback_logged, "Should log warning about using fallback dataset"
    
    def test_dataset_health_validation_property(self, dataset_manager):
        """
        Property: Dataset health validation should always return a structured report.
        """
        # Test with various mock datasets
        test_datasets = [
            # Valid SFT dataset
            Mock(column_names=['messages'], features={'messages': 'list'}, __len__=Mock(return_value=100)),
            # Valid preference dataset  
            Mock(column_names=['prompt', 'chosen', 'rejected'], features={'prompt': 'string'}, __len__=Mock(return_value=50)),
            # Empty dataset
            Mock(column_names=[], features={}, __len__=Mock(return_value=0)),
        ]
        
        for i, mock_dataset in enumerate(test_datasets):
            # Mock dataset iteration
            if i == 0:  # SFT dataset
                mock_dataset.__iter__ = Mock(return_value=iter([
                    {'messages': [{'role': 'user', 'content': 'test'}]},
                    {'messages': [{'role': 'assistant', 'content': 'response'}]}
                ]))
            elif i == 1:  # Preference dataset
                mock_dataset.__iter__ = Mock(return_value=iter([
                    {'prompt': 'test', 'chosen': 'good', 'rejected': 'bad'},
                    {'prompt': 'test2', 'chosen': 'better', 'rejected': 'worse'}
                ]))
            else:  # Empty dataset
                mock_dataset.__iter__ = Mock(return_value=iter([]))
            
            # Mock indexing
            if i == 0:
                mock_dataset.__getitem__ = Mock(side_effect=lambda idx: {'messages': [{'role': 'user', 'content': f'test{idx}'}]})
            elif i == 1:
                mock_dataset.__getitem__ = Mock(side_effect=lambda idx: {'prompt': f'test{idx}', 'chosen': 'good', 'rejected': 'bad'})
            else:
                mock_dataset.__getitem__ = Mock(side_effect=IndexError("Empty dataset"))
            
            health_report = dataset_manager.validate_dataset_health(mock_dataset)
            
            # Property: Health report should always have required fields
            required_fields = [
                'dataset_type', 'is_streaming', 'sample_count', 'column_names',
                'features', 'issues', 'recommendations', 'sample_validation'
            ]
            
            for field in required_fields:
                assert field in health_report, f"Health report missing required field: {field}"
            
            # Property: Issues and recommendations should be lists
            assert isinstance(health_report['issues'], list)
            assert isinstance(health_report['recommendations'], list)
            
            # Property: Sample validation should be a dict
            assert isinstance(health_report['sample_validation'], dict)


class TestComprehensiveErrorHandling:
    """
    **Property 24: Comprehensive Error Handling**
    **Validates: Requirement 9.5**
    
    For any failure scenario, the RLHF_Pipeline SHALL provide detailed error logs 
    and recovery instructions.
    """
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.paths.cache_dir = temp_dir
            config.paths.base_output_dir = temp_dir
            config.paths.logs_dir = temp_dir
            yield config
    
    @pytest.fixture
    def error_handler(self, temp_config):
        """Create error handler for testing."""
        return ErrorHandler(temp_config)
    
    @given(
        error_message=st.text(min_size=1, max_size=200),
        category=st.sampled_from(list(ErrorCategory)),
        severity=st.sampled_from(list(ErrorSeverity)),
        auto_recover=st.booleans()
    )
    @settings(max_examples=20, deadline=10000)
    def test_comprehensive_error_handling_property(self, error_handler, error_message, category, severity, auto_recover):
        """
        Property: For any error scenario, comprehensive handling is provided.
        
        This test verifies that for any error:
        1. Error context is properly created and stored
        2. Recovery instructions are generated
        3. Appropriate logging occurs
        4. Recovery is attempted if enabled
        5. Error statistics are updated
        """
        # Create a test exception
        test_exception = Exception(error_message)
        
        # Mock logger to capture all log calls
        with patch('rlhf_phi3.utils.error_handler.logger') as mock_logger:
            # Handle the error
            error_context = error_handler.handle_error(
                error=test_exception,
                category=category,
                severity=severity,
                auto_recover=auto_recover
            )
            
            # Property: Error context should be properly structured
            assert isinstance(error_context, ErrorContext)
            assert error_context.error_type == "Exception"
            assert error_context.message == error_message
            assert error_context.category == category
            assert error_context.severity == severity
            assert error_context.auto_recovery_attempted == auto_recover
            
            # Property: Error should be added to history
            assert len(error_handler.error_history) > 0
            assert error_handler.error_history[-1] == error_context
            
            # Property: Recovery instructions should be generated
            assert 'recovery_instructions' in error_context.details
            instructions = error_context.details['recovery_instructions']
            assert isinstance(instructions, str)
            assert len(instructions) > 0
            
            # Property: Appropriate logging should occur based on severity
            log_calls = (mock_logger.critical.call_args_list + 
                        mock_logger.error.call_args_list + 
                        mock_logger.warning.call_args_list + 
                        mock_logger.info.call_args_list)
            assert len(log_calls) > 0, "Should log the error"
            
            # Property: Error statistics should be updated
            stats = error_handler.get_error_statistics()
            assert stats['total_errors'] > 0
            assert category.value in stats.get('category_counts', {}) or stats['total_errors'] == 1
    
    @given(
        memory_threshold=st.floats(min_value=0.1, max_value=0.99),
        current_batch_size=st.integers(min_value=1, max_value=64)
    )
    @settings(max_examples=10, deadline=5000)
    def test_memory_recovery_strategy_property(self, temp_config, memory_threshold, current_batch_size):
        """
        Property: Memory recovery strategy should handle all memory scenarios consistently.
        """
        strategy = MemoryRecoveryStrategy()
        strategy.memory_threshold = memory_threshold
        
        # Create memory error context
        error_context = ErrorContext(
            error_type="RuntimeError",
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            message="CUDA out of memory",
            details={},
            timestamp=0.0,
            recovery_suggestions=[]
        )
        
        # Property: Should be able to handle memory errors
        assert strategy.can_handle(error_context)
        
        # Mock GPU availability and memory stats
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache'), \
             patch('torch.cuda.memory_allocated', return_value=int(1024**3 * 8)), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_device = Mock()
            mock_device.total_memory = int(1024**3 * 10)  # 10GB total
            mock_props.return_value = mock_device
            
            success, message = strategy.recover(
                error_context, 
                current_batch_size=current_batch_size
            )
            
            # Property: Recovery should always succeed (at minimum, cache clearing)
            assert success is True
            assert isinstance(message, str)
            assert len(message) > 0
            
            # Property: Should suggest batch size reduction if memory usage high
            gpu_utilization = 8 / 10  # 80% from our mock
            if gpu_utilization > memory_threshold:
                assert 'suggested_batch_size' in error_context.details
                suggested_size = error_context.details['suggested_batch_size']
                assert suggested_size <= current_batch_size
                assert suggested_size >= strategy.min_batch_size
    
    @given(
        dataset_type=st.sampled_from(['sft', 'preference']),
        original_dataset=st.text(min_size=5, max_size=30).filter(lambda x: '/' in x)
    )
    @settings(max_examples=10, deadline=10000)
    def test_dataset_recovery_strategy_property(self, temp_config, dataset_type, original_dataset):
        """
        Property: Dataset recovery strategy should handle all dataset failure scenarios.
        """
        assume(original_dataset.count('/') == 1)
        assume(not original_dataset.startswith('/') and not original_dataset.endswith('/'))
        
        strategy = DatasetRecoveryStrategy(temp_config)
        
        # Create dataset error context
        error_context = ErrorContext(
            error_type="ConnectionError",
            category=ErrorCategory.DATASET,
            severity=ErrorSeverity.MEDIUM,
            message="Failed to connect to HuggingFace Hub",
            details={},
            timestamp=0.0,
            recovery_suggestions=[]
        )
        
        # Property: Should be able to handle dataset errors
        assert strategy.can_handle(error_context)
        
        # Mock successful alternative dataset loading
        with patch('rlhf_phi3.utils.error_handler.load_dataset') as mock_load:
            # Make first alternative succeed
            alternatives = strategy.alternative_datasets.get(dataset_type, [])
            if alternatives:
                mock_dataset = Mock()
                mock_dataset.__len__ = Mock(return_value=10)
                
                def load_side_effect(dataset_name, **kwargs):
                    if dataset_name == alternatives[0]:
                        return mock_dataset
                    else:
                        raise Exception("Dataset not available")
                
                mock_load.side_effect = load_side_effect
                
                success, message = strategy.recover(
                    error_context,
                    dataset_type=dataset_type,
                    dataset_name=original_dataset
                )
                
                # Property: Should succeed when alternative is available
                assert success is True
                assert isinstance(message, str)
                assert 'alternative dataset' in message.lower()
                
                # Property: Should update error context with alternative info
                assert 'alternative_dataset' in error_context.details
                assert error_context.details['alternative_dataset'] == alternatives[0]
    
    def test_error_statistics_consistency_property(self, error_handler):
        """
        Property: Error statistics should remain consistent across operations.
        """
        initial_stats = error_handler.get_error_statistics()
        
        # Handle multiple errors
        test_errors = [
            (Exception("Memory error"), ErrorCategory.MEMORY, ErrorSeverity.HIGH),
            (Exception("Dataset error"), ErrorCategory.DATASET, ErrorSeverity.MEDIUM),
            (Exception("Auth error"), ErrorCategory.AUTHENTICATION, ErrorSeverity.LOW),
        ]
        
        for error, category, severity in test_errors:
            error_handler.handle_error(error, category, severity, auto_recover=False)
        
        final_stats = error_handler.get_error_statistics()
        
        # Property: Total errors should increase by number of handled errors
        assert final_stats['total_errors'] == initial_stats['total_errors'] + len(test_errors)
        
        # Property: Category counts should be accurate
        expected_categories = {cat.value for _, cat, _ in test_errors}
        for category in expected_categories:
            assert category in final_stats['category_counts']
            assert final_stats['category_counts'][category] > 0
        
        # Property: Recovery rate should be calculable
        assert 'recovery_rate' in final_stats
        assert 0.0 <= final_stats['recovery_rate'] <= 1.0
    
    def test_error_context_manager_property(self, error_handler):
        """
        Property: Error context manager should handle all exception types consistently.
        """
        test_exceptions = [
            RuntimeError("Runtime error"),
            ValueError("Value error"),
            KeyError("Key error"),
            TypeError("Type error"),
        ]
        
        for exception in test_exceptions:
            initial_count = len(error_handler.error_history)
            
            # Test that context manager catches and handles the exception
            with pytest.raises(type(exception)):
                with error_handler.error_context(
                    category=ErrorCategory.TRAINING,
                    severity=ErrorSeverity.MEDIUM,
                    auto_recover=False
                ):
                    raise exception
            
            # Property: Error should be added to history
            assert len(error_handler.error_history) == initial_count + 1
            
            # Property: Latest error should match the raised exception
            latest_error = error_handler.error_history[-1]
            assert latest_error.error_type == type(exception).__name__
            assert latest_error.message == str(exception)


class ErrorHandlingStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for error handling system.
    
    This tests that the error handling system maintains consistency
    across sequences of operations and state changes.
    """
    
    def __init__(self):
        super().__init__()
        self.temp_dir = None
        self.config = None
        self.error_handler = None
        self.handled_errors = []
    
    @initialize()
    def setup(self):
        """Initialize the error handling system."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.paths.cache_dir = self.temp_dir
        self.config.paths.base_output_dir = self.temp_dir
        self.config.paths.logs_dir = self.temp_dir
        
        self.error_handler = ErrorHandler(self.config)
        self.handled_errors = []
    
    @rule(
        error_message=st.text(min_size=1, max_size=100),
        category=st.sampled_from(list(ErrorCategory)),
        severity=st.sampled_from(list(ErrorSeverity))
    )
    def handle_error(self, error_message, category, severity):
        """Handle an error and track it."""
        error = Exception(error_message)
        error_context = self.error_handler.handle_error(
            error, category, severity, auto_recover=False
        )
        self.handled_errors.append(error_context)
    
    @rule()
    def get_statistics(self):
        """Get error statistics and verify consistency."""
        stats = self.error_handler.get_error_statistics()
        
        # Verify statistics match our tracking
        assert stats['total_errors'] == len(self.handled_errors)
        assert len(self.error_handler.error_history) == len(self.handled_errors)
    
    @rule()
    def clear_history(self):
        """Clear error history and verify state."""
        self.error_handler.clear_error_history()
        self.handled_errors.clear()
        
        stats = self.error_handler.get_error_statistics()
        assert stats['total_errors'] == 0
        assert len(self.error_handler.error_history) == 0
    
    @invariant()
    def error_history_consistency(self):
        """Error history should always be consistent with our tracking."""
        assert len(self.error_handler.error_history) == len(self.handled_errors)
        
        # Verify each error in history matches our tracking
        for i, (tracked, stored) in enumerate(zip(self.handled_errors, self.error_handler.error_history)):
            assert tracked.error_type == stored.error_type
            assert tracked.message == stored.message
            assert tracked.category == stored.category
            assert tracked.severity == stored.severity
    
    def teardown(self):
        """Clean up temporary directory."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# Test the stateful machine
TestErrorHandlingStateMachine = ErrorHandlingStateMachine.TestCase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])