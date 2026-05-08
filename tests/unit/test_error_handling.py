"""
Unit Tests for Error Handling and Recovery Systems

This module contains unit tests for specific error scenarios and recovery mechanisms.
Tests cover memory exhaustion recovery, dataset loading fallbacks, and authentication
failure handling.

Requirements tested:
- 9.1: GPU memory exhaustion recovery with automatic batch size reduction
- 9.2: Dataset loading fallbacks with alternative sources and clear guidance
- 9.3: Google Drive authentication failure recovery with local storage fallback
- 9.5: Comprehensive error handling with detailed logs and recovery instructions
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import torch
import gc

from rlhf_phi3.config.config_manager import Config
from rlhf_phi3.utils.error_handler import (
    ErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext,
    MemoryRecoveryStrategy, DatasetRecoveryStrategy, 
    AuthenticationRecoveryStrategy, TrainingRecoveryStrategy,
    handle_memory_error, safe_dataset_load, create_fallback_storage
)
from rlhf_phi3.data.dataset_manager import DatasetManager
from rlhf_phi3.checkpoints.checkpoint_manager import CheckpointManager


class TestMemoryExhaustionRecovery:
    """
    Test GPU memory exhaustion recovery mechanisms.
    
    Requirement 9.1: GPU memory exhaustion recovery with automatic batch size reduction
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
    def memory_strategy(self):
        """Create memory recovery strategy."""
        return MemoryRecoveryStrategy()
    
    def test_memory_error_detection(self, memory_strategy):
        """Test that memory errors are correctly detected."""
        # Test various memory-related error messages
        memory_errors = [
            ErrorContext("RuntimeError", ErrorCategory.MEMORY, ErrorSeverity.HIGH, 
                        "CUDA out of memory", {}, 0.0, []),
            ErrorContext("RuntimeError", ErrorCategory.TRAINING, ErrorSeverity.HIGH, 
                        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", {}, 0.0, []),
            ErrorContext("Exception", ErrorCategory.MODEL, ErrorSeverity.MEDIUM, 
                        "GPU memory exhausted", {}, 0.0, []),
        ]
        
        for error_context in memory_errors:
            assert memory_strategy.can_handle(error_context), f"Should detect memory error: {error_context.message}"
    
    def test_memory_recovery_success(self, memory_strategy):
        """Test successful memory recovery."""
        error_context = ErrorContext(
            "RuntimeError", ErrorCategory.MEMORY, ErrorSeverity.HIGH,
            "CUDA out of memory", {}, 0.0, []
        )
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('gc.collect', return_value=42) as mock_gc, \
             patch('torch.cuda.memory_allocated', return_value=8 * 1024**3), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            # Mock GPU properties
            mock_device = Mock()
            mock_device.total_memory = 10 * 1024**3  # 10GB
            mock_props.return_value = mock_device
            
            success, message = memory_strategy.recover(error_context, current_batch_size=8)
            
            # Verify recovery actions
            assert success is True
            assert "Memory recovery completed" in message
            mock_empty_cache.assert_called_once()
            mock_gc.assert_called_once()
            
            # Verify batch size suggestion (80% memory usage > 90% threshold)
            assert 'suggested_batch_size' in error_context.details
            assert error_context.details['suggested_batch_size'] == 4  # Half of original
    
    def test_memory_recovery_no_gpu(self, memory_strategy):
        """Test memory recovery when GPU is not available."""
        error_context = ErrorContext(
            "RuntimeError", ErrorCategory.MEMORY, ErrorSeverity.HIGH,
            "Out of memory", {}, 0.0, []
        )
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('gc.collect', return_value=10) as mock_gc:
            
            success, message = memory_strategy.recover(error_context)
            
            assert success is True
            assert "Memory recovery completed" in message
            mock_gc.assert_called_once()
    
    def test_memory_decorator(self):
        """Test memory error handling decorator."""
        call_count = 0
        
        @handle_memory_error
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("CUDA out of memory")
            return "success"
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache'), \
             patch('gc.collect'):
            
            result = failing_function()
            
            assert result == "success"
            assert call_count == 2  # Failed once, then succeeded
    
    def test_memory_decorator_non_memory_error(self):
        """Test that decorator doesn't interfere with non-memory errors."""
        @handle_memory_error
        def failing_function():
            raise ValueError("Not a memory error")
        
        with pytest.raises(ValueError, match="Not a memory error"):
            failing_function()


class TestDatasetLoadingFallbacks:
    """
    Test dataset loading fallback mechanisms.
    
    Requirement 9.2: Dataset loading fallbacks with alternative sources and clear guidance
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
    def dataset_strategy(self, temp_config):
        """Create dataset recovery strategy."""
        return DatasetRecoveryStrategy(temp_config)
    
    @pytest.fixture
    def dataset_manager(self, temp_config):
        """Create dataset manager with mocked tokenizer."""
        with patch('rlhf_phi3.data.dataset_manager.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_tokenizer.from_pretrained.return_value.pad_token = "[PAD]"
            mock_tokenizer.from_pretrained.return_value.eos_token = "[EOS]"
            
            manager = DatasetManager(temp_config)
            manager.tokenizer = mock_tokenizer.from_pretrained.return_value
            yield manager
    
    def test_dataset_error_detection(self, dataset_strategy):
        """Test that dataset errors are correctly detected."""
        dataset_errors = [
            ErrorContext("ConnectionError", ErrorCategory.DATASET, ErrorSeverity.MEDIUM,
                        "Failed to connect to HuggingFace Hub", {}, 0.0, []),
            ErrorContext("HTTPError", ErrorCategory.NETWORK, ErrorSeverity.HIGH,
                        "Dataset not found", {}, 0.0, []),
            ErrorContext("Exception", ErrorCategory.TRAINING, ErrorSeverity.LOW,
                        "huggingface_hub error", {}, 0.0, []),
        ]
        
        for error_context in dataset_errors:
            assert dataset_strategy.can_handle(error_context), f"Should detect dataset error: {error_context.message}"
    
    def test_successful_dataset_fallback(self, dataset_strategy):
        """Test successful fallback to alternative dataset."""
        error_context = ErrorContext(
            "ConnectionError", ErrorCategory.DATASET, ErrorSeverity.MEDIUM,
            "Dataset loading failed", {}, 0.0, []
        )
        
        # Mock successful loading of first alternative
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        with patch('rlhf_phi3.utils.error_handler.load_dataset') as mock_load:
            mock_load.return_value = mock_dataset
            
            success, message = dataset_strategy.recover(
                error_context,
                dataset_type='sft',
                dataset_name='failed/dataset'
            )
            
            assert success is True
            assert "Dataset recovery successful" in message
            assert "alternative dataset" in message
            assert 'alternative_dataset' in error_context.details
    
    def test_all_datasets_fail(self, dataset_strategy):
        """Test when all alternative datasets fail."""
        error_context = ErrorContext(
            "ConnectionError", ErrorCategory.DATASET, ErrorSeverity.MEDIUM,
            "Dataset loading failed", {}, 0.0, []
        )
        
        with patch('rlhf_phi3.utils.error_handler.load_dataset') as mock_load:
            mock_load.side_effect = Exception("All datasets failed")
            
            success, message = dataset_strategy.recover(
                error_context,
                dataset_type='sft',
                dataset_name='failed/dataset'
            )
            
            assert success is False
            assert "All alternative datasets failed" in message
            assert "check your internet connection" in message.lower()
    
    def test_dataset_manager_fallback_integration(self, dataset_manager):
        """Test dataset manager fallback integration."""
        with patch('rlhf_phi3.data.dataset_manager.load_dataset') as mock_load:
            # First call (primary) fails, second call (alternative) succeeds
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=50)
            
            call_count = 0
            def load_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Primary dataset failed")
                return mock_dataset
            
            mock_load.side_effect = load_side_effect
            
            result = dataset_manager.load_dataset_with_fallback(
                dataset_name="primary/dataset",
                dataset_type="sft",
                max_retries=1
            )
            
            assert result is not None
            assert result == mock_dataset
            assert mock_load.call_count >= 2  # Primary + at least one alternative
    
    def test_safe_dataset_load_utility(self):
        """Test safe dataset loading utility function."""
        alternatives = ["alt1/dataset", "alt2/dataset"]
        
        with patch('rlhf_phi3.utils.error_handler.load_dataset') as mock_load:
            # First two fail, third succeeds
            mock_dataset = Mock()
            call_count = 0
            
            def load_side_effect(dataset_name, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception(f"Dataset {dataset_name} failed")
                return mock_dataset
            
            mock_load.side_effect = load_side_effect
            
            result = safe_dataset_load(
                "primary/dataset",
                alternatives=alternatives
            )
            
            assert result == mock_dataset
            assert mock_load.call_count == 3
    
    def test_dataset_health_validation(self, dataset_manager):
        """Test dataset health validation functionality."""
        # Mock SFT dataset
        mock_sft_dataset = Mock()
        mock_sft_dataset.column_names = ['messages']
        mock_sft_dataset.features = {'messages': 'list'}
        mock_sft_dataset.__len__ = Mock(return_value=100)
        mock_sft_dataset.__iter__ = Mock(return_value=iter([
            {'messages': [{'role': 'user', 'content': 'test'}]}
        ]))
        mock_sft_dataset.__getitem__ = Mock(return_value={'messages': [{'role': 'user', 'content': 'test'}]})
        
        health_report = dataset_manager.validate_dataset_health(mock_sft_dataset)
        
        assert health_report['dataset_type'] == 'Mock'
        assert health_report['sample_count'] == 100
        assert 'messages' in health_report['column_names']
        assert isinstance(health_report['issues'], list)
        assert isinstance(health_report['recommendations'], list)
    
    def test_cache_cleanup(self, dataset_manager):
        """Test corrupted cache cleanup functionality."""
        # Create some fake cache files
        cache_dir = Path(dataset_manager.cache_dir)
        test_files = [
            cache_dir / "test_dataset_cache.json",
            cache_dir / "another_dataset" / "data.arrow",
        ]
        
        for file_path in test_files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("fake cache data")
        
        # Test cleanup
        cleanup_results = dataset_manager.clear_corrupted_cache()
        
        assert cleanup_results['files_removed'] >= len(test_files)
        assert cleanup_results['space_freed_mb'] > 0
        assert isinstance(cleanup_results['errors'], list)


class TestAuthenticationFailureHandling:
    """
    Test Google Drive authentication failure handling.
    
    Requirement 9.3: Google Drive authentication failure recovery with local storage fallback
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
    def auth_strategy(self, temp_config):
        """Create authentication recovery strategy."""
        return AuthenticationRecoveryStrategy(temp_config)
    
    def test_auth_error_detection(self, auth_strategy):
        """Test that authentication errors are correctly detected."""
        auth_errors = [
            ErrorContext("AuthError", ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH,
                        "Google Drive authentication failed", {}, 0.0, []),
            ErrorContext("Exception", ErrorCategory.STORAGE, ErrorSeverity.MEDIUM,
                        "Invalid credentials", {}, 0.0, []),
            ErrorContext("ConnectionError", ErrorCategory.NETWORK, ErrorSeverity.LOW,
                        "Google API error", {}, 0.0, []),
        ]
        
        for error_context in auth_errors:
            assert auth_strategy.can_handle(error_context), f"Should detect auth error: {error_context.message}"
    
    def test_successful_auth_recovery(self, auth_strategy, temp_config):
        """Test successful authentication recovery with local fallback."""
        error_context = ErrorContext(
            "AuthError", ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH,
            "Google Drive authentication failed", {}, 0.0, []
        )
        
        success, message = auth_strategy.recover(error_context)
        
        assert success is True
        assert "Falling back to local storage" in message
        assert "WARNING" in message
        assert 'fallback_path' in error_context.details
        assert 'drive_sync_disabled' in error_context.details
        
        # Verify fallback directory was created
        fallback_path = Path(error_context.details['fallback_path'])
        assert fallback_path.exists()
    
    def test_checkpoint_manager_auth_failure(self, temp_config):
        """Test checkpoint manager handling of authentication failures."""
        # Mock Google Drive dependencies as unavailable
        with patch('rlhf_phi3.checkpoints.checkpoint_manager.GOOGLE_DRIVE_AVAILABLE', False):
            checkpoint_manager = CheckpointManager(
                base_path=temp_config.paths.base_output_dir,
                enable_drive_sync=True
            )
            
            # Should fall back to local storage
            assert checkpoint_manager.enable_drive_sync is False
            assert checkpoint_manager.drive_manager is None
    
    def test_checkpoint_manager_drive_failure_handling(self, temp_config):
        """Test checkpoint manager drive failure handling method."""
        checkpoint_manager = CheckpointManager(
            base_path=temp_config.paths.base_output_dir,
            enable_drive_sync=False
        )
        
        # Test fallback to local
        success, message = checkpoint_manager.handle_drive_failure(fallback_to_local=True)
        
        assert success is True
        assert "local storage fallback" in message.lower()
        assert checkpoint_manager.enable_drive_sync is False
        
        # Test without fallback
        success, message = checkpoint_manager.handle_drive_failure(fallback_to_local=False)
        
        assert success is False
        assert "authentication failed" in message.lower()
    
    def test_create_fallback_storage_utility(self, temp_config):
        """Test fallback storage creation utility."""
        base_path = temp_config.paths.base_output_dir
        
        fallback_path = create_fallback_storage(base_path, "test_fallback")
        
        assert fallback_path.exists()
        assert fallback_path.is_dir()
        assert fallback_path.name == "test_fallback"
        assert fallback_path.parent == Path(base_path)


class TestTrainingErrorRecovery:
    """
    Test training-related error recovery mechanisms.
    
    Requirement 9.5: Comprehensive error handling with detailed logs and recovery instructions
    """
    
    @pytest.fixture
    def training_strategy(self):
        """Create training recovery strategy."""
        return TrainingRecoveryStrategy()
    
    def test_training_error_detection(self, training_strategy):
        """Test that training errors are correctly detected."""
        training_errors = [
            ErrorContext("RuntimeError", ErrorCategory.TRAINING, ErrorSeverity.HIGH,
                        "Loss diverged to NaN", {}, 0.0, []),
            ErrorContext("ValueError", ErrorCategory.MODEL, ErrorSeverity.MEDIUM,
                        "Loss contains inf values", {}, 0.0, []),
            ErrorContext("Exception", ErrorCategory.TRAINING, ErrorSeverity.LOW,
                        "Training loss divergence detected", {}, 0.0, []),
        ]
        
        for error_context in training_errors:
            assert training_strategy.can_handle(error_context), f"Should detect training error: {error_context.message}"
    
    def test_loss_divergence_recovery(self, training_strategy):
        """Test recovery from loss divergence."""
        error_context = ErrorContext(
            "RuntimeError", ErrorCategory.TRAINING, ErrorSeverity.HIGH,
            "Loss diverged", {}, 0.0, []
        )
        
        success, message = training_strategy.recover(
            error_context,
            current_loss=100.0,
            previous_loss=1.0,
            learning_rate=1e-3
        )
        
        assert success is True
        assert "Training recovery analysis complete" in message
        assert 'suggested_learning_rate' in error_context.details
        
        # Should suggest reduced learning rate
        suggested_lr = error_context.details['suggested_learning_rate']
        assert suggested_lr == 1e-3 * training_strategy.learning_rate_reduction_factor
    
    def test_nan_loss_recovery(self, training_strategy):
        """Test recovery from NaN loss values."""
        error_context = ErrorContext(
            "ValueError", ErrorCategory.TRAINING, ErrorSeverity.HIGH,
            "Loss contains NaN", {}, 0.0, []
        )
        
        success, message = training_strategy.recover(
            error_context,
            current_loss=float('nan'),
            previous_loss=1.0,
            learning_rate=1e-4
        )
        
        assert success is True
        assert len(error_context.recovery_suggestions) > 0
        
        # Should include gradient clipping suggestion
        suggestions = error_context.recovery_suggestions
        gradient_clipping_suggested = any("gradient clipping" in s.lower() for s in suggestions)
        assert gradient_clipping_suggested


class TestComprehensiveErrorHandling:
    """
    Test comprehensive error handling system integration.
    
    Requirement 9.5: Comprehensive error handling with detailed logs and recovery instructions
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
    
    def test_error_handler_initialization(self, error_handler):
        """Test error handler proper initialization."""
        assert len(error_handler.recovery_strategies) == 4  # Memory, Dataset, Auth, Training
        assert error_handler.total_errors == 0
        assert error_handler.recovered_errors == 0
        assert len(error_handler.error_history) == 0
    
    def test_error_handling_workflow(self, error_handler):
        """Test complete error handling workflow."""
        test_error = RuntimeError("CUDA out of memory")
        
        with patch('rlhf_phi3.utils.error_handler.logger') as mock_logger:
            error_context = error_handler.handle_error(
                error=test_error,
                category=ErrorCategory.MEMORY,
                severity=ErrorSeverity.HIGH,
                auto_recover=True
            )
            
            # Verify error context
            assert error_context.error_type == "RuntimeError"
            assert error_context.category == ErrorCategory.MEMORY
            assert error_context.severity == ErrorSeverity.HIGH
            assert error_context.auto_recovery_attempted is True
            
            # Verify logging occurred
            assert mock_logger.error.called or mock_logger.warning.called or mock_logger.critical.called
            
            # Verify recovery instructions generated
            assert 'recovery_instructions' in error_context.details
            instructions = error_context.details['recovery_instructions']
            assert "Recovery Instructions:" in instructions
    
    def test_error_statistics_tracking(self, error_handler):
        """Test error statistics tracking."""
        # Handle multiple errors
        errors = [
            (RuntimeError("Memory error"), ErrorCategory.MEMORY, ErrorSeverity.HIGH),
            (ConnectionError("Network error"), ErrorCategory.DATASET, ErrorSeverity.MEDIUM),
            (ValueError("Auth error"), ErrorCategory.AUTHENTICATION, ErrorSeverity.LOW),
        ]
        
        for error, category, severity in errors:
            error_handler.handle_error(error, category, severity, auto_recover=False)
        
        stats = error_handler.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['recovery_rate'] == 0.0  # No auto-recovery attempted
        assert len(stats['category_counts']) == 3
        assert stats['category_counts']['memory'] == 1
        assert stats['category_counts']['dataset'] == 1
        assert stats['category_counts']['authentication'] == 1
    
    def test_error_context_manager(self, error_handler):
        """Test error context manager functionality."""
        with patch('rlhf_phi3.utils.error_handler.logger'):
            # Test successful execution (no error)
            with error_handler.error_context(ErrorCategory.TRAINING):
                result = "success"
            
            assert result == "success"
            assert len(error_handler.error_history) == 0
            
            # Test error handling
            with pytest.raises(ValueError):
                with error_handler.error_context(
                    ErrorCategory.TRAINING, 
                    ErrorSeverity.CRITICAL,
                    auto_recover=False
                ):
                    raise ValueError("Test error")
            
            assert len(error_handler.error_history) == 1
    
    def test_error_report_export(self, error_handler, temp_config):
        """Test error report export functionality."""
        # Generate some errors
        error_handler.handle_error(
            RuntimeError("Test error 1"), 
            ErrorCategory.MEMORY, 
            ErrorSeverity.HIGH
        )
        error_handler.handle_error(
            ValueError("Test error 2"), 
            ErrorCategory.TRAINING, 
            ErrorSeverity.MEDIUM
        )
        
        # Export report
        report_path = Path(temp_config.paths.logs_dir) / "error_report.txt"
        error_handler.export_error_report(report_path)
        
        assert report_path.exists()
        
        # Verify report content
        report_content = report_path.read_text()
        assert "RLHF Phi-3 Pipeline Error Report" in report_content
        assert "Total Errors: 2" in report_content
        assert "RuntimeError" in report_content
        assert "ValueError" in report_content
    
    def test_error_history_management(self, error_handler):
        """Test error history management."""
        # Add some errors
        for i in range(5):
            error_handler.handle_error(
                Exception(f"Error {i}"),
                ErrorCategory.TRAINING,
                ErrorSeverity.LOW
            )
        
        assert len(error_handler.error_history) == 5
        assert error_handler.total_errors == 5
        
        # Clear history
        error_handler.clear_error_history()
        
        assert len(error_handler.error_history) == 0
        assert error_handler.total_errors == 0
        assert error_handler.recovered_errors == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])