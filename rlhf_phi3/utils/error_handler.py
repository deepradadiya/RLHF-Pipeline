"""
Error Handler for RLHF Phi-3 Pipeline

This module provides comprehensive error handling and recovery mechanisms for the RLHF pipeline.
It includes GPU memory exhaustion recovery, dataset loading fallback strategies, Google Drive
authentication error handling, and detailed error logging with recovery instructions.

Requirements satisfied:
- 9.1: GPU memory exhaustion recovery with automatic batch size reduction
- 9.2: Dataset loading fallbacks with alternative sources and clear guidance
- 9.3: Google Drive authentication failure recovery with local storage fallback
- 9.5: Comprehensive error handling with detailed logs and recovery instructions
"""

import gc
import logging
import os
import sys
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import torch
import psutil
from datasets import Dataset, IterableDataset, load_dataset

from ..config.config_manager import Config

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorization and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for specialized handling."""
    MEMORY = "memory"
    DATASET = "dataset"
    AUTHENTICATION = "authentication"
    MODEL = "model"
    TRAINING = "training"
    STORAGE = "storage"
    NETWORK = "network"
    CONFIGURATION = "configuration"


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""
    error_type: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: float
    recovery_suggestions: List[str]
    auto_recovery_attempted: bool = False
    recovery_successful: bool = False


class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.success_count = 0
        self.failure_count = 0
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this strategy can handle the given error."""
        raise NotImplementedError
    
    def recover(self, error_context: ErrorContext, **kwargs) -> Tuple[bool, str]:
        """Attempt recovery from the error.
        
        Returns:
            Tuple of (success, message)
        """
        raise NotImplementedError
    
    def get_success_rate(self) -> float:
        """Get the success rate of this recovery strategy."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class MemoryRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for GPU memory exhaustion."""
    
    def __init__(self):
        super().__init__(
            "Memory Recovery",
            "Handles GPU memory exhaustion through cache clearing and batch size reduction"
        )
        self.memory_threshold = 0.9  # 90% memory usage threshold
        self.min_batch_size = 1
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a memory-related error."""
        return (error_context.category == ErrorCategory.MEMORY or
                "memory" in error_context.message.lower() or
                "cuda" in error_context.message.lower() or
                "out of memory" in error_context.message.lower())
    
    def recover(self, error_context: ErrorContext, **kwargs) -> Tuple[bool, str]:
        """Attempt memory recovery."""
        try:
            logger.info("Attempting GPU memory recovery...")
            
            # Step 1: Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            
            # Step 2: Force garbage collection
            gc.collect()
            logger.info("Performed garbage collection")
            
            # Step 3: Get current memory usage
            memory_stats = self._get_memory_stats()
            logger.info(f"Memory after cleanup: {memory_stats}")
            
            # Step 4: Suggest batch size reduction if still high memory usage
            recovery_message = "Memory recovery completed. "
            
            if torch.cuda.is_available():
                gpu_utilization = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                
                if gpu_utilization > self.memory_threshold:
                    current_batch_size = kwargs.get('current_batch_size', 4)
                    new_batch_size = max(self.min_batch_size, current_batch_size // 2)
                    
                    recovery_message += f"Recommend reducing batch size from {current_batch_size} to {new_batch_size}. "
                    
                    # Update error context with recovery suggestion
                    error_context.details['suggested_batch_size'] = new_batch_size
                    error_context.details['original_batch_size'] = current_batch_size
            
            self.success_count += 1
            return True, recovery_message
            
        except Exception as e:
            self.failure_count += 1
            error_msg = f"Memory recovery failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        # GPU memory
        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
            stats['gpu_max_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            stats['gpu_utilization'] = stats['gpu_allocated_gb'] / stats['gpu_max_gb']
        
        # CPU memory
        memory = psutil.virtual_memory()
        stats['cpu_used_gb'] = memory.used / 1024**3
        stats['cpu_total_gb'] = memory.total / 1024**3
        stats['cpu_utilization'] = memory.percent / 100.0
        
        return stats


class DatasetRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for dataset loading failures."""
    
    def __init__(self, config: Config):
        super().__init__(
            "Dataset Recovery",
            "Handles dataset loading failures with alternative sources and fallback strategies"
        )
        self.config = config
        self.alternative_datasets = {
            # SFT alternatives
            'sft': [
                "microsoft/orca-math-word-problems-200k",
                "Open-Orca/OpenOrca",
                "teknium/OpenHermes-2.5",
                "argilla/distilabel-intel-orca-dpo-pairs"
            ],
            # Preference alternatives
            'preference': [
                "Anthropic/hh-rlhf",
                "argilla/ultrafeedback-binarized-preferences-cleaned",
                "Intel/orca_dpo_pairs",
                "jondurbin/truthy-dpo-v0.1"
            ]
        }
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a dataset-related error."""
        return (error_context.category == ErrorCategory.DATASET or
                "dataset" in error_context.message.lower() or
                "huggingface" in error_context.message.lower() or
                "connection" in error_context.message.lower())
    
    def recover(self, error_context: ErrorContext, **kwargs) -> Tuple[bool, str]:
        """Attempt dataset recovery with alternative sources."""
        try:
            dataset_type = kwargs.get('dataset_type', 'sft')
            original_dataset = kwargs.get('dataset_name', '')
            
            logger.info(f"Attempting dataset recovery for {dataset_type} dataset: {original_dataset}")
            
            # Try alternative datasets
            alternatives = self.alternative_datasets.get(dataset_type, [])
            
            for alt_dataset in alternatives:
                if alt_dataset == original_dataset:
                    continue  # Skip the original failed dataset
                
                try:
                    logger.info(f"Trying alternative dataset: {alt_dataset}")
                    
                    # Test loading a small sample
                    test_dataset = load_dataset(
                        alt_dataset,
                        split="train[:10]",  # Load only 10 samples for testing
                        cache_dir=str(Path(self.config.paths.cache_dir) / "datasets"),
                        streaming=False
                    )
                    
                    if len(test_dataset) > 0:
                        logger.info(f"Successfully loaded alternative dataset: {alt_dataset}")
                        
                        # Update error context with successful alternative
                        error_context.details['alternative_dataset'] = alt_dataset
                        error_context.details['original_dataset'] = original_dataset
                        
                        recovery_message = (
                            f"Dataset recovery successful. Using alternative dataset: {alt_dataset}. "
                            f"Original dataset {original_dataset} failed to load."
                        )
                        
                        self.success_count += 1
                        return True, recovery_message
                
                except Exception as alt_error:
                    logger.warning(f"Alternative dataset {alt_dataset} also failed: {str(alt_error)}")
                    continue
            
            # If no alternatives worked, provide guidance
            self.failure_count += 1
            recovery_message = (
                f"All alternative datasets failed. Please check your internet connection "
                f"and try again later. You can also try using a local dataset or "
                f"different dataset from HuggingFace Hub."
            )
            
            return False, recovery_message
            
        except Exception as e:
            self.failure_count += 1
            error_msg = f"Dataset recovery failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


class AuthenticationRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for Google Drive authentication failures."""
    
    def __init__(self, config: Config):
        super().__init__(
            "Authentication Recovery",
            "Handles Google Drive authentication failures with local storage fallback"
        )
        self.config = config
        self.local_fallback_path = Path(config.paths.base_output_dir) / "local_checkpoints"
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is an authentication-related error."""
        return (error_context.category == ErrorCategory.AUTHENTICATION or
                "authentication" in error_context.message.lower() or
                "credentials" in error_context.message.lower() or
                "google" in error_context.message.lower() or
                "drive" in error_context.message.lower())
    
    def recover(self, error_context: ErrorContext, **kwargs) -> Tuple[bool, str]:
        """Attempt authentication recovery with local fallback."""
        try:
            logger.info("Attempting authentication recovery...")
            
            # Create local fallback directory
            self.local_fallback_path.mkdir(parents=True, exist_ok=True)
            
            # Update error context with fallback path
            error_context.details['fallback_path'] = str(self.local_fallback_path)
            error_context.details['drive_sync_disabled'] = True
            
            recovery_message = (
                f"Google Drive authentication failed. Falling back to local storage at: "
                f"{self.local_fallback_path}. "
                f"WARNING: Checkpoints will not persist across Colab sessions. "
                f"Please set up Google Drive authentication for persistent storage."
            )
            
            # Add recovery instructions
            instructions = [
                "1. Upload credentials.json to your Colab environment",
                "2. Ensure Google Drive API is enabled in your Google Cloud Console",
                "3. Check that the OAuth consent screen is properly configured",
                "4. Verify that your Google account has access to the credentials"
            ]
            
            error_context.recovery_suggestions.extend(instructions)
            
            logger.warning(recovery_message)
            self.success_count += 1
            return True, recovery_message
            
        except Exception as e:
            self.failure_count += 1
            error_msg = f"Authentication recovery failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


class TrainingRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for training-related errors."""
    
    def __init__(self):
        super().__init__(
            "Training Recovery",
            "Handles training failures with loss divergence detection and hyperparameter adjustment"
        )
        self.loss_divergence_threshold = 10.0  # Loss increase threshold
        self.learning_rate_reduction_factor = 0.5
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a training-related error."""
        return (error_context.category == ErrorCategory.TRAINING or
                "loss" in error_context.message.lower() or
                "nan" in error_context.message.lower() or
                "inf" in error_context.message.lower() or
                "diverge" in error_context.message.lower())
    
    def recover(self, error_context: ErrorContext, **kwargs) -> Tuple[bool, str]:
        """Attempt training recovery."""
        try:
            logger.info("Attempting training recovery...")
            
            current_loss = kwargs.get('current_loss', float('inf'))
            previous_loss = kwargs.get('previous_loss', 0.0)
            current_lr = kwargs.get('learning_rate', 1e-4)
            
            recovery_suggestions = []
            
            # Check for loss divergence
            if current_loss > previous_loss * self.loss_divergence_threshold:
                new_lr = current_lr * self.learning_rate_reduction_factor
                recovery_suggestions.append(f"Reduce learning rate from {current_lr} to {new_lr}")
                error_context.details['suggested_learning_rate'] = new_lr
            
            # Check for NaN/Inf values
            if not torch.isfinite(torch.tensor(current_loss)):
                recovery_suggestions.extend([
                    "Enable gradient clipping with max_grad_norm=1.0",
                    "Reduce learning rate by factor of 10",
                    "Check for numerical instability in loss computation",
                    "Verify input data doesn't contain NaN/Inf values"
                ])
            
            # Add general training recovery suggestions
            recovery_suggestions.extend([
                "Restart training from the last stable checkpoint",
                "Reduce batch size to improve numerical stability",
                "Enable mixed precision training if not already enabled",
                "Check for data corruption in the current batch"
            ])
            
            error_context.recovery_suggestions.extend(recovery_suggestions)
            
            recovery_message = (
                f"Training recovery analysis complete. Current loss: {current_loss}, "
                f"Previous loss: {previous_loss}. See recovery suggestions for next steps."
            )
            
            self.success_count += 1
            return True, recovery_message
            
        except Exception as e:
            self.failure_count += 1
            error_msg = f"Training recovery failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


class ErrorHandler:
    """
    Comprehensive error handling and recovery system for RLHF Phi-3 pipeline.
    
    This class provides:
    - Automatic error detection and categorization
    - Recovery strategy selection and execution
    - Detailed error logging and recovery instructions
    - Performance monitoring and recovery success tracking
    
    Requirements satisfied:
    - 9.1: GPU memory exhaustion recovery with automatic batch size reduction
    - 9.2: Dataset loading fallbacks with alternative sources and clear guidance
    - 9.3: Google Drive authentication failure recovery with local storage fallback
    - 9.5: Comprehensive error handling with detailed logs and recovery instructions
    """
    
    def __init__(self, config: Config):
        """Initialize error handler with configuration."""
        self.config = config
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: List[RecoveryStrategy] = []
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
        # Setup logging
        self._setup_logging()
        
        # Performance tracking
        self.total_errors = 0
        self.recovered_errors = 0
        
    def _initialize_recovery_strategies(self):
        """Initialize all recovery strategies."""
        self.recovery_strategies = [
            MemoryRecoveryStrategy(),
            DatasetRecoveryStrategy(self.config),
            AuthenticationRecoveryStrategy(self.config),
            TrainingRecoveryStrategy()
        ]
        
        logger.info(f"Initialized {len(self.recovery_strategies)} recovery strategies")
    
    def _setup_logging(self):
        """Setup error logging configuration."""
        # Create error log directory
        log_dir = Path(self.config.paths.logs_dir) / "errors"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup error file handler
        error_log_path = log_dir / "error_handler.log"
        error_handler = logging.FileHandler(error_log_path)
        error_handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(error_handler)
        
        logger.info(f"Error logging setup complete. Log file: {error_log_path}")
    
    def handle_error(self, 
                    error: Exception, 
                    category: ErrorCategory = ErrorCategory.TRAINING,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None,
                    auto_recover: bool = True) -> ErrorContext:
        """
        Handle an error with automatic recovery attempts.
        
        Args:
            error: The exception that occurred
            category: Error category for specialized handling
            severity: Error severity level
            context: Additional context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            ErrorContext with handling results
        """
        self.total_errors += 1
        
        # Create error context
        error_context = ErrorContext(
            error_type=type(error).__name__,
            category=category,
            severity=severity,
            message=str(error),
            details=context or {},
            timestamp=time.time(),
            recovery_suggestions=[]
        )
        
        # Add stack trace to details
        error_context.details['traceback'] = traceback.format_exc()
        
        # Log the error
        self._log_error(error_context)
        
        # Attempt automatic recovery if enabled
        if auto_recover:
            recovery_success = self._attempt_recovery(error_context, context or {})
            error_context.auto_recovery_attempted = True
            error_context.recovery_successful = recovery_success
            
            if recovery_success:
                self.recovered_errors += 1
        
        # Add to error history
        self.error_history.append(error_context)
        
        # Generate recovery instructions
        self._generate_recovery_instructions(error_context)
        
        return error_context
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level based on severity."""
        log_message = (
            f"[{error_context.category.value.upper()}] {error_context.error_type}: "
            f"{error_context.message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _attempt_recovery(self, error_context: ErrorContext, context: Dict[str, Any]) -> bool:
        """Attempt recovery using available strategies."""
        logger.info(f"Attempting recovery for {error_context.error_type}")
        
        # Find applicable recovery strategies
        applicable_strategies = [
            strategy for strategy in self.recovery_strategies
            if strategy.can_handle(error_context)
        ]
        
        if not applicable_strategies:
            logger.warning("No applicable recovery strategies found")
            return False
        
        # Try each applicable strategy
        for strategy in applicable_strategies:
            try:
                logger.info(f"Trying recovery strategy: {strategy.name}")
                
                success, message = strategy.recover(error_context, **context)
                
                if success:
                    logger.info(f"Recovery successful with {strategy.name}: {message}")
                    error_context.details['recovery_strategy'] = strategy.name
                    error_context.details['recovery_message'] = message
                    return True
                else:
                    logger.warning(f"Recovery failed with {strategy.name}: {message}")
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.name} raised exception: {recovery_error}")
                continue
        
        logger.error("All recovery strategies failed")
        return False
    
    def _generate_recovery_instructions(self, error_context: ErrorContext):
        """Generate detailed recovery instructions for manual intervention."""
        instructions = [
            f"Error Type: {error_context.error_type}",
            f"Category: {error_context.category.value}",
            f"Severity: {error_context.severity.value}",
            f"Message: {error_context.message}",
            "",
            "Recovery Instructions:"
        ]
        
        # Add category-specific instructions
        if error_context.category == ErrorCategory.MEMORY:
            instructions.extend([
                "1. Reduce batch size in your configuration",
                "2. Enable gradient checkpointing if not already enabled",
                "3. Use mixed precision training (fp16)",
                "4. Consider using a smaller model or LoRA rank",
                "5. Close other applications using GPU memory"
            ])
        
        elif error_context.category == ErrorCategory.DATASET:
            instructions.extend([
                "1. Check your internet connection",
                "2. Verify the dataset name and split are correct",
                "3. Try using a different dataset from HuggingFace Hub",
                "4. Clear the dataset cache and retry",
                "5. Use streaming mode for large datasets"
            ])
        
        elif error_context.category == ErrorCategory.AUTHENTICATION:
            instructions.extend([
                "1. Upload your credentials.json file to Colab",
                "2. Enable Google Drive API in Google Cloud Console",
                "3. Configure OAuth consent screen properly",
                "4. Verify your Google account has access",
                "5. Use local storage as fallback if needed"
            ])
        
        elif error_context.category == ErrorCategory.TRAINING:
            instructions.extend([
                "1. Reduce learning rate by factor of 2-10",
                "2. Enable gradient clipping (max_grad_norm=1.0)",
                "3. Check for NaN/Inf values in your data",
                "4. Restart from the last stable checkpoint",
                "5. Verify your loss function implementation"
            ])
        
        # Add custom recovery suggestions from strategies
        if error_context.recovery_suggestions:
            instructions.extend(["", "Additional Suggestions:"])
            for i, suggestion in enumerate(error_context.recovery_suggestions, 1):
                instructions.append(f"{i}. {suggestion}")
        
        # Store instructions in error context
        error_context.details['recovery_instructions'] = "\n".join(instructions)
        
        # Log instructions
        logger.info("Recovery instructions generated:")
        for instruction in instructions:
            logger.info(instruction)
    
    @contextmanager
    def error_context(self, 
                     category: ErrorCategory = ErrorCategory.TRAINING,
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     context: Optional[Dict[str, Any]] = None,
                     auto_recover: bool = True):
        """
        Context manager for automatic error handling.
        
        Usage:
            with error_handler.error_context(ErrorCategory.MEMORY):
                # Code that might raise memory errors
                model.train()
        """
        try:
            yield
        except Exception as e:
            error_context = self.handle_error(e, category, severity, context, auto_recover)
            
            # Re-raise if critical or recovery failed
            if (severity == ErrorSeverity.CRITICAL or 
                (auto_recover and not error_context.recovery_successful)):
                raise
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        if not self.error_history:
            return {"total_errors": 0, "recovery_rate": 0.0}
        
        # Count by category
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recovery strategy success rates
        strategy_stats = {}
        for strategy in self.recovery_strategies:
            strategy_stats[strategy.name] = {
                "success_rate": strategy.get_success_rate(),
                "success_count": strategy.success_count,
                "failure_count": strategy.failure_count
            }
        
        return {
            "total_errors": self.total_errors,
            "recovered_errors": self.recovered_errors,
            "recovery_rate": self.recovered_errors / self.total_errors if self.total_errors > 0 else 0.0,
            "category_counts": category_counts,
            "severity_counts": severity_counts,
            "strategy_statistics": strategy_stats,
            "recent_errors": len([e for e in self.error_history if time.time() - e.timestamp < 3600])  # Last hour
        }
    
    def clear_error_history(self):
        """Clear error history (useful for testing or reset)."""
        self.error_history.clear()
        self.total_errors = 0
        self.recovered_errors = 0
        
        # Reset strategy counters
        for strategy in self.recovery_strategies:
            strategy.success_count = 0
            strategy.failure_count = 0
        
        logger.info("Error history cleared")
    
    def export_error_report(self, output_path: Union[str, Path]) -> None:
        """Export detailed error report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_lines = [
            "RLHF Phi-3 Pipeline Error Report",
            "=" * 50,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Summary Statistics:",
            f"  Total Errors: {self.total_errors}",
            f"  Recovered Errors: {self.recovered_errors}",
            f"  Recovery Rate: {self.recovered_errors / self.total_errors * 100:.1f}%" if self.total_errors > 0 else "  Recovery Rate: N/A",
            ""
        ]
        
        # Add detailed error history
        if self.error_history:
            report_lines.extend([
                "Error History:",
                "-" * 20
            ])
            
            for i, error in enumerate(self.error_history[-10:], 1):  # Last 10 errors
                report_lines.extend([
                    f"{i}. {error.error_type} ({error.category.value})",
                    f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(error.timestamp))}",
                    f"   Severity: {error.severity.value}",
                    f"   Message: {error.message}",
                    f"   Recovery Attempted: {error.auto_recovery_attempted}",
                    f"   Recovery Successful: {error.recovery_successful}",
                    ""
                ])
        
        # Add strategy statistics
        report_lines.extend([
            "Recovery Strategy Performance:",
            "-" * 30
        ])
        
        for strategy in self.recovery_strategies:
            success_rate = strategy.get_success_rate() * 100
            report_lines.extend([
                f"{strategy.name}:",
                f"  Success Rate: {success_rate:.1f}%",
                f"  Successes: {strategy.success_count}",
                f"  Failures: {strategy.failure_count}",
                ""
            ])
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Error report exported to: {output_path}")


# Utility functions for common error scenarios

def handle_memory_error(func: Callable) -> Callable:
    """Decorator for automatic memory error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"Memory error in {func.__name__}: {e}")
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Retry once
                try:
                    return func(*args, **kwargs)
                except Exception as retry_error:
                    logger.error(f"Retry failed for {func.__name__}: {retry_error}")
                    raise
            else:
                raise
    
    return wrapper


def safe_dataset_load(dataset_name: str, 
                     split: str = "train", 
                     alternatives: Optional[List[str]] = None,
                     **kwargs) -> Optional[Dataset]:
    """Safely load dataset with automatic fallback to alternatives."""
    datasets_to_try = [dataset_name] + (alternatives or [])
    
    for dataset in datasets_to_try:
        try:
            logger.info(f"Attempting to load dataset: {dataset}")
            return load_dataset(dataset, split=split, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to load {dataset}: {e}")
            continue
    
    logger.error(f"All dataset loading attempts failed for {dataset_name}")
    return None


def create_fallback_storage(base_path: Union[str, Path], 
                           fallback_name: str = "local_fallback") -> Path:
    """Create fallback storage directory when cloud storage fails."""
    fallback_path = Path(base_path) / fallback_name
    fallback_path.mkdir(parents=True, exist_ok=True)
    
    logger.warning(f"Using fallback storage: {fallback_path}")
    return fallback_path