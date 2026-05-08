"""
SFT (Supervised Fine-Tuning) Trainer for RLHF Phi-3 Pipeline

This module implements the first stage of the RLHF pipeline: Supervised Fine-Tuning.
It handles instruction-following dataset training with PEFT/LoRA, progress tracking,
and checkpoint integration.

Key Features:
- Instruction-following dataset training with Phi-3 chat template
- PEFT/LoRA integration for memory efficiency
- Progress tracking and experiment logging
- Checkpoint persistence and recovery
- Memory optimization and monitoring

Requirements satisfied:
- 1.1: SFT stage implementation as first stage of three-stage pipeline
- 5.1: PEFT/LoRA implementation to reduce trainable parameters
- 6.1: Progress tracking and metric logging
"""

import os
import gc
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, asdict

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback
)
from peft import PeftModel
from trl import SFTTrainer
from datasets import Dataset

from ..config.config_manager import Config
from ..models.model_manager import ModelManager
from ..data.dataset_manager import DatasetManager
from ..checkpoints.checkpoint_manager import CheckpointManager
from ..tracking.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainingResult:
    """Result of SFT training execution."""
    success: bool
    checkpoint_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    final_loss: Optional[float] = None
    total_steps: Optional[int] = None


class SFTTrainingCallback(TrainerCallback):
    """Custom callback for SFT training monitoring and logging."""
    
    def __init__(
        self, 
        experiment_tracker: ExperimentTracker,
        model_manager: ModelManager,
        checkpoint_manager: CheckpointManager,
        config: Config
    ):
        """
        Initialize SFT training callback.
        
        Args:
            experiment_tracker: Experiment tracking instance
            model_manager: Model manager instance
            checkpoint_manager: Checkpoint manager instance
            config: Pipeline configuration
        """
        self.experiment_tracker = experiment_tracker
        self.model_manager = model_manager
        self.checkpoint_manager = checkpoint_manager
        self.config = config
        
        # Training monitoring
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []
        self.memory_history = []
        
        # Early stopping configuration
        self.early_stopping_patience = getattr(config.training.sft, 'early_stopping_patience', 5)
        self.loss_divergence_threshold = 2.0
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        logger.info("Starting SFT training")
        self.experiment_tracker.log_metrics(
            {"sft/training_started": 1},
            step=0
        )
        
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when logging metrics."""
        if logs is None:
            return
            
        # Monitor memory usage
        self._monitor_memory_usage(state.global_step)
        
        # Log metrics to experiment tracker
        sft_logs = {f"sft/{k}": v for k, v in logs.items()}
        self.experiment_tracker.log_metrics(sft_logs, state.global_step)
        
        # Track loss for early stopping and divergence detection
        if 'train_loss' in logs:
            current_loss = logs['train_loss']
            self.loss_history.append(current_loss)
            
            # Update best loss and patience counter
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Check for loss divergence
            if self._check_loss_divergence(current_loss):
                logger.warning("Loss divergence detected in SFT training - stopping")
                control.should_training_stop = True
                
            # Log loss statistics
            if len(self.loss_history) >= 10:
                recent_avg = np.mean(self.loss_history[-10:])
                self.experiment_tracker.log_metrics({
                    "sft/loss_recent_avg": recent_avg,
                    "sft/loss_best": self.best_loss,
                    "sft/patience_counter": self.patience_counter
                }, state.global_step)
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when saving a checkpoint."""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # Log checkpoint to experiment tracker
        self.experiment_tracker.log_model_checkpoint(
            checkpoint_path,
            "sft",
            {
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": self.best_loss,
                "learning_rate": state.log_history[-1].get('learning_rate', 0) if state.log_history else 0
            }
        )
        
        logger.info(f"SFT checkpoint saved at step {state.global_step}: {checkpoint_path}")
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called after evaluation."""
        if logs:
            eval_logs = {f"sft/eval_{k}": v for k, v in logs.items()}
            self.experiment_tracker.log_metrics(eval_logs, state.global_step)
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        logger.info("SFT training completed")
        
        # Log final statistics
        final_metrics = {
            "sft/training_completed": 1,
            "sft/total_steps": state.global_step,
            "sft/final_loss": self.best_loss,
            "sft/total_epochs": state.epoch
        }
        
        if self.memory_history:
            final_metrics["sft/peak_memory_gb"] = max(
                entry.get('gpu_allocated_gb', 0) for entry in self.memory_history
            )
        
        self.experiment_tracker.log_metrics(final_metrics, state.global_step)
    
    def _monitor_memory_usage(self, step: int) -> None:
        """Monitor and log memory usage."""
        try:
            memory_stats = self.model_manager.get_memory_usage()
            
            # Store in history
            memory_entry = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                **memory_stats
            }
            self.memory_history.append(memory_entry)
            
            # Log to experiment tracker (less frequently to avoid spam)
            if step % (self.config.logging.log_steps * 5) == 0:
                memory_logs = {f"sft/memory_{k}": v for k, v in memory_stats.items()}
                self.experiment_tracker.log_metrics(memory_logs, step)
            
            # Check for memory issues and trigger optimization if needed
            if 'gpu_utilization' in memory_stats:
                if memory_stats['gpu_utilization'] > 0.95:
                    logger.warning(f"High GPU memory usage in SFT: {memory_stats['gpu_utilization']:.1%}")
                    self.model_manager.handle_memory_exhaustion()
                    
        except Exception as e:
            logger.warning(f"SFT memory monitoring failed: {str(e)}")
    
    def _check_loss_divergence(self, current_loss: float) -> bool:
        """Check if training loss is diverging."""
        try:
            # Need sufficient history to check divergence
            if len(self.loss_history) < 10:
                return False
            
            # Check for NaN or inf
            if np.isnan(current_loss) or np.isinf(current_loss):
                logger.error(f"Invalid loss value in SFT: {current_loss}")
                return True
            
            # Calculate recent average
            recent_losses = self.loss_history[-10:]
            recent_avg = np.mean(recent_losses)
            
            # Check for sudden increase
            if current_loss > recent_avg * self.loss_divergence_threshold:
                logger.warning(
                    f"SFT loss divergence detected: {current_loss:.4f} vs recent avg {recent_avg:.4f}"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"SFT loss divergence check failed: {str(e)}")
            return False


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for the RLHF pipeline.
    
    This class implements the first stage of RLHF training, handling instruction-following
    dataset training with PEFT/LoRA integration, progress tracking, and checkpoint management.
    
    Requirements satisfied:
    - 1.1: SFT stage implementation as first stage of three-stage pipeline
    - 5.1: PEFT/LoRA implementation to reduce trainable parameters
    - 6.1: Progress tracking and metric logging
    """
    
    def __init__(
        self,
        config: Config,
        model_manager: Optional[ModelManager] = None,
        dataset_manager: Optional[DatasetManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        experiment_tracker: Optional[ExperimentTracker] = None
    ):
        """
        Initialize SFT trainer.
        
        Args:
            config: Pipeline configuration
            model_manager: Optional model manager instance
            dataset_manager: Optional dataset manager instance
            checkpoint_manager: Optional checkpoint manager instance
            experiment_tracker: Optional experiment tracker instance
        """
        self.config = config
        
        # Initialize component managers
        self.model_manager = model_manager or ModelManager(config)
        self.dataset_manager = dataset_manager or DatasetManager(config)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            base_path=config.paths.base_output_dir,
            enable_drive_sync=True
        )
        self.experiment_tracker = experiment_tracker or ExperimentTracker(
            project_name=config.wandb.project,
            config=config,
            entity=config.wandb.entity
        )
        
        # Training state
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_dataset = None
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.logging.level))
        logger.info("SFTTrainer initialized")
    
    def train(
        self,
        dataset_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs
    ) -> SFTTrainingResult:
        """
        Execute SFT training.
        
        Args:
            dataset_name: Name of the SFT dataset to use
            resume_from_checkpoint: Path to checkpoint to resume from
            **kwargs: Additional training arguments
            
        Returns:
            SFTTrainingResult with training outcomes
            
        Requirement 1.1: SFT stage implementation
        Requirement 5.1: PEFT/LoRA implementation
        Requirement 6.1: Progress tracking and metric logging
        """
        logger.info("Starting SFT (Supervised Fine-Tuning) training")
        start_time = time.time()
        
        try:
            # Start experiment tracking
            run_name = f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.experiment_tracker.start_run(stage="sft", run_name=run_name)
            
            # Step 1: Load and prepare model
            logger.info("Loading and preparing model for SFT")
            self._prepare_model()
            
            # Step 2: Load and preprocess dataset
            logger.info("Loading and preprocessing SFT dataset")
            self._prepare_dataset(dataset_name)
            
            # Step 3: Setup training arguments and trainer
            logger.info("Setting up SFT trainer")
            self._setup_trainer(resume_from_checkpoint, **kwargs)
            
            # Step 4: Execute training
            logger.info("Executing SFT training")
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Step 5: Save final checkpoint
            logger.info("Saving SFT checkpoint")
            checkpoint_path = self._save_checkpoint(train_result)
            
            # Step 6: Prepare result
            duration = time.time() - start_time
            result = SFTTrainingResult(
                success=True,
                checkpoint_path=checkpoint_path,
                metrics=train_result.metrics if hasattr(train_result, 'metrics') else {},
                duration_seconds=duration,
                memory_peak_gb=self._get_peak_memory_usage(),
                final_loss=train_result.training_loss if hasattr(train_result, 'training_loss') else None,
                total_steps=train_result.global_step if hasattr(train_result, 'global_step') else None
            )
            
            logger.info(f"SFT training completed successfully in {duration:.2f} seconds")
            logger.info(f"Final checkpoint: {checkpoint_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"SFT training failed: {str(e)}")
            duration = time.time() - start_time
            
            result = SFTTrainingResult(
                success=False,
                error_message=str(e),
                duration_seconds=duration,
                memory_peak_gb=self._get_peak_memory_usage()
            )
            
            # Log error details
            self._log_training_error(e, duration)
            
            return result
            
        finally:
            # Cleanup and finish experiment tracking
            self._cleanup_resources()
            self.experiment_tracker.finish_run()
    
    def _prepare_model(self) -> None:
        """Load and prepare model for SFT training."""
        try:
            # Load base model
            base_model = self.model_manager.load_base_model()
            
            # Apply PEFT configuration
            peft_model = self.model_manager.apply_peft(base_model, task_type="CAUSAL_LM")
            
            # Prepare for training (gradient checkpointing, etc.)
            self.model = self.model_manager.prepare_for_training(peft_model)
            
            # Get tokenizer
            self.tokenizer = self.model_manager.tokenizer
            
            # Log model information
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model loaded: {total_params:,} total parameters")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
            
            # Log to experiment tracker
            self.experiment_tracker.log_metrics({
                "sft/model_total_params": total_params,
                "sft/model_trainable_params": trainable_params,
                "sft/model_trainable_ratio": trainable_params / total_params
            }, step=0)
            
        except Exception as e:
            logger.error(f"Failed to prepare model for SFT: {str(e)}")
            raise
    
    def _prepare_dataset(self, dataset_name: Optional[str] = None) -> None:
        """Load and preprocess dataset for SFT training."""
        try:
            # Use dataset name from config if not provided
            if dataset_name is None:
                dataset_name = self.config.datasets.sft.name
            
            # Load SFT dataset
            logger.info(f"Loading SFT dataset: {dataset_name}")
            raw_dataset = self.dataset_manager.load_sft_dataset(
                dataset_name=dataset_name,
                streaming=False  # Load full dataset for SFT
            )
            
            # Preprocess dataset
            logger.info("Preprocessing SFT dataset")
            self.training_dataset = self.dataset_manager.preprocess_sft_data(raw_dataset)
            
            # Validate dataset
            if not self._validate_dataset():
                raise ValueError("SFT dataset validation failed")
            
            # Log dataset information
            dataset_size = len(self.training_dataset)
            logger.info(f"SFT dataset prepared: {dataset_size:,} samples")
            
            # Sample a few examples for logging
            sample_texts = []
            for i in range(min(3, dataset_size)):
                sample = self.training_dataset[i]
                if 'text' in sample:
                    sample_texts.append(sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text'])
            
            self.experiment_tracker.log_metrics({
                "sft/dataset_size": dataset_size,
                "sft/dataset_name": dataset_name
            }, step=0)
            
            # Log sample texts (first few characters for inspection)
            for i, text in enumerate(sample_texts):
                logger.info(f"Sample {i+1}: {text}")
            
        except Exception as e:
            logger.error(f"Failed to prepare SFT dataset: {str(e)}")
            raise
    
    def _setup_trainer(self, resume_from_checkpoint: Optional[str] = None, **kwargs) -> None:
        """Setup the SFT trainer with proper configuration."""
        try:
            # Create training arguments
            training_args = self._create_training_arguments(**kwargs)
            
            # Create custom callback
            sft_callback = SFTTrainingCallback(
                experiment_tracker=self.experiment_tracker,
                model_manager=self.model_manager,
                checkpoint_manager=self.checkpoint_manager,
                config=self.config
            )
            
            # Create SFT trainer
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=training_args,
                train_dataset=self.training_dataset,
                dataset_text_field="text",  # Field containing the formatted text
                max_seq_length=self.config.model.max_length,
                packing=False,  # Disable packing for stability
                callbacks=[sft_callback]
            )
            
            # Add early stopping if configured
            if hasattr(self.config.training.sft, 'early_stopping_patience'):
                early_stopping = EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.sft.early_stopping_patience,
                    early_stopping_threshold=0.001
                )
                self.trainer.add_callback(early_stopping)
            
            logger.info("SFT trainer setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup SFT trainer: {str(e)}")
            raise
    
    def _create_training_arguments(self, **kwargs) -> TrainingArguments:
        """Create training arguments for SFT training."""
        # Base output directory
        output_dir = Path(self.config.paths.base_output_dir) / "sft_training"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Merge config with any overrides
        sft_config = self.config.training.sft
        
        return TrainingArguments(
            output_dir=str(output_dir),
            
            # Training schedule
            num_train_epochs=kwargs.get('num_train_epochs', sft_config.epochs),
            max_steps=kwargs.get('max_steps', sft_config.max_steps),
            
            # Batch size and accumulation
            per_device_train_batch_size=kwargs.get('per_device_train_batch_size', sft_config.batch_size),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', sft_config.gradient_accumulation_steps),
            
            # Learning rate and optimization
            learning_rate=kwargs.get('learning_rate', sft_config.learning_rate),
            warmup_steps=kwargs.get('warmup_steps', sft_config.warmup_steps),
            lr_scheduler_type=kwargs.get('lr_scheduler_type', self.config.optimization.scheduler_type),
            optim=kwargs.get('optim', self.config.optimization.optimizer_type),
            weight_decay=kwargs.get('weight_decay', self.config.optimization.weight_decay),
            max_grad_norm=kwargs.get('max_grad_norm', self.config.optimization.max_grad_norm),
            
            # Memory optimization
            fp16=kwargs.get('fp16', self.config.optimization.fp16),
            gradient_checkpointing=kwargs.get('gradient_checkpointing', self.config.optimization.gradient_checkpointing),
            dataloader_num_workers=kwargs.get('dataloader_num_workers', self.config.optimization.dataloader_num_workers),
            
            # Logging and checkpointing
            logging_steps=kwargs.get('logging_steps', self.config.logging.log_steps),
            save_steps=kwargs.get('save_steps', self.config.checkpointing.save_steps),
            save_total_limit=kwargs.get('save_total_limit', self.config.checkpointing.save_total_limit),
            
            # Evaluation (if eval dataset is provided)
            evaluation_strategy="no",  # No evaluation dataset for SFT by default
            
            # Model selection
            load_best_model_at_end=False,
            metric_for_best_model="loss",
            greater_is_better=False,
            
            # Experiment tracking
            report_to="wandb" if self.config.wandb.project else "none",
            run_name=f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            # Misc
            remove_unused_columns=False,
            push_to_hub=False,
            seed=42,
        )
    
    def _validate_dataset(self) -> bool:
        """Validate the prepared SFT dataset."""
        try:
            if self.training_dataset is None:
                logger.error("Training dataset is None")
                return False
            
            dataset_size = len(self.training_dataset)
            if dataset_size == 0:
                logger.error("Training dataset is empty")
                return False
            
            # Check required fields
            required_fields = ['text']  # SFTTrainer expects 'text' field
            sample = self.training_dataset[0]
            
            for field in required_fields:
                if field not in sample:
                    logger.error(f"Missing required field '{field}' in SFT dataset")
                    return False
                
                if sample[field] is None or (isinstance(sample[field], str) and len(sample[field].strip()) == 0):
                    logger.error(f"Empty '{field}' field in SFT dataset sample")
                    return False
            
            # Check text field content
            text_sample = sample['text']
            if not isinstance(text_sample, str):
                logger.error(f"Text field must be string, got {type(text_sample)}")
                return False
            
            # Check for minimum reasonable length
            if len(text_sample.strip()) < 10:
                logger.warning(f"Very short text sample: {len(text_sample)} characters")
            
            logger.info(f"SFT dataset validation passed: {dataset_size:,} samples")
            return True
            
        except Exception as e:
            logger.error(f"SFT dataset validation failed: {str(e)}")
            return False
    
    def _save_checkpoint(self, train_result) -> str:
        """Save the final SFT checkpoint."""
        try:
            # Extract metrics
            metrics = {}
            if hasattr(train_result, 'metrics'):
                metrics = train_result.metrics
            
            # Save checkpoint using checkpoint manager
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.trainer.optimizer if self.trainer else None,
                epoch=metrics.get('epoch', 0),
                step=metrics.get('global_step', 0),
                stage="sft",
                metrics=metrics,
                config_hash=str(hash(str(asdict(self.config))))
            )
            
            checkpoint_path = self.checkpoint_manager.base_path / checkpoint_id
            
            # Also save in trainer's output directory for compatibility
            if self.trainer:
                trainer_checkpoint_dir = Path(self.trainer.args.output_dir) / "final_checkpoint"
                trainer_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model and tokenizer
                self.model.save_pretrained(trainer_checkpoint_dir)
                self.tokenizer.save_pretrained(trainer_checkpoint_dir)
                
                logger.info(f"SFT model also saved to: {trainer_checkpoint_dir}")
            
            logger.info(f"SFT checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save SFT checkpoint: {str(e)}")
            raise
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage during training."""
        try:
            memory_stats = self.model_manager.get_memory_usage()
            return memory_stats.get('gpu_allocated_gb', 0.0)
        except Exception:
            return 0.0
    
    def _log_training_error(self, error: Exception, duration: float) -> None:
        """Log detailed error information."""
        error_details = {
            'stage': 'sft',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save error details to file
        error_file = Path(self.config.paths.base_output_dir) / f"sft_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(error_file, 'w') as f:
            json.dump(error_details, f, indent=2)
        
        logger.info(f"SFT error details saved to: {error_file}")
        
        # Log to experiment tracker
        try:
            self.experiment_tracker.log_metrics({
                "sft/error": 1,
                "sft/error_duration": duration
            }, step=0)
        except Exception:
            pass  # Don't fail on logging errors
    
    def _cleanup_resources(self) -> None:
        """Clean up resources and free memory."""
        try:
            logger.info("Cleaning up SFT trainer resources")
            
            # Clear model references
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.trainer is not None:
                del self.trainer
                self.trainer = None
            
            if self.training_dataset is not None:
                del self.training_dataset
                self.training_dataset = None
            
            # Cleanup model manager
            if self.model_manager:
                self.model_manager.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("SFT trainer resource cleanup completed")
            
        except Exception as e:
            logger.warning(f"SFT trainer resource cleanup failed: {str(e)}")


def create_sft_trainer(
    config: Config,
    model_manager: Optional[ModelManager] = None,
    dataset_manager: Optional[DatasetManager] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    experiment_tracker: Optional[ExperimentTracker] = None
) -> SFTTrainer:
    """
    Factory function to create an SFT trainer instance.
    
    Args:
        config: Pipeline configuration
        model_manager: Optional model manager instance
        dataset_manager: Optional dataset manager instance
        checkpoint_manager: Optional checkpoint manager instance
        experiment_tracker: Optional experiment tracker instance
        
    Returns:
        Configured SFTTrainer instance
    """
    return SFTTrainer(
        config=config,
        model_manager=model_manager,
        dataset_manager=dataset_manager,
        checkpoint_manager=checkpoint_manager,
        experiment_tracker=experiment_tracker
    )