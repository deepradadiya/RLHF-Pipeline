"""
Reward Model Trainer for RLHF Phi-3 Pipeline

This module implements the second stage of the RLHF pipeline: Reward Model Training.
It handles preference dataset training for reward modeling, evaluation, and validation.

Key Features:
- Preference dataset training with human feedback pairs
- Reward model evaluation and validation
- PEFT/LoRA integration for memory efficiency
- Progress tracking and experiment logging
- Checkpoint persistence and recovery

Requirements satisfied:
- 1.2: Reward Model training stage as second stage of three-stage pipeline
- 7.5: Support for preference datasets in reward training
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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..config.config_manager import Config
from ..models.model_manager import ModelManager
from ..data.dataset_manager import DatasetManager
from ..checkpoints.checkpoint_manager import CheckpointManager
from ..tracking.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class RewardTrainingResult:
    """Result of reward model training execution."""
    success: bool
    checkpoint_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    total_steps: Optional[int] = None


class RewardModel(nn.Module):
    """
    Reward model wrapper that adds a reward head to the base language model.
    
    This model takes the SFT-trained model and adds a scalar reward head
    to predict preference scores for response pairs.
    """
    
    def __init__(self, base_model: PeftModel, config: Config):
        """
        Initialize reward model.
        
        Args:
            base_model: SFT-trained PEFT model
            config: Pipeline configuration
        """
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Get hidden size from base model
        hidden_size = base_model.config.hidden_size
        
        # Add reward head (single scalar output)
        self.reward_head = nn.Linear(hidden_size, 1)
        
        # Initialize reward head weights
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)
        
        # Freeze base model parameters except LoRA
        for name, param in self.base_model.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through reward model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with rewards and hidden states
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        # Get sequence lengths for each example
        sequence_lengths = attention_mask.sum(dim=1) - 1
        
        # Extract last token representations
        batch_size = last_hidden_state.size(0)
        last_token_hidden = last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device),
            sequence_lengths
        ]
        
        # Compute rewards
        rewards = self.reward_head(last_token_hidden).squeeze(-1)
        
        return {
            'rewards': rewards,
            'hidden_states': outputs.hidden_states,
            'last_hidden_state': last_hidden_state
        }


class RewardTrainingCallback(TrainerCallback):
    """Custom callback for reward model training monitoring and logging."""
    
    def __init__(
        self, 
        experiment_tracker: ExperimentTracker,
        model_manager: ModelManager,
        checkpoint_manager: CheckpointManager,
        config: Config
    ):
        """
        Initialize reward training callback.
        
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
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.loss_history = []
        self.accuracy_history = []
        self.memory_history = []
        
        # Early stopping configuration
        self.early_stopping_patience = getattr(config.training.reward, 'early_stopping_patience', 3)
        self.min_delta = getattr(config.training.reward, 'min_delta', 0.001)
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Handle logging events."""
        if state.log_history:
            latest_log = state.log_history[-1]
            
            # Track memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                self.memory_history.append(memory_used)
                latest_log['memory_gb'] = memory_used
            
            # Log to experiment tracker
            step = latest_log.get('step', state.global_step)
            
            # Extract metrics
            metrics_to_log = {}
            for key, value in latest_log.items():
                if key not in ['epoch', 'step'] and isinstance(value, (int, float)):
                    metrics_to_log[key] = value
            
            if metrics_to_log:
                self.experiment_tracker.log_metrics(metrics_to_log, step)
            
            # Track training progress
            if 'train_loss' in latest_log:
                self.loss_history.append(latest_log['train_loss'])
            
            if 'eval_accuracy' in latest_log:
                current_accuracy = latest_log['eval_accuracy']
                self.accuracy_history.append(current_accuracy)
                
                # Check for improvement
                if current_accuracy > self.best_accuracy + self.min_delta:
                    self.best_accuracy = current_accuracy
                    self.patience_counter = 0
                    logger.info(f"New best accuracy: {self.best_accuracy:.4f}")
                else:
                    self.patience_counter += 1
                    logger.info(f"No improvement for {self.patience_counter} evaluations")
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Handle evaluation events."""
        # Check early stopping
        if self.patience_counter >= self.early_stopping_patience:
            logger.warning(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
            control.should_training_stop = True
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Handle checkpoint saving events."""
        try:
            # Save checkpoint through checkpoint manager
            model = kwargs.get('model')
            if model is not None:
                checkpoint_name = f"reward_step_{state.global_step}"
                self.checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=None,  # Will be handled by trainer
                    epoch=int(state.epoch) if state.epoch else 0,
                    stage="reward",
                    checkpoint_name=checkpoint_name
                )
                logger.info(f"Reward model checkpoint saved: {checkpoint_name}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")


class RewardTrainer:
    """
    Reward Model Trainer for RLHF Pipeline.
    
    Implements preference-based training using human feedback pairs to train
    a reward model that can evaluate response quality.
    """
    
    def __init__(
        self,
        config: Config,
        model_manager: ModelManager,
        dataset_manager: DatasetManager,
        checkpoint_manager: CheckpointManager,
        experiment_tracker: ExperimentTracker
    ):
        """
        Initialize reward trainer.
        
        Args:
            config: Pipeline configuration
            model_manager: Model manager instance
            dataset_manager: Dataset manager instance
            checkpoint_manager: Checkpoint manager instance
            experiment_tracker: Experiment tracker instance
        """
        self.config = config
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.checkpoint_manager = checkpoint_manager
        self.experiment_tracker = experiment_tracker
        
        # Training state
        self.reward_model = None
        self.tokenizer = None
        self.training_dataset = None
        self.eval_dataset = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_reward_model(self, sft_checkpoint_path: str) -> RewardModel:
        """
        Prepare reward model from SFT checkpoint.
        
        Args:
            sft_checkpoint_path: Path to SFT checkpoint
            
        Returns:
            Prepared reward model
            
        Requirement 1.2: Load SFT model as base for reward training
        """
        self.logger.info(f"Loading SFT model from {sft_checkpoint_path}")
        
        try:
            # Load SFT model
            sft_model, _, _ = self.checkpoint_manager.load_checkpoint(
                Path(sft_checkpoint_path).name
            )
            
            if sft_model is None:
                # Fallback: load from path directly
                base_model = self.model_manager.load_base_model()
                sft_model = PeftModel.from_pretrained(base_model, sft_checkpoint_path)
            
            # Create reward model wrapper
            reward_model = RewardModel(sft_model, self.config)
            
            # Get tokenizer
            self.tokenizer = self.model_manager.tokenizer
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            reward_model = reward_model.to(device)
            
            self.logger.info("Reward model prepared successfully")
            return reward_model
            
        except Exception as e:
            self.logger.error(f"Failed to prepare reward model: {str(e)}")
            raise
    
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and evaluation datasets.
        
        Returns:
            Tuple of (training_dataset, eval_dataset)
            
        Requirement 7.5: Support for preference datasets
        """
        self.logger.info("Loading preference datasets")
        
        try:
            # Load preference dataset
            preference_dataset = self.dataset_manager.load_preference_dataset(streaming=False)
            
            # Preprocess for reward training
            processed_dataset = self.dataset_manager.preprocess_preference_data(preference_dataset)
            
            # Split into train/eval
            if len(processed_dataset) < 100:
                # Small dataset - use 80/20 split
                split_point = int(0.8 * len(processed_dataset))
                train_dataset = processed_dataset.select(range(split_point))
                eval_dataset = processed_dataset.select(range(split_point, len(processed_dataset)))
            else:
                # Larger dataset - use train_test_split
                split_dataset = processed_dataset.train_test_split(test_size=0.2, seed=42)
                train_dataset = split_dataset['train']
                eval_dataset = split_dataset['test']
            
            self.logger.info(f"Training dataset size: {len(train_dataset)}")
            self.logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            self.logger.error(f"Failed to prepare datasets: {str(e)}")
            raise
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics for reward model.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        
        # Convert to binary predictions (chosen > rejected)
        # Predictions shape: (batch_size, 2) for [chosen_reward, rejected_reward]
        if predictions.shape[1] == 2:
            binary_preds = (predictions[:, 0] > predictions[:, 1]).astype(int)
        else:
            # Fallback: use sign of difference
            binary_preds = (predictions > 0).astype(int)
        
        # Labels should be 1 (chosen is better)
        binary_labels = np.ones_like(binary_preds)
        
        # Compute metrics
        accuracy = accuracy_score(binary_labels, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_labels, binary_preds, average='binary', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def create_training_arguments(self) -> TrainingArguments:
        """
        Create training arguments for reward model training.
        
        Returns:
            Training arguments
        """
        # Get reward training config
        reward_config = self.config.training.reward
        
        # Create output directory
        output_dir = Path(self.config.paths.output_dir) / "reward_training"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return TrainingArguments(
            output_dir=str(output_dir),
            
            # Training parameters
            num_train_epochs=reward_config.epochs,
            per_device_train_batch_size=reward_config.batch_size,
            per_device_eval_batch_size=reward_config.eval_batch_size,
            gradient_accumulation_steps=reward_config.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=reward_config.learning_rate,
            weight_decay=reward_config.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.999,
            max_grad_norm=1.0,
            
            # Scheduling
            lr_scheduler_type="cosine",
            warmup_steps=reward_config.warmup_steps,
            
            # Evaluation and logging
            evaluation_strategy="steps",
            eval_steps=reward_config.eval_steps,
            logging_steps=reward_config.logging_steps,
            
            # Checkpointing
            save_strategy="steps",
            save_steps=reward_config.save_steps,
            save_total_limit=3,
            
            # Memory optimization
            fp16=True,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            
            # Reproducibility
            seed=42,
            data_seed=42,
            
            # Misc
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            
            # Reporting
            report_to=["wandb"] if self.config.tracking.use_wandb else [],
            run_name=f"reward_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def train(self, sft_checkpoint_path: str) -> RewardTrainingResult:
        """
        Execute reward model training.
        
        Args:
            sft_checkpoint_path: Path to SFT checkpoint
            
        Returns:
            Training result with metrics and checkpoint path
            
        Requirement 1.2: Reward model training stage implementation
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting reward model training")
            
            # Prepare model and datasets
            self.reward_model = self.prepare_reward_model(sft_checkpoint_path)
            self.training_dataset, self.eval_dataset = self.prepare_datasets()
            
            # Create training arguments
            training_args = self.create_training_arguments()
            
            # Create trainer
            trainer = Trainer(
                model=self.reward_model,
                args=training_args,
                train_dataset=self.training_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.compute_metrics,
                tokenizer=self.tokenizer,
            )
            
            # Add custom callback
            callback = RewardTrainingCallback(
                self.experiment_tracker,
                self.model_manager,
                self.checkpoint_manager,
                self.config
            )
            trainer.add_callback(callback)
            
            # Execute training
            self.logger.info("Starting reward model training loop")
            train_result = trainer.train()
            
            # Get final metrics
            final_metrics = trainer.evaluate()
            
            # Save final checkpoint
            checkpoint_path = self._save_final_checkpoint(trainer.model, final_metrics)
            
            # Calculate training duration and memory usage
            duration = time.time() - start_time
            memory_peak = max(callback.memory_history) if callback.memory_history else 0.0
            
            # Create result
            result = RewardTrainingResult(
                success=True,
                checkpoint_path=checkpoint_path,
                metrics=final_metrics,
                duration_seconds=duration,
                memory_peak_gb=memory_peak,
                final_loss=final_metrics.get('eval_loss', 0.0),
                final_accuracy=final_metrics.get('eval_accuracy', 0.0),
                total_steps=train_result.global_step
            )
            
            self.logger.info(f"Reward training completed successfully in {duration:.2f} seconds")
            self.logger.info(f"Final accuracy: {result.final_accuracy:.4f}")
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            return result
            
        except Exception as e:
            error_msg = f"Reward training failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            return RewardTrainingResult(
                success=False,
                error_message=error_msg,
                duration_seconds=time.time() - start_time
            )
        
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _save_final_checkpoint(self, model: RewardModel, metrics: Dict[str, float]) -> str:
        """
        Save final reward model checkpoint.
        
        Args:
            model: Trained reward model
            metrics: Final evaluation metrics
            
        Returns:
            Path to saved checkpoint
        """
        try:
            checkpoint_name = f"reward_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save through checkpoint manager
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=None,
                epoch=0,  # Final checkpoint
                stage="reward",
                checkpoint_name=checkpoint_name,
                metadata={
                    'metrics': metrics,
                    'model_type': 'reward_model',
                    'training_completed': True
                }
            )
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save final checkpoint: {str(e)}")
            raise
    
    def evaluate_model(self, checkpoint_path: str, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate trained reward model.
        
        Args:
            checkpoint_path: Path to reward model checkpoint
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        try:
            self.logger.info(f"Evaluating reward model from {checkpoint_path}")
            
            # Load model
            model, _, _ = self.checkpoint_manager.load_checkpoint(
                Path(checkpoint_path).name
            )
            
            if model is None:
                raise ValueError(f"Could not load model from {checkpoint_path}")
            
            # Use provided dataset or default eval dataset
            if eval_dataset is None:
                if self.eval_dataset is None:
                    _, self.eval_dataset = self.prepare_datasets()
                eval_dataset = self.eval_dataset
            
            # Create trainer for evaluation
            training_args = TrainingArguments(
                output_dir="./tmp_eval",
                per_device_eval_batch_size=self.config.training.reward.eval_batch_size,
                remove_unused_columns=False,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics,
                tokenizer=self.tokenizer,
            )
            
            # Run evaluation
            eval_results = trainer.evaluate()
            
            self.logger.info("Reward model evaluation completed")
            for key, value in eval_results.items():
                self.logger.info(f"{key}: {value:.4f}")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Reward model evaluation failed: {str(e)}")
            raise


def create_reward_trainer(
    config: Config,
    model_manager: ModelManager,
    dataset_manager: DatasetManager,
    checkpoint_manager: CheckpointManager,
    experiment_tracker: ExperimentTracker
) -> RewardTrainer:
    """
    Factory function to create reward trainer.
    
    Args:
        config: Pipeline configuration
        model_manager: Model manager instance
        dataset_manager: Dataset manager instance
        checkpoint_manager: Checkpoint manager instance
        experiment_tracker: Experiment tracker instance
        
    Returns:
        Configured reward trainer
    """
    return RewardTrainer(
        config=config,
        model_manager=model_manager,
        dataset_manager=dataset_manager,
        checkpoint_manager=checkpoint_manager,
        experiment_tracker=experiment_tracker
    )