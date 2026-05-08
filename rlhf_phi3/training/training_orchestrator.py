"""
Training Orchestrator for RLHF Phi-3 Pipeline

This module implements the central orchestrator that coordinates the three-stage
RLHF training pipeline: Supervised Fine-Tuning (SFT) → Reward Model Training → 
Proximal Policy Optimization (PPO).

Key Features:
- Three-stage pipeline coordination with automatic progression
- Stage validation and transition management
- Failure state preservation and error diagnostics
- Memory monitoring and automatic optimization
- Training resumption and checkpoint integration

Requirements satisfied:
- 1.1: Three sequential training stages (SFT → Reward → PPO)
- 1.2: Automatic progression between stages
- 1.3: Automatic progression to PPO stage
- 1.4: Failure state preservation and error diagnostics
- 1.5: Stage validation before proceeding
- 5.5: Memory monitoring and reporting
- 9.4: Loss divergence handling with early stopping
"""

import os
import gc
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict
from enum import Enum

import torch
import numpy as np
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import PeftModel
from trl import SFTTrainer, RewardTrainer, PPOTrainer, PPOConfig

from ..config.config_manager import Config
from ..models.model_manager import ModelManager
from ..data.dataset_manager import DatasetManager
from ..checkpoints.checkpoint_manager import CheckpointManager
from ..tracking.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class TrainingStage(Enum):
    """Enumeration of training stages."""
    SFT = "sft"
    REWARD = "reward"
    PPO = "ppo"


@dataclass
class StageResult:
    """Result of a training stage execution."""
    stage: TrainingStage
    success: bool
    checkpoint_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    memory_peak_gb: Optional[float] = None


@dataclass
class PipelineState:
    """Current state of the training pipeline."""
    current_stage: Optional[TrainingStage] = None
    completed_stages: List[TrainingStage] = None
    stage_results: Dict[TrainingStage, StageResult] = None
    pipeline_start_time: Optional[datetime] = None
    last_checkpoint_path: Optional[str] = None
    failure_count: int = 0
    
    def __post_init__(self):
        if self.completed_stages is None:
            self.completed_stages = []
        if self.stage_results is None:
            self.stage_results = {}


class TrainingOrchestrator:
    """
    Central orchestrator for the three-stage RLHF training pipeline.
    
    This class coordinates the complete RLHF pipeline with proper sequencing,
    error handling, and state management. It integrates with all pipeline
    components to provide a unified training experience.
    
    Requirements satisfied:
    - 1.1: Three sequential training stages (SFT → Reward → PPO)
    - 1.2: Automatic progression between stages
    - 1.3: Automatic progression to PPO stage
    - 1.4: Failure state preservation and error diagnostics
    - 1.5: Stage validation before proceeding
    - 5.5: Memory monitoring and reporting
    - 9.4: Loss divergence handling with early stopping
    """
    
    def __init__(self, config: Config):
        """
        Initialize the Training Orchestrator.
        
        Args:
            config: Complete pipeline configuration
            
        Requirement 1.1: Three sequential training stages
        """
        self.config = config
        self.pipeline_state = PipelineState()
        
        # Initialize component managers
        self.model_manager = ModelManager(config)
        self.dataset_manager = DatasetManager(config)
        self.checkpoint_manager = CheckpointManager(
            base_path=config.paths.base_output_dir,
            enable_drive_sync=True
        )
        self.experiment_tracker = ExperimentTracker(
            project_name=config.wandb.project,
            config=config,
            entity=config.wandb.entity
        )
        
        # Training state tracking
        self.current_model = None
        self.current_tokenizer = None
        self.loss_history = []
        self.memory_usage_history = []
        
        # Early stopping configuration
        self.early_stopping_patience = 5
        self.loss_divergence_threshold = 2.0  # Factor increase that triggers early stopping
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.logging.level))
        logger.info("TrainingOrchestrator initialized")
    
    def run_full_pipeline(self) -> str:
        """
        Execute the complete three-stage RLHF pipeline.
        
        Returns:
            Path to the final trained model
            
        Requirements 1.1, 1.2, 1.3: Complete three-stage pipeline with automatic progression
        """
        logger.info("Starting complete RLHF pipeline execution")
        
        try:
            self.pipeline_state.pipeline_start_time = datetime.now(timezone.utc)
            
            # Stage 1: Supervised Fine-Tuning
            logger.info("=== Starting Stage 1: Supervised Fine-Tuning ===")
            sft_checkpoint = self.run_sft_stage()
            
            if not sft_checkpoint:
                raise RuntimeError("SFT stage failed - cannot proceed to reward training")
            
            # Stage 2: Reward Model Training
            logger.info("=== Starting Stage 2: Reward Model Training ===")
            reward_checkpoint = self.run_reward_stage(sft_checkpoint)
            
            if not reward_checkpoint:
                raise RuntimeError("Reward stage failed - cannot proceed to PPO training")
            
            # Stage 3: Proximal Policy Optimization
            logger.info("=== Starting Stage 3: Proximal Policy Optimization ===")
            final_checkpoint = self.run_ppo_stage(sft_checkpoint, reward_checkpoint)
            
            if not final_checkpoint:
                raise RuntimeError("PPO stage failed - pipeline incomplete")
            
            # Pipeline completed successfully
            total_duration = (datetime.now(timezone.utc) - self.pipeline_state.pipeline_start_time).total_seconds()
            
            logger.info(f"RLHF pipeline completed successfully in {total_duration:.2f} seconds")
            logger.info(f"Final model checkpoint: {final_checkpoint}")
            
            # Log pipeline summary
            self._log_pipeline_summary()
            
            return final_checkpoint
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            self._handle_pipeline_failure(e)
            raise
        finally:
            # Cleanup resources
            self._cleanup_resources()
    
    def run_sft_stage(self) -> Optional[str]:
        """
        Execute the Supervised Fine-Tuning stage.
        
        Returns:
            Path to SFT checkpoint if successful, None otherwise
            
        Requirement 1.1: SFT stage implementation
        Requirement 1.5: Stage validation before proceeding
        """
        stage = TrainingStage.SFT
        logger.info("Starting SFT (Supervised Fine-Tuning) stage")
        
        start_time = time.time()
        
        try:
            # Update pipeline state
            self.pipeline_state.current_stage = stage
            
            # Start experiment tracking
            self.experiment_tracker.start_run(
                stage=stage.value,
                run_name=f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Load and prepare model
            logger.info("Loading base model for SFT")
            base_model = self.model_manager.load_base_model()
            peft_model = self.model_manager.apply_peft(base_model)
            training_model = self.model_manager.prepare_for_training(peft_model)
            
            self.current_model = training_model
            self.current_tokenizer = self.model_manager.tokenizer
            
            # Load and preprocess SFT dataset
            logger.info("Loading SFT dataset")
            sft_dataset = self.dataset_manager.load_sft_dataset(streaming=False)
            processed_dataset = self.dataset_manager.preprocess_sft_data(sft_dataset)
            
            # Validate dataset
            if not self._validate_sft_dataset(processed_dataset):
                raise ValueError("SFT dataset validation failed")
            
            # Setup training arguments
            training_args = self._create_sft_training_arguments()
            
            # Create trainer
            trainer = SFTTrainer(
                model=training_model,
                tokenizer=self.current_tokenizer,
                args=training_args,
                train_dataset=processed_dataset,
                dataset_text_field="text",
                max_seq_length=self.config.model.max_length,
                packing=False,  # Disable packing for stability
            )
            
            # Add training callbacks
            trainer.add_callback(self._create_training_callback(stage))
            
            # Execute training with monitoring
            logger.info("Starting SFT training")
            train_result = trainer.train()
            
            # Validate training completion
            if not self._validate_stage_completion(stage, train_result):
                raise RuntimeError("SFT stage validation failed")
            
            # Save checkpoint
            checkpoint_path = self._save_stage_checkpoint(trainer.model, trainer.optimizer, stage, train_result)
            
            # Record stage result
            duration = time.time() - start_time
            memory_peak = self._get_peak_memory_usage()
            
            stage_result = StageResult(
                stage=stage,
                success=True,
                checkpoint_path=checkpoint_path,
                metrics=train_result.metrics if hasattr(train_result, 'metrics') else {},
                duration_seconds=duration,
                memory_peak_gb=memory_peak
            )
            
            self.pipeline_state.stage_results[stage] = stage_result
            self.pipeline_state.completed_stages.append(stage)
            self.pipeline_state.last_checkpoint_path = checkpoint_path
            
            logger.info(f"SFT stage completed successfully in {duration:.2f} seconds")
            logger.info(f"SFT checkpoint saved: {checkpoint_path}")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"SFT stage failed: {str(e)}")
            self._handle_stage_failure(stage, e, time.time() - start_time)
            return None
        finally:
            self.experiment_tracker.finish_run()
    
    def run_reward_stage(self, sft_checkpoint: str) -> Optional[str]:
        """
        Execute the Reward Model Training stage.
        
        Args:
            sft_checkpoint: Path to the SFT checkpoint
            
        Returns:
            Path to reward model checkpoint if successful, None otherwise
            
        Requirement 1.2: Automatic progression between stages
        Requirement 1.5: Stage validation before proceeding
        """
        stage = TrainingStage.REWARD
        logger.info("Starting Reward Model Training stage")
        
        start_time = time.time()
        
        try:
            # Update pipeline state
            self.pipeline_state.current_stage = stage
            
            # Start experiment tracking
            self.experiment_tracker.start_run(
                stage=stage.value,
                run_name=f"reward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Load SFT model as base for reward model
            logger.info(f"Loading SFT model from {sft_checkpoint}")
            sft_model, _, _ = self.checkpoint_manager.load_checkpoint(
                Path(sft_checkpoint).name
            )
            
            if sft_model is None:
                # Fallback: load from path directly
                from peft import PeftModel
                base_model = self.model_manager.load_base_model()
                sft_model = PeftModel.from_pretrained(base_model, sft_checkpoint)
            
            # Prepare model for reward training
            reward_model = self.model_manager.prepare_for_training(sft_model)
            self.current_model = reward_model
            
            # Load and preprocess preference dataset
            logger.info("Loading preference dataset")
            preference_dataset = self.dataset_manager.load_preference_dataset(streaming=False)
            processed_dataset = self.dataset_manager.preprocess_preference_data(preference_dataset)
            
            # Validate dataset
            if not self._validate_preference_dataset(processed_dataset):
                raise ValueError("Preference dataset validation failed")
            
            # Setup training arguments
            training_args = self._create_reward_training_arguments()
            
            # Create reward trainer
            trainer = RewardTrainer(
                model=reward_model,
                tokenizer=self.current_tokenizer,
                args=training_args,
                train_dataset=processed_dataset,
                max_length=self.config.model.max_length,
            )
            
            # Add training callbacks
            trainer.add_callback(self._create_training_callback(stage))
            
            # Execute training with monitoring
            logger.info("Starting reward model training")
            train_result = trainer.train()
            
            # Validate training completion
            if not self._validate_stage_completion(stage, train_result):
                raise RuntimeError("Reward stage validation failed")
            
            # Save checkpoint
            checkpoint_path = self._save_stage_checkpoint(trainer.model, trainer.optimizer, stage, train_result)
            
            # Record stage result
            duration = time.time() - start_time
            memory_peak = self._get_peak_memory_usage()
            
            stage_result = StageResult(
                stage=stage,
                success=True,
                checkpoint_path=checkpoint_path,
                metrics=train_result.metrics if hasattr(train_result, 'metrics') else {},
                duration_seconds=duration,
                memory_peak_gb=memory_peak
            )
            
            self.pipeline_state.stage_results[stage] = stage_result
            self.pipeline_state.completed_stages.append(stage)
            self.pipeline_state.last_checkpoint_path = checkpoint_path
            
            logger.info(f"Reward stage completed successfully in {duration:.2f} seconds")
            logger.info(f"Reward checkpoint saved: {checkpoint_path}")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Reward stage failed: {str(e)}")
            self._handle_stage_failure(stage, e, time.time() - start_time)
            return None
        finally:
            self.experiment_tracker.finish_run()
    
    def run_ppo_stage(self, sft_checkpoint: str, reward_checkpoint: str) -> Optional[str]:
        """
        Execute the Proximal Policy Optimization stage.
        
        Args:
            sft_checkpoint: Path to the SFT checkpoint
            reward_checkpoint: Path to the reward model checkpoint
            
        Returns:
            Path to final PPO checkpoint if successful, None otherwise
            
        Requirement 1.3: Automatic progression to PPO stage
        Requirement 1.5: Stage validation before proceeding
        """
        stage = TrainingStage.PPO
        logger.info("Starting PPO (Proximal Policy Optimization) stage")
        
        start_time = time.time()
        
        try:
            # Update pipeline state
            self.pipeline_state.current_stage = stage
            
            # Start experiment tracking
            self.experiment_tracker.start_run(
                stage=stage.value,
                run_name=f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Load SFT model as policy model
            logger.info(f"Loading SFT model from {sft_checkpoint}")
            sft_model, _, _ = self.checkpoint_manager.load_checkpoint(
                Path(sft_checkpoint).name
            )
            
            if sft_model is None:
                from peft import PeftModel
                base_model = self.model_manager.load_base_model()
                sft_model = PeftModel.from_pretrained(base_model, sft_checkpoint)
            
            # Load reward model
            logger.info(f"Loading reward model from {reward_checkpoint}")
            reward_model, _, _ = self.checkpoint_manager.load_checkpoint(
                Path(reward_checkpoint).name
            )
            
            if reward_model is None:
                from peft import PeftModel
                base_model = self.model_manager.load_base_model()
                reward_model = PeftModel.from_pretrained(base_model, reward_checkpoint)
            
            # Prepare models for PPO training
            policy_model = self.model_manager.prepare_for_training(sft_model)
            self.current_model = policy_model
            
            # Load dataset for PPO (typically prompts from SFT dataset)
            logger.info("Loading dataset for PPO")
            sft_dataset = self.dataset_manager.load_sft_dataset(streaming=False)
            
            # Extract prompts for PPO training
            ppo_dataset = self._prepare_ppo_dataset(sft_dataset)
            
            # Validate dataset
            if not self._validate_ppo_dataset(ppo_dataset):
                raise ValueError("PPO dataset validation failed")
            
            # Setup PPO configuration
            ppo_config = self._create_ppo_config()
            
            # Create PPO trainer
            trainer = PPOTrainer(
                config=ppo_config,
                model=policy_model,
                ref_model=None,  # Will use the same model as reference
                reward_model=reward_model,
                tokenizer=self.current_tokenizer,
                dataset=ppo_dataset,
            )
            
            # Execute PPO training with monitoring
            logger.info("Starting PPO training")
            
            # PPO training loop with custom monitoring
            for epoch in range(self.config.training.ppo.ppo_epochs):
                logger.info(f"PPO Epoch {epoch + 1}/{self.config.training.ppo.ppo_epochs}")
                
                for batch_idx, batch in enumerate(trainer.dataloader):
                    if batch_idx >= self.config.training.ppo.max_steps:
                        break
                    
                    # Generate responses
                    query_tensors = batch["input_ids"]
                    response_tensors = trainer.generate(
                        query_tensors,
                        return_prompt=False,
                        **trainer.generation_kwargs
                    )
                    
                    # Get rewards
                    rewards = trainer.compute_rewards(query_tensors, response_tensors)
                    
                    # PPO step
                    stats = trainer.step(query_tensors, response_tensors, rewards)
                    
                    # Log metrics
                    if batch_idx % self.config.logging.log_steps == 0:
                        self.experiment_tracker.log_metrics(stats, batch_idx)
                        
                        # Monitor for loss divergence
                        if self._check_loss_divergence(stats):
                            logger.warning("Loss divergence detected - implementing early stopping")
                            break
                    
                    # Memory monitoring
                    self._monitor_memory_usage(batch_idx)
            
            # Save final checkpoint
            checkpoint_path = self._save_stage_checkpoint(
                trainer.model, 
                None,  # PPO trainer doesn't expose optimizer directly
                stage, 
                {"final_epoch": self.config.training.ppo.ppo_epochs}
            )
            
            # Validate training completion
            if not self._validate_stage_completion(stage, {"checkpoint_path": checkpoint_path}):
                raise RuntimeError("PPO stage validation failed")
            
            # Record stage result
            duration = time.time() - start_time
            memory_peak = self._get_peak_memory_usage()
            
            stage_result = StageResult(
                stage=stage,
                success=True,
                checkpoint_path=checkpoint_path,
                metrics={"training_duration": duration},
                duration_seconds=duration,
                memory_peak_gb=memory_peak
            )
            
            self.pipeline_state.stage_results[stage] = stage_result
            self.pipeline_state.completed_stages.append(stage)
            self.pipeline_state.last_checkpoint_path = checkpoint_path
            
            logger.info(f"PPO stage completed successfully in {duration:.2f} seconds")
            logger.info(f"Final checkpoint saved: {checkpoint_path}")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"PPO stage failed: {str(e)}")
            self._handle_stage_failure(stage, e, time.time() - start_time)
            return None
        finally:
            self.experiment_tracker.finish_run()
    
    def resume_from_stage(self, stage: str, checkpoint_path: str) -> str:
        """
        Resume training from a specific stage and checkpoint.
        
        Args:
            stage: Stage to resume from ('sft', 'reward', or 'ppo')
            checkpoint_path: Path to the checkpoint to resume from
            
        Returns:
            Path to the final trained model
            
        Requirement 1.4: Training resumption capabilities
        """
        logger.info(f"Resuming training from {stage} stage with checkpoint {checkpoint_path}")
        
        try:
            stage_enum = TrainingStage(stage)
            
            # Update pipeline state
            self.pipeline_state.current_stage = stage_enum
            
            # Load checkpoint
            model_path, optimizer_state, metadata = self.checkpoint_manager.load_checkpoint(
                Path(checkpoint_path).name
            )
            
            if model_path is None:
                raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")
            
            # Resume based on stage
            if stage_enum == TrainingStage.SFT:
                return self.run_full_pipeline()
            elif stage_enum == TrainingStage.REWARD:
                # Need SFT checkpoint - assume it's the previous stage
                sft_checkpoint = self._find_previous_checkpoint(TrainingStage.SFT)
                if not sft_checkpoint:
                    raise RuntimeError("Cannot find SFT checkpoint for reward stage resume")
                
                reward_checkpoint = self.run_reward_stage(sft_checkpoint)
                if reward_checkpoint:
                    return self.run_ppo_stage(sft_checkpoint, reward_checkpoint)
                else:
                    raise RuntimeError("Reward stage failed during resume")
            elif stage_enum == TrainingStage.PPO:
                # Need both SFT and reward checkpoints
                sft_checkpoint = self._find_previous_checkpoint(TrainingStage.SFT)
                reward_checkpoint = self._find_previous_checkpoint(TrainingStage.REWARD)
                
                if not sft_checkpoint or not reward_checkpoint:
                    raise RuntimeError("Cannot find required checkpoints for PPO stage resume")
                
                return self.run_ppo_stage(sft_checkpoint, reward_checkpoint)
            
        except Exception as e:
            logger.error(f"Failed to resume from stage {stage}: {str(e)}")
            self._handle_pipeline_failure(e)
            raise
    
    def validate_stage_completion(self, stage: str) -> bool:
        """
        Validate successful completion of a training stage.
        
        Args:
            stage: Stage to validate ('sft', 'reward', or 'ppo')
            
        Returns:
            True if stage completed successfully, False otherwise
            
        Requirement 1.5: Stage validation before proceeding
        """
        try:
            stage_enum = TrainingStage(stage)
            
            # Check if stage is in completed stages
            if stage_enum not in self.pipeline_state.completed_stages:
                logger.warning(f"Stage {stage} not found in completed stages")
                return False
            
            # Check stage result
            if stage_enum not in self.pipeline_state.stage_results:
                logger.warning(f"No result found for stage {stage}")
                return False
            
            stage_result = self.pipeline_state.stage_results[stage_enum]
            
            # Validate stage result
            if not stage_result.success:
                logger.warning(f"Stage {stage} marked as failed")
                return False
            
            # Validate checkpoint exists
            if not stage_result.checkpoint_path:
                logger.warning(f"No checkpoint path for stage {stage}")
                return False
            
            checkpoint_path = Path(stage_result.checkpoint_path)
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
                return False
            
            logger.info(f"Stage {stage} validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Stage validation failed for {stage}: {str(e)}")
            return False
    
    # Helper methods for training orchestration
    
    def _validate_sft_dataset(self, dataset) -> bool:
        """Validate SFT dataset format and content."""
        try:
            if len(dataset) == 0:
                logger.error("SFT dataset is empty")
                return False
            
            # Check required fields
            required_fields = ['input_ids', 'attention_mask', 'labels']
            for field in required_fields:
                if field not in dataset.column_names:
                    logger.error(f"Missing required field in SFT dataset: {field}")
                    return False
            
            # Sample validation
            sample = dataset[0]
            for field in required_fields:
                if sample[field] is None or len(sample[field]) == 0:
                    logger.error(f"Empty {field} in SFT dataset sample")
                    return False
            
            logger.info(f"SFT dataset validation passed: {len(dataset)} samples")
            return True
            
        except Exception as e:
            logger.error(f"SFT dataset validation failed: {str(e)}")
            return False
    
    def _validate_preference_dataset(self, dataset) -> bool:
        """Validate preference dataset format and content."""
        try:
            if len(dataset) == 0:
                logger.error("Preference dataset is empty")
                return False
            
            # Check required fields
            required_fields = ['chosen_input_ids', 'chosen_attention_mask', 
                             'rejected_input_ids', 'rejected_attention_mask']
            for field in required_fields:
                if field not in dataset.column_names:
                    logger.error(f"Missing required field in preference dataset: {field}")
                    return False
            
            # Sample validation
            sample = dataset[0]
            for field in required_fields:
                if sample[field] is None or len(sample[field]) == 0:
                    logger.error(f"Empty {field} in preference dataset sample")
                    return False
            
            logger.info(f"Preference dataset validation passed: {len(dataset)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Preference dataset validation failed: {str(e)}")
            return False
    
    def _validate_ppo_dataset(self, dataset) -> bool:
        """Validate PPO dataset format and content."""
        try:
            if len(dataset) == 0:
                logger.error("PPO dataset is empty")
                return False
            
            # Check required fields
            required_fields = ['input_ids', 'attention_mask']
            for field in required_fields:
                if field not in dataset.column_names:
                    logger.error(f"Missing required field in PPO dataset: {field}")
                    return False
            
            logger.info(f"PPO dataset validation passed: {len(dataset)} samples")
            return True
            
        except Exception as e:
            logger.error(f"PPO dataset validation failed: {str(e)}")
            return False
    
    def _create_sft_training_arguments(self) -> TrainingArguments:
        """Create training arguments for SFT stage."""
        output_dir = Path(self.config.paths.base_output_dir) / "sft_training"
        
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.sft.epochs,
            per_device_train_batch_size=self.config.training.sft.batch_size,
            gradient_accumulation_steps=self.config.training.sft.gradient_accumulation_steps,
            learning_rate=self.config.training.sft.learning_rate,
            warmup_steps=self.config.training.sft.warmup_steps,
            max_steps=self.config.training.sft.max_steps,
            logging_steps=self.config.logging.log_steps,
            save_steps=self.config.checkpointing.save_steps,
            eval_steps=self.config.logging.eval_steps,
            save_total_limit=self.config.checkpointing.save_total_limit,
            load_best_model_at_end=False,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.config.optimization.fp16,
            gradient_checkpointing=self.config.optimization.gradient_checkpointing,
            dataloader_num_workers=self.config.optimization.dataloader_num_workers,
            optim=self.config.optimization.optimizer_type,
            lr_scheduler_type=self.config.optimization.scheduler_type,
            weight_decay=self.config.optimization.weight_decay,
            max_grad_norm=self.config.optimization.max_grad_norm,
            report_to="wandb" if self.config.wandb.project else "none",
            run_name=f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            remove_unused_columns=False,
        )
    
    def _create_reward_training_arguments(self) -> TrainingArguments:
        """Create training arguments for reward model stage."""
        output_dir = Path(self.config.paths.base_output_dir) / "reward_training"
        
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.reward.epochs,
            per_device_train_batch_size=self.config.training.reward.batch_size,
            gradient_accumulation_steps=self.config.training.reward.gradient_accumulation_steps,
            learning_rate=self.config.training.reward.learning_rate,
            warmup_steps=self.config.training.reward.warmup_steps,
            max_steps=self.config.training.reward.max_steps,
            logging_steps=self.config.logging.log_steps,
            save_steps=self.config.checkpointing.save_steps,
            eval_steps=self.config.logging.eval_steps,
            save_total_limit=self.config.checkpointing.save_total_limit,
            load_best_model_at_end=False,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.config.optimization.fp16,
            gradient_checkpointing=self.config.optimization.gradient_checkpointing,
            dataloader_num_workers=self.config.optimization.dataloader_num_workers,
            optim=self.config.optimization.optimizer_type,
            lr_scheduler_type=self.config.optimization.scheduler_type,
            weight_decay=self.config.optimization.weight_decay,
            max_grad_norm=self.config.optimization.max_grad_norm,
            report_to="wandb" if self.config.wandb.project else "none",
            run_name=f"reward_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            remove_unused_columns=False,
        )
    
    def _create_ppo_config(self) -> PPOConfig:
        """Create PPO configuration for PPO stage."""
        return PPOConfig(
            model_name=self.config.model.name,
            learning_rate=self.config.training.ppo.learning_rate,
            batch_size=self.config.training.ppo.batch_size,
            mini_batch_size=self.config.training.ppo.mini_batch_size,
            gradient_accumulation_steps=self.config.training.ppo.gradient_accumulation_steps,
            ppo_epochs=self.config.training.ppo.ppo_epochs,
            max_grad_norm=self.config.optimization.max_grad_norm,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            seed=42,
        )
    
    def _prepare_ppo_dataset(self, sft_dataset):
        """Prepare dataset for PPO training by extracting prompts."""
        try:
            # Extract prompts from SFT dataset
            prompts = []
            
            for example in sft_dataset:
                # Extract the user prompt from the formatted text
                if 'text' in example:
                    text = example['text']
                    # Simple extraction - find user message
                    if '<|user|>' in text:
                        user_start = text.find('<|user|>') + len('<|user|>')
                        user_end = text.find('<|end|>', user_start)
                        if user_end > user_start:
                            prompt = text[user_start:user_end].strip()
                            prompts.append(prompt)
                elif 'messages' in example:
                    # Extract from messages format
                    messages = example['messages']
                    for msg in messages:
                        if msg.get('role') == 'user':
                            prompts.append(msg['content'])
                            break
            
            # Tokenize prompts
            tokenized_prompts = []
            for prompt in prompts[:1000]:  # Limit for memory efficiency
                tokens = self.current_tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.config.model.max_length // 2,  # Leave room for generation
                    return_tensors="pt"
                )
                tokenized_prompts.append({
                    'input_ids': tokens['input_ids'].squeeze(),
                    'attention_mask': tokens['attention_mask'].squeeze(),
                    'text': prompt
                })
            
            # Convert to dataset format expected by PPO trainer
            from datasets import Dataset
            return Dataset.from_list(tokenized_prompts)
            
        except Exception as e:
            logger.error(f"Failed to prepare PPO dataset: {str(e)}")
            raise
    
    def _create_training_callback(self, stage: TrainingStage):
        """Create training callback for monitoring and early stopping."""
        from transformers import TrainerCallback
        
        class OrchestrationCallback(TrainerCallback):
            def __init__(self, orchestrator, stage):
                self.orchestrator = orchestrator
                self.stage = stage
                self.best_loss = float('inf')
                self.patience_counter = 0
            
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if logs:
                    # Monitor memory usage
                    self.orchestrator._monitor_memory_usage(state.global_step)
                    
                    # Log metrics to experiment tracker
                    self.orchestrator.experiment_tracker.log_metrics(logs, state.global_step, self.stage.value)
                    
                    # Check for loss divergence
                    if 'train_loss' in logs:
                        current_loss = logs['train_loss']
                        self.orchestrator.loss_history.append(current_loss)
                        
                        # Early stopping check
                        if current_loss < self.best_loss:
                            self.best_loss = current_loss
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                        
                        # Check for divergence
                        if self.orchestrator._check_loss_divergence({'train_loss': current_loss}):
                            logger.warning("Loss divergence detected - stopping training")
                            control.should_training_stop = True
            
            def on_save(self, args, state, control, model=None, **kwargs):
                # Log checkpoint to experiment tracker
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                self.orchestrator.experiment_tracker.log_model_checkpoint(
                    checkpoint_path, 
                    self.stage.value,
                    {"step": state.global_step, "epoch": state.epoch}
                )
        
        return OrchestrationCallback(self, stage)
    
    def _validate_stage_completion(self, stage: TrainingStage, train_result) -> bool:
        """Validate that a training stage completed successfully."""
        try:
            # Check if training completed without errors
            if hasattr(train_result, 'metrics'):
                metrics = train_result.metrics
                
                # Check for reasonable loss values
                if 'train_loss' in metrics:
                    final_loss = metrics['train_loss']
                    if np.isnan(final_loss) or np.isinf(final_loss):
                        logger.error(f"Invalid final loss for {stage.value}: {final_loss}")
                        return False
                    
                    if final_loss > 100.0:  # Unreasonably high loss
                        logger.warning(f"High final loss for {stage.value}: {final_loss}")
                
                # Check training steps
                if 'train_steps_per_second' in metrics:
                    if metrics['train_steps_per_second'] <= 0:
                        logger.error(f"Invalid training speed for {stage.value}")
                        return False
            
            logger.info(f"Stage {stage.value} validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Stage validation failed for {stage.value}: {str(e)}")
            return False
    
    def _save_stage_checkpoint(self, model, optimizer, stage: TrainingStage, train_result) -> str:
        """Save checkpoint for a completed training stage."""
        try:
            # Extract metrics
            metrics = {}
            if hasattr(train_result, 'metrics'):
                metrics = train_result.metrics
            elif isinstance(train_result, dict):
                metrics = train_result
            
            # Save checkpoint
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=metrics.get('epoch', 0),
                step=metrics.get('global_step', 0),
                stage=stage.value,
                metrics=metrics,
                config_hash=str(hash(str(asdict(self.config))))
            )
            
            checkpoint_path = self.checkpoint_manager.base_path / checkpoint_id
            logger.info(f"Saved {stage.value} checkpoint: {checkpoint_path}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save {stage.value} checkpoint: {str(e)}")
            raise
    
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
            self.memory_usage_history.append(memory_entry)
            
            # Log to experiment tracker
            if step % (self.config.logging.log_steps * 5) == 0:  # Log less frequently
                self.experiment_tracker.log_metrics(
                    {f"memory/{k}": v for k, v in memory_stats.items()},
                    step
                )
            
            # Check for memory issues
            if 'gpu_utilization' in memory_stats:
                if memory_stats['gpu_utilization'] > 0.95:
                    logger.warning(f"High GPU memory usage: {memory_stats['gpu_utilization']:.1%}")
                    
                    # Trigger memory optimization
                    self.model_manager.handle_memory_exhaustion()
            
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {str(e)}")
    
    def _check_loss_divergence(self, metrics: Dict[str, float]) -> bool:
        """Check if training loss is diverging."""
        try:
            if 'train_loss' not in metrics:
                return False
            
            current_loss = metrics['train_loss']
            
            # Need at least some history to check divergence
            if len(self.loss_history) < 10:
                return False
            
            # Calculate recent average
            recent_losses = self.loss_history[-10:]
            recent_avg = np.mean(recent_losses)
            
            # Check for sudden increase
            if current_loss > recent_avg * self.loss_divergence_threshold:
                logger.warning(f"Loss divergence detected: {current_loss:.4f} vs recent avg {recent_avg:.4f}")
                return True
            
            # Check for NaN or inf
            if np.isnan(current_loss) or np.isinf(current_loss):
                logger.error(f"Invalid loss value: {current_loss}")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Loss divergence check failed: {str(e)}")
            return False
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage from history."""
        try:
            if not self.memory_usage_history:
                return 0.0
            
            gpu_usage = [entry.get('gpu_allocated_gb', 0) for entry in self.memory_usage_history]
            return max(gpu_usage) if gpu_usage else 0.0
            
        except Exception:
            return 0.0
    
    def _find_previous_checkpoint(self, stage: TrainingStage) -> Optional[str]:
        """Find the most recent checkpoint for a given stage."""
        try:
            checkpoints = self.checkpoint_manager.list_checkpoints(stage.value)
            return checkpoints[0] if checkpoints else None
        except Exception:
            return None
    
    def _handle_stage_failure(self, stage: TrainingStage, error: Exception, duration: float) -> None:
        """Handle failure of a training stage."""
        logger.error(f"Stage {stage.value} failed after {duration:.2f} seconds: {str(error)}")
        
        # Record failure in pipeline state
        stage_result = StageResult(
            stage=stage,
            success=False,
            error_message=str(error),
            duration_seconds=duration,
            memory_peak_gb=self._get_peak_memory_usage()
        )
        
        self.pipeline_state.stage_results[stage] = stage_result
        self.pipeline_state.failure_count += 1
        
        # Log detailed error information
        error_details = {
            'stage': stage.value,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'duration_seconds': duration,
            'memory_peak_gb': stage_result.memory_peak_gb,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save error details to file
        error_file = Path(self.config.paths.base_output_dir) / f"error_{stage.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(error_file, 'w') as f:
            json.dump(error_details, f, indent=2)
        
        logger.info(f"Error details saved to: {error_file}")
        
        # Log to experiment tracker if available
        try:
            self.experiment_tracker.log_metrics(
                {
                    f"{stage.value}/error": 1,
                    f"{stage.value}/duration_seconds": duration,
                    f"{stage.value}/memory_peak_gb": stage_result.memory_peak_gb or 0
                },
                step=0
            )
        except Exception:
            pass  # Don't fail on logging errors
    
    def _handle_pipeline_failure(self, error: Exception) -> None:
        """Handle overall pipeline failure."""
        logger.error(f"Pipeline execution failed: {str(error)}")
        
        # Save pipeline state
        pipeline_state_file = Path(self.config.paths.base_output_dir) / f"pipeline_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        pipeline_state_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert pipeline state to serializable format
            state_dict = {
                'current_stage': self.pipeline_state.current_stage.value if self.pipeline_state.current_stage else None,
                'completed_stages': [s.value for s in self.pipeline_state.completed_stages],
                'stage_results': {
                    s.value: asdict(result) for s, result in self.pipeline_state.stage_results.items()
                },
                'pipeline_start_time': self.pipeline_state.pipeline_start_time.isoformat() if self.pipeline_state.pipeline_start_time else None,
                'last_checkpoint_path': self.pipeline_state.last_checkpoint_path,
                'failure_count': self.pipeline_state.failure_count,
                'error_message': str(error),
                'traceback': traceback.format_exc()
            }
            
            with open(pipeline_state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
            
            logger.info(f"Pipeline state saved to: {pipeline_state_file}")
            
        except Exception as save_error:
            logger.error(f"Failed to save pipeline state: {str(save_error)}")
    
    def _log_pipeline_summary(self) -> None:
        """Log a summary of the completed pipeline."""
        try:
            total_duration = 0
            total_memory_peak = 0
            
            summary = {
                'pipeline_completed': True,
                'total_stages': len(self.pipeline_state.completed_stages),
                'completed_stages': [s.value for s in self.pipeline_state.completed_stages],
                'failure_count': self.pipeline_state.failure_count,
                'start_time': self.pipeline_state.pipeline_start_time.isoformat() if self.pipeline_state.pipeline_start_time else None,
                'end_time': datetime.now(timezone.utc).isoformat(),
            }
            
            # Add stage details
            for stage, result in self.pipeline_state.stage_results.items():
                stage_key = f"{stage.value}_stage"
                summary[stage_key] = {
                    'success': result.success,
                    'duration_seconds': result.duration_seconds,
                    'memory_peak_gb': result.memory_peak_gb,
                    'checkpoint_path': result.checkpoint_path,
                    'metrics_count': len(result.metrics) if result.metrics else 0
                }
                
                if result.duration_seconds:
                    total_duration += result.duration_seconds
                if result.memory_peak_gb:
                    total_memory_peak = max(total_memory_peak, result.memory_peak_gb)
            
            summary['total_duration_seconds'] = total_duration
            summary['peak_memory_gb'] = total_memory_peak
            
            # Save summary
            summary_file = Path(self.config.paths.base_output_dir) / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("=== PIPELINE SUMMARY ===")
            logger.info(f"Total duration: {total_duration:.2f} seconds")
            logger.info(f"Peak memory usage: {total_memory_peak:.2f} GB")
            logger.info(f"Completed stages: {len(self.pipeline_state.completed_stages)}")
            logger.info(f"Final checkpoint: {self.pipeline_state.last_checkpoint_path}")
            logger.info(f"Summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to log pipeline summary: {str(e)}")
    
    def _cleanup_resources(self) -> None:
        """Clean up resources and free memory."""
        try:
            logger.info("Cleaning up orchestrator resources")
            
            # Clear model references
            if self.current_model is not None:
                del self.current_model
                self.current_model = None
            
            if self.current_tokenizer is not None:
                del self.current_tokenizer
                self.current_tokenizer = None
            
            # Clear history
            self.loss_history.clear()
            self.memory_usage_history.clear()
            
            # Cleanup model manager
            if self.model_manager:
                self.model_manager.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.warning(f"Resource cleanup failed: {str(e)}")