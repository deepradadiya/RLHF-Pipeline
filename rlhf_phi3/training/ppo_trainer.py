"""
PPO (Proximal Policy Optimization) Trainer for RLHF Phi-3 Pipeline

This module implements the third and final stage of the RLHF pipeline: PPO training.
It handles RLHF training using TRL library integration with policy optimization
guided by the reward model.

Key Features:
- PPO training with TRL library integration
- Policy optimization with reward model guidance
- PEFT/LoRA integration for memory efficiency
- Progress tracking and experiment logging
- Checkpoint persistence and recovery

Requirements satisfied:
- 1.3: PPO stage implementation as final stage of three-stage pipeline
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
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import Dataset

from ..config.config_manager import Config
from ..models.model_manager import ModelManager
from ..data.dataset_manager import DatasetManager
from ..checkpoints.checkpoint_manager import CheckpointManager
from ..tracking.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class PPOTrainingResult:
    """Result of PPO training execution."""
    success: bool
    checkpoint_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    final_reward: Optional[float] = None
    final_kl_divergence: Optional[float] = None
    total_steps: Optional[int] = None


class PPOTrainingCallback:
    """Custom callback for PPO training monitoring and logging."""
    
    def __init__(
        self, 
        experiment_tracker: ExperimentTracker,
        model_manager: ModelManager,
        checkpoint_manager: CheckpointManager,
        config: Config
    ):
        """
        Initialize PPO training callback.
        
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
        self.best_reward = float('-inf')
        self.reward_history = []
        self.kl_history = []
        self.memory_history = []
        self.step_count = 0
        
        # Early stopping configuration
        self.early_stopping_patience = getattr(config.training.ppo, 'early_stopping_patience', 10)
        self.min_reward_improvement = getattr(config.training.ppo, 'min_reward_improvement', 0.01)
        self.patience_counter = 0
    
    def on_step_end(self, trainer: PPOTrainer, step: int, logs: Dict[str, float]):
        """Handle step end events."""
        self.step_count = step
        
        # Track memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            self.memory_history.append(memory_used)
            logs['memory_gb'] = memory_used
        
        # Track rewards and KL divergence
        if 'env/reward_mean' in logs:
            current_reward = logs['env/reward_mean']
            self.reward_history.append(current_reward)
            
            # Check for improvement
            if current_reward > self.best_reward + self.min_reward_improvement:
                self.best_reward = current_reward
                self.patience_counter = 0
                logger.info(f"New best reward: {self.best_reward:.4f}")
            else:
                self.patience_counter += 1
        
        if 'objective/kl' in logs:
            self.kl_history.append(logs['objective/kl'])
        
        # Log to experiment tracker
        self.experiment_tracker.log_metrics(logs, step)
        
        # Log progress
        if step % 10 == 0:
            logger.info(f"PPO Step {step}: Reward={logs.get('env/reward_mean', 0):.4f}, "
                       f"KL={logs.get('objective/kl', 0):.4f}")
    
    def should_stop_training(self) -> bool:
        """Check if training should stop early."""
        return self.patience_counter >= self.early_stopping_patience
    
    def save_checkpoint(self, trainer: PPOTrainer, step: int):
        """Save checkpoint during training."""
        try:
            checkpoint_name = f"ppo_step_{step}"
            self.checkpoint_manager.save_checkpoint(
                model=trainer.model,
                optimizer=None,  # PPO handles optimizer internally
                epoch=0,
                stage="ppo",
                checkpoint_name=checkpoint_name,
                metadata={
                    'step': step,
                    'best_reward': self.best_reward,
                    'model_type': 'ppo_model'
                }
            )
            logger.info(f"PPO checkpoint saved: {checkpoint_name}")
        except Exception as e:
            logger.error(f"Failed to save PPO checkpoint: {str(e)}")


class PPOTrainerWrapper:
    """
    PPO Trainer for RLHF Pipeline.
    
    Implements the final stage of RLHF using Proximal Policy Optimization
    to fine-tune the policy model using rewards from the reward model.
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
        Initialize PPO trainer.
        
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
        
        # Training components
        self.policy_model = None
        self.reward_model = None
        self.tokenizer = None
        self.ppo_trainer = None
        self.dataset = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_policy_model(self, sft_checkpoint_path: str) -> AutoModelForCausalLMWithValueHead:
        """
        Prepare policy model from SFT checkpoint.
        
        Args:
            sft_checkpoint_path: Path to SFT checkpoint
            
        Returns:
            Policy model with value head
            
        Requirement 1.3: Load SFT model as base for PPO training
        """
        self.logger.info(f"Loading SFT model for PPO from {sft_checkpoint_path}")
        
        try:
            # Load SFT model
            sft_model, _, _ = self.checkpoint_manager.load_checkpoint(
                Path(sft_checkpoint_path).name
            )
            
            if sft_model is None:
                # Fallback: load from path directly
                base_model = self.model_manager.load_base_model()
                sft_model = PeftModel.from_pretrained(base_model, sft_checkpoint_path)
            
            # Create policy model with value head
            policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                sft_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Get tokenizer
            self.tokenizer = self.model_manager.tokenizer
            
            self.logger.info("Policy model prepared successfully")
            return policy_model
            
        except Exception as e:
            self.logger.error(f"Failed to prepare policy model: {str(e)}")
            raise
    
    def prepare_reward_model(self, reward_checkpoint_path: str):
        """
        Prepare reward model from checkpoint.
        
        Args:
            reward_checkpoint_path: Path to reward model checkpoint
            
        Returns:
            Loaded reward model
        """
        self.logger.info(f"Loading reward model from {reward_checkpoint_path}")
        
        try:
            # Load reward model
            reward_model, _, _ = self.checkpoint_manager.load_checkpoint(
                Path(reward_checkpoint_path).name
            )
            
            if reward_model is None:
                raise ValueError(f"Could not load reward model from {reward_checkpoint_path}")
            
            # Set to evaluation mode
            reward_model.eval()
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            reward_model = reward_model.to(device)
            
            self.logger.info("Reward model prepared successfully")
            return reward_model
            
        except Exception as e:
            self.logger.error(f"Failed to prepare reward model: {str(e)}")
            raise
    
    def prepare_dataset(self) -> Dataset:
        """
        Prepare dataset for PPO training.
        
        Returns:
            Prepared dataset with prompts
        """
        self.logger.info("Loading dataset for PPO training")
        
        try:
            # Load SFT dataset for prompts
            sft_dataset = self.dataset_manager.load_sft_dataset(streaming=False)
            
            # Extract prompts from SFT dataset
            prompts = []
            for example in sft_dataset:
                if 'messages' in example:
                    # Extract user messages as prompts
                    for message in example['messages']:
                        if message.get('role') == 'user':
                            prompts.append(message['content'])
                elif 'prompt' in example:
                    prompts.append(example['prompt'])
                elif 'text' in example:
                    # Extract first part as prompt
                    text = example['text']
                    if len(text) > 100:
                        prompts.append(text[:100])
            
            # Limit dataset size for PPO training
            max_prompts = self.config.training.ppo.max_prompts
            if len(prompts) > max_prompts:
                prompts = prompts[:max_prompts]
            
            # Create dataset
            dataset = Dataset.from_dict({'query': prompts})
            
            self.logger.info(f"PPO dataset prepared with {len(dataset)} prompts")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to prepare PPO dataset: {str(e)}")
            raise
    
    def create_ppo_config(self) -> PPOConfig:
        """
        Create PPO configuration.
        
        Returns:
            PPO configuration
        """
        ppo_config = self.config.training.ppo
        
        return PPOConfig(
            # Model configuration
            model_name=self.config.model.name,
            
            # Training parameters
            steps=ppo_config.steps,
            learning_rate=ppo_config.learning_rate,
            batch_size=ppo_config.batch_size,
            mini_batch_size=ppo_config.mini_batch_size,
            gradient_accumulation_steps=ppo_config.gradient_accumulation_steps,
            
            # PPO specific parameters
            ppo_epochs=ppo_config.ppo_epochs,
            gamma=ppo_config.gamma,
            lam=ppo_config.lam,
            cliprange=ppo_config.cliprange,
            cliprange_value=ppo_config.cliprange_value,
            vf_coef=ppo_config.vf_coef,
            
            # KL divergence control
            init_kl_coef=ppo_config.init_kl_coef,
            target_kl=ppo_config.target_kl,
            adap_kl_ctrl=True,
            
            # Generation parameters
            max_length=self.config.model.max_length,
            
            # Optimization
            optimize_cuda_cache=True,
            
            # Logging
            log_with="wandb" if self.config.tracking.use_wandb else None,
            project_kwargs={
                "project_name": self.config.tracking.wandb_project,
                "run_name": f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            } if self.config.tracking.use_wandb else None,
        )
    
    def compute_reward(self, texts: List[str]) -> List[float]:
        """
        Compute rewards for generated texts using reward model.
        
        Args:
            texts: List of generated texts
            
        Returns:
            List of reward scores
        """
        if self.reward_model is None:
            # Fallback: return dummy rewards
            return [0.0] * len(texts)
        
        try:
            rewards = []
            device = next(self.reward_model.parameters()).device
            
            with torch.no_grad():
                for text in texts:
                    # Tokenize text
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.model.max_length
                    ).to(device)
                    
                    # Get reward
                    outputs = self.reward_model(**inputs)
                    reward = outputs['rewards'].item()
                    rewards.append(reward)
            
            return rewards
            
        except Exception as e:
            self.logger.error(f"Failed to compute rewards: {str(e)}")
            # Return dummy rewards on error
            return [0.0] * len(texts)
    
    def train(
        self, 
        sft_checkpoint_path: str, 
        reward_checkpoint_path: str
    ) -> PPOTrainingResult:
        """
        Execute PPO training.
        
        Args:
            sft_checkpoint_path: Path to SFT checkpoint
            reward_checkpoint_path: Path to reward model checkpoint
            
        Returns:
            Training result with metrics and checkpoint path
            
        Requirement 1.3: PPO training stage implementation
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting PPO training")
            
            # Prepare models and dataset
            self.policy_model = self.prepare_policy_model(sft_checkpoint_path)
            self.reward_model = self.prepare_reward_model(reward_checkpoint_path)
            self.dataset = self.prepare_dataset()
            
            # Create PPO configuration
            ppo_config = self.create_ppo_config()
            
            # Create PPO trainer
            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=self.policy_model,
                tokenizer=self.tokenizer,
                dataset=self.dataset
            )
            
            # Create callback
            callback = PPOTrainingCallback(
                self.experiment_tracker,
                self.model_manager,
                self.checkpoint_manager,
                self.config
            )
            
            # Generation parameters
            generation_kwargs = {
                "min_length": -1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "max_new_tokens": 128,
            }
            
            # Training loop
            self.logger.info("Starting PPO training loop")
            
            for step, batch in enumerate(self.ppo_trainer.dataloader):
                if step >= ppo_config.steps:
                    break
                
                # Generate responses
                query_tensors = batch["input_ids"]
                response_tensors = self.ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **generation_kwargs
                )
                
                # Decode responses
                batch["response"] = self.tokenizer.batch_decode(
                    response_tensors, skip_special_tokens=True
                )
                
                # Compute rewards
                texts = [q + r for q, r in zip(batch["query"], batch["response"])]
                rewards = self.compute_reward(texts)
                rewards = [torch.tensor(r) for r in rewards]
                
                # PPO step
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                
                # Log statistics
                callback.on_step_end(self.ppo_trainer, step, stats)
                
                # Save checkpoint periodically
                if step % self.config.training.ppo.save_steps == 0:
                    callback.save_checkpoint(self.ppo_trainer, step)
                
                # Check early stopping
                if callback.should_stop_training():
                    self.logger.info(f"Early stopping at step {step}")
                    break
                
                # Memory cleanup
                if step % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # Save final checkpoint
            final_checkpoint_path = self._save_final_checkpoint(
                self.ppo_trainer.model, callback.best_reward, step
            )
            
            # Calculate training duration and memory usage
            duration = time.time() - start_time
            memory_peak = max(callback.memory_history) if callback.memory_history else 0.0
            
            # Create result
            result = PPOTrainingResult(
                success=True,
                checkpoint_path=final_checkpoint_path,
                metrics={
                    'final_reward': callback.best_reward,
                    'final_kl': callback.kl_history[-1] if callback.kl_history else 0.0,
                    'steps_completed': step
                },
                duration_seconds=duration,
                memory_peak_gb=memory_peak,
                final_reward=callback.best_reward,
                final_kl_divergence=callback.kl_history[-1] if callback.kl_history else 0.0,
                total_steps=step
            )
            
            self.logger.info(f"PPO training completed successfully in {duration:.2f} seconds")
            self.logger.info(f"Final reward: {result.final_reward:.4f}")
            self.logger.info(f"Checkpoint saved: {final_checkpoint_path}")
            
            return result
            
        except Exception as e:
            error_msg = f"PPO training failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            return PPOTrainingResult(
                success=False,
                error_message=error_msg,
                duration_seconds=time.time() - start_time
            )
        
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _save_final_checkpoint(
        self, 
        model: AutoModelForCausalLMWithValueHead, 
        final_reward: float,
        final_step: int
    ) -> str:
        """
        Save final PPO model checkpoint.
        
        Args:
            model: Trained PPO model
            final_reward: Final reward score
            final_step: Final training step
            
        Returns:
            Path to saved checkpoint
        """
        try:
            checkpoint_name = f"ppo_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save through checkpoint manager
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=None,
                epoch=0,  # Final checkpoint
                stage="ppo",
                checkpoint_name=checkpoint_name,
                metadata={
                    'final_reward': final_reward,
                    'final_step': final_step,
                    'model_type': 'ppo_model',
                    'training_completed': True
                }
            )
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save final PPO checkpoint: {str(e)}")
            raise
    
    def evaluate_model(self, checkpoint_path: str, eval_prompts: List[str]) -> Dict[str, float]:
        """
        Evaluate trained PPO model.
        
        Args:
            checkpoint_path: Path to PPO model checkpoint
            eval_prompts: List of evaluation prompts
            
        Returns:
            Evaluation metrics
        """
        try:
            self.logger.info(f"Evaluating PPO model from {checkpoint_path}")
            
            # Load model
            model, _, _ = self.checkpoint_manager.load_checkpoint(
                Path(checkpoint_path).name
            )
            
            if model is None:
                raise ValueError(f"Could not load model from {checkpoint_path}")
            
            model.eval()
            
            # Generate responses
            responses = []
            rewards = []
            
            with torch.no_grad():
                for prompt in eval_prompts:
                    # Tokenize prompt
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.model.max_length
                    )
                    
                    # Generate response
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    # Decode response
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    responses.append(response)
                    
                    # Compute reward if reward model available
                    if self.reward_model is not None:
                        full_text = prompt + response
                        reward = self.compute_reward([full_text])[0]
                        rewards.append(reward)
            
            # Calculate metrics
            metrics = {
                'num_responses': len(responses),
                'avg_response_length': np.mean([len(r.split()) for r in responses]),
            }
            
            if rewards:
                metrics.update({
                    'avg_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'min_reward': np.min(rewards),
                    'max_reward': np.max(rewards)
                })
            
            self.logger.info("PPO model evaluation completed")
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"PPO model evaluation failed: {str(e)}")
            raise


def create_ppo_trainer(
    config: Config,
    model_manager: ModelManager,
    dataset_manager: DatasetManager,
    checkpoint_manager: CheckpointManager,
    experiment_tracker: ExperimentTracker
) -> PPOTrainerWrapper:
    """
    Factory function to create PPO trainer.
    
    Args:
        config: Pipeline configuration
        model_manager: Model manager instance
        dataset_manager: Dataset manager instance
        checkpoint_manager: Checkpoint manager instance
        experiment_tracker: Experiment tracker instance
        
    Returns:
        Configured PPO trainer
    """
    return PPOTrainerWrapper(
        config=config,
        model_manager=model_manager,
        dataset_manager=dataset_manager,
        checkpoint_manager=checkpoint_manager,
        experiment_tracker=experiment_tracker
    )