"""
Integration tests for training stages in RLHF Phi-3 Pipeline.

This module tests the complete integration of all three training stages:
SFT, Reward Model, and PPO training with toy datasets.

Requirements satisfied:
- 11.3: Integration tests covering complete end-to-end pipeline
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from rlhf_phi3.config.config_manager import Config, create_config
from rlhf_phi3.models.model_manager import ModelManager
from rlhf_phi3.data.dataset_manager import DatasetManager
from rlhf_phi3.checkpoints.checkpoint_manager import CheckpointManager
from rlhf_phi3.tracking.experiment_tracker import ExperimentTracker
from rlhf_phi3.training.sft_trainer import SFTTrainer, create_sft_trainer
from rlhf_phi3.training.reward_trainer import RewardTrainer, create_reward_trainer
from rlhf_phi3.training.ppo_trainer import PPOTrainerWrapper, create_ppo_trainer


class TestTrainingStagesIntegration:
    """Integration tests for all training stages."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration."""
        config_dict = {
            'model': {
                'name': 'microsoft/DialoGPT-small',  # Smaller model for testing
                'max_length': 256,
                'device': 'cpu'  # Use CPU for testing
            },
            'training': {
                'sft': {
                    'epochs': 1,
                    'batch_size': 2,
                    'learning_rate': 5e-5,
                    'warmup_steps': 10,
                    'logging_steps': 5,
                    'save_steps': 10,
                    'eval_steps': 10,
                    'gradient_accumulation_steps': 1,
                    'weight_decay': 0.01,
                    'early_stopping_patience': 3
                },
                'reward': {
                    'epochs': 1,
                    'batch_size': 2,
                    'eval_batch_size': 2,
                    'learning_rate': 1e-5,
                    'warmup_steps': 5,
                    'logging_steps': 5,
                    'save_steps': 10,
                    'eval_steps': 10,
                    'gradient_accumulation_steps': 1,
                    'weight_decay': 0.01,
                    'early_stopping_patience': 2
                },
                'ppo': {
                    'steps': 20,
                    'batch_size': 2,
                    'mini_batch_size': 1,
                    'learning_rate': 1e-6,
                    'gradient_accumulation_steps': 1,
                    'ppo_epochs': 2,
                    'gamma': 0.99,
                    'lam': 0.95,
                    'cliprange': 0.2,
                    'cliprange_value': 0.2,
                    'vf_coef': 0.1,
                    'init_kl_coef': 0.2,
                    'target_kl': 0.1,
                    'max_prompts': 10,
                    'save_steps': 10,
                    'early_stopping_patience': 5
                }
            },
            'data': {
                'sft_dataset': 'test_sft',
                'preference_dataset': 'test_preference',
                'cache_dir': str(Path(temp_dir) / 'cache'),
                'streaming': False
            },
            'paths': {
                'output_dir': temp_dir,
                'checkpoint_dir': str(Path(temp_dir) / 'checkpoints'),
                'cache_dir': str(Path(temp_dir) / 'cache')
            },
            'tracking': {
                'use_wandb': False,
                'wandb_project': 'test-rlhf-phi3'
            },
            'lora': {
                'r': 8,
                'alpha': 16,
                'dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj']
            }
        }
        
        return create_config(config_dict)
    
    @pytest.fixture
    def toy_sft_dataset(self):
        """Create toy SFT dataset."""
        data = {
            'messages': [
                [
                    {'role': 'user', 'content': 'Hello, how are you?'},
                    {'role': 'assistant', 'content': 'I am doing well, thank you!'}
                ],
                [
                    {'role': 'user', 'content': 'What is the capital of France?'},
                    {'role': 'assistant', 'content': 'The capital of France is Paris.'}
                ],
                [
                    {'role': 'user', 'content': 'Can you help me with math?'},
                    {'role': 'assistant', 'content': 'Of course! I would be happy to help you with math.'}
                ],
                [
                    {'role': 'user', 'content': 'Tell me a joke.'},
                    {'role': 'assistant', 'content': 'Why did the chicken cross the road? To get to the other side!'}
                ]
            ]
        }
        return Dataset.from_dict(data)
    
    @pytest.fixture
    def toy_preference_dataset(self):
        """Create toy preference dataset."""
        data = {
            'prompt': [
                'What is the best way to learn programming?',
                'How do I make a good first impression?',
                'What should I cook for dinner?'
            ],
            'chosen': [
                'Start with the basics and practice regularly with small projects.',
                'Be genuine, listen actively, and show interest in others.',
                'Consider something healthy and balanced like grilled chicken with vegetables.'
            ],
            'rejected': [
                'Just copy code from the internet without understanding it.',
                'Try to impress everyone by talking only about yourself.',
                'Order fast food every night, it is easier.'
            ]
        }
        return Dataset.from_dict(data)
    
    @pytest.fixture
    def mock_components(self, test_config, temp_dir):
        """Create mock components for testing."""
        # Create real components but with mocked heavy operations
        model_manager = Mock(spec=ModelManager)
        dataset_manager = Mock(spec=DatasetManager)
        checkpoint_manager = Mock(spec=CheckpointManager)
        experiment_tracker = Mock(spec=ExperimentTracker)
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.batch_decode.return_value = ['test response']
        tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        model_manager.tokenizer = tokenizer
        model_manager.load_base_model.return_value = Mock()
        model_manager.apply_peft.return_value = Mock()
        model_manager.prepare_for_training.return_value = Mock()
        
        # Mock checkpoint operations
        checkpoint_manager.save_checkpoint.return_value = str(Path(temp_dir) / 'test_checkpoint')
        checkpoint_manager.load_checkpoint.return_value = (Mock(), Mock(), 0)
        
        # Mock experiment tracking
        experiment_tracker.start_run.return_value = None
        experiment_tracker.log_metrics.return_value = None
        experiment_tracker.finish_run.return_value = None
        
        return {
            'model_manager': model_manager,
            'dataset_manager': dataset_manager,
            'checkpoint_manager': checkpoint_manager,
            'experiment_tracker': experiment_tracker
        }
    
    def test_sft_training_integration(
        self, 
        test_config, 
        toy_sft_dataset, 
        mock_components
    ):
        """Test SFT training stage integration."""
        # Setup dataset manager mock
        mock_components['dataset_manager'].load_sft_dataset.return_value = toy_sft_dataset
        mock_components['dataset_manager'].preprocess_sft_data.return_value = toy_sft_dataset
        
        # Create SFT trainer
        sft_trainer = create_sft_trainer(
            config=test_config,
            model_manager=mock_components['model_manager'],
            dataset_manager=mock_components['dataset_manager'],
            checkpoint_manager=mock_components['checkpoint_manager'],
            experiment_tracker=mock_components['experiment_tracker']
        )
        
        # Mock training to avoid actual model training
        with patch.object(sft_trainer, '_run_training') as mock_training:
            mock_training.return_value = Mock(
                metrics={'train_loss': 0.5, 'eval_loss': 0.6},
                global_step=10
            )
            
            # Execute SFT training
            result = sft_trainer.train()
            
            # Verify result
            assert result.success is True
            assert result.checkpoint_path is not None
            assert result.metrics is not None
            assert result.duration_seconds is not None
            
            # Verify component interactions
            mock_components['model_manager'].load_base_model.assert_called_once()
            mock_components['model_manager'].apply_peft.assert_called_once()
            mock_components['dataset_manager'].load_sft_dataset.assert_called_once()
            mock_components['checkpoint_manager'].save_checkpoint.assert_called()
            mock_components['experiment_tracker'].start_run.assert_called_once()
    
    def test_reward_training_integration(
        self, 
        test_config, 
        toy_preference_dataset, 
        mock_components,
        temp_dir
    ):
        """Test reward model training stage integration."""
        # Setup dataset manager mock
        mock_components['dataset_manager'].load_preference_dataset.return_value = toy_preference_dataset
        mock_components['dataset_manager'].preprocess_preference_data.return_value = toy_preference_dataset
        
        # Create reward trainer
        reward_trainer = create_reward_trainer(
            config=test_config,
            model_manager=mock_components['model_manager'],
            dataset_manager=mock_components['dataset_manager'],
            checkpoint_manager=mock_components['checkpoint_manager'],
            experiment_tracker=mock_components['experiment_tracker']
        )
        
        # Mock SFT checkpoint path
        sft_checkpoint = str(Path(temp_dir) / 'sft_checkpoint')
        
        # Mock training components
        with patch.object(reward_trainer, 'prepare_reward_model') as mock_prepare_model, \
             patch.object(reward_trainer, 'prepare_datasets') as mock_prepare_datasets, \
             patch('rlhf_phi3.training.reward_trainer.Trainer') as mock_trainer_class:
            
            # Setup mocks
            mock_prepare_model.return_value = Mock()
            mock_prepare_datasets.return_value = (toy_preference_dataset, toy_preference_dataset)
            
            mock_trainer = Mock()
            mock_trainer.train.return_value = Mock(global_step=20)
            mock_trainer.evaluate.return_value = {'eval_loss': 0.4, 'eval_accuracy': 0.8}
            mock_trainer_class.return_value = mock_trainer
            
            # Execute reward training
            result = reward_trainer.train(sft_checkpoint)
            
            # Verify result
            assert result.success is True
            assert result.checkpoint_path is not None
            assert result.final_accuracy is not None
            assert result.duration_seconds is not None
            
            # Verify component interactions
            mock_prepare_model.assert_called_once_with(sft_checkpoint)
            mock_prepare_datasets.assert_called_once()
            mock_components['experiment_tracker'].start_run.assert_called_once()
    
    def test_ppo_training_integration(
        self, 
        test_config, 
        toy_sft_dataset, 
        mock_components,
        temp_dir
    ):
        """Test PPO training stage integration."""
        # Setup dataset manager mock
        mock_components['dataset_manager'].load_sft_dataset.return_value = toy_sft_dataset
        
        # Create PPO trainer
        ppo_trainer = create_ppo_trainer(
            config=test_config,
            model_manager=mock_components['model_manager'],
            dataset_manager=mock_components['dataset_manager'],
            checkpoint_manager=mock_components['checkpoint_manager'],
            experiment_tracker=mock_components['experiment_tracker']
        )
        
        # Mock checkpoint paths
        sft_checkpoint = str(Path(temp_dir) / 'sft_checkpoint')
        reward_checkpoint = str(Path(temp_dir) / 'reward_checkpoint')
        
        # Mock training components
        with patch.object(ppo_trainer, 'prepare_policy_model') as mock_prepare_policy, \
             patch.object(ppo_trainer, 'prepare_reward_model') as mock_prepare_reward, \
             patch.object(ppo_trainer, 'prepare_dataset') as mock_prepare_dataset, \
             patch('rlhf_phi3.training.ppo_trainer.PPOTrainer') as mock_ppo_trainer_class:
            
            # Setup mocks
            mock_prepare_policy.return_value = Mock()
            mock_prepare_reward.return_value = Mock()
            mock_prepare_dataset.return_value = Dataset.from_dict({'query': ['test prompt']})
            
            mock_ppo_trainer = Mock()
            mock_ppo_trainer.dataloader = [{'input_ids': torch.tensor([[1, 2, 3]])}]
            mock_ppo_trainer.generate.return_value = torch.tensor([[4, 5, 6]])
            mock_ppo_trainer.step.return_value = {'env/reward_mean': 0.5, 'objective/kl': 0.1}
            mock_ppo_trainer_class.return_value = mock_ppo_trainer
            
            # Mock tokenizer decode
            ppo_trainer.tokenizer = Mock()
            ppo_trainer.tokenizer.batch_decode.return_value = ['test response']
            
            # Mock reward computation
            with patch.object(ppo_trainer, 'compute_reward') as mock_compute_reward:
                mock_compute_reward.return_value = [0.5]
                
                # Execute PPO training
                result = ppo_trainer.train(sft_checkpoint, reward_checkpoint)
                
                # Verify result
                assert result.success is True
                assert result.checkpoint_path is not None
                assert result.final_reward is not None
                assert result.duration_seconds is not None
                
                # Verify component interactions
                mock_prepare_policy.assert_called_once_with(sft_checkpoint)
                mock_prepare_reward.assert_called_once_with(reward_checkpoint)
                mock_prepare_dataset.assert_called_once()
    
    def test_complete_pipeline_integration(
        self, 
        test_config, 
        toy_sft_dataset, 
        toy_preference_dataset, 
        mock_components,
        temp_dir
    ):
        """Test complete three-stage pipeline integration."""
        # Setup dataset manager mocks
        mock_components['dataset_manager'].load_sft_dataset.return_value = toy_sft_dataset
        mock_components['dataset_manager'].preprocess_sft_data.return_value = toy_sft_dataset
        mock_components['dataset_manager'].load_preference_dataset.return_value = toy_preference_dataset
        mock_components['dataset_manager'].preprocess_preference_data.return_value = toy_preference_dataset
        
        # Create trainers
        sft_trainer = create_sft_trainer(
            config=test_config,
            model_manager=mock_components['model_manager'],
            dataset_manager=mock_components['dataset_manager'],
            checkpoint_manager=mock_components['checkpoint_manager'],
            experiment_tracker=mock_components['experiment_tracker']
        )
        
        reward_trainer = create_reward_trainer(
            config=test_config,
            model_manager=mock_components['model_manager'],
            dataset_manager=mock_components['dataset_manager'],
            checkpoint_manager=mock_components['checkpoint_manager'],
            experiment_tracker=mock_components['experiment_tracker']
        )
        
        ppo_trainer = create_ppo_trainer(
            config=test_config,
            model_manager=mock_components['model_manager'],
            dataset_manager=mock_components['dataset_manager'],
            checkpoint_manager=mock_components['checkpoint_manager'],
            experiment_tracker=mock_components['experiment_tracker']
        )
        
        # Mock training methods to avoid actual training
        with patch.object(sft_trainer, '_run_training') as mock_sft_training, \
             patch.object(reward_trainer, 'prepare_reward_model') as mock_prepare_reward_model, \
             patch.object(reward_trainer, 'prepare_datasets') as mock_prepare_reward_datasets, \
             patch('rlhf_phi3.training.reward_trainer.Trainer') as mock_reward_trainer_class, \
             patch.object(ppo_trainer, 'prepare_policy_model') as mock_prepare_policy, \
             patch.object(ppo_trainer, 'prepare_reward_model') as mock_prepare_reward, \
             patch.object(ppo_trainer, 'prepare_dataset') as mock_prepare_ppo_dataset, \
             patch('rlhf_phi3.training.ppo_trainer.PPOTrainer') as mock_ppo_trainer_class:
            
            # Setup SFT mocks
            mock_sft_training.return_value = Mock(
                metrics={'train_loss': 0.5},
                global_step=10
            )
            
            # Setup reward training mocks
            mock_prepare_reward_model.return_value = Mock()
            mock_prepare_reward_datasets.return_value = (toy_preference_dataset, toy_preference_dataset)
            mock_reward_trainer = Mock()
            mock_reward_trainer.train.return_value = Mock(global_step=20)
            mock_reward_trainer.evaluate.return_value = {'eval_accuracy': 0.8}
            mock_reward_trainer_class.return_value = mock_reward_trainer
            
            # Setup PPO mocks
            mock_prepare_policy.return_value = Mock()
            mock_prepare_reward.return_value = Mock()
            mock_prepare_ppo_dataset.return_value = Dataset.from_dict({'query': ['test']})
            mock_ppo_trainer_instance = Mock()
            mock_ppo_trainer_instance.dataloader = [{'input_ids': torch.tensor([[1]])}]
            mock_ppo_trainer_instance.generate.return_value = torch.tensor([[2]])
            mock_ppo_trainer_instance.step.return_value = {'env/reward_mean': 0.5}
            mock_ppo_trainer_class.return_value = mock_ppo_trainer_instance
            
            # Execute complete pipeline
            # Stage 1: SFT
            sft_result = sft_trainer.train()
            assert sft_result.success is True
            sft_checkpoint = sft_result.checkpoint_path
            
            # Stage 2: Reward Model
            reward_result = reward_trainer.train(sft_checkpoint)
            assert reward_result.success is True
            reward_checkpoint = reward_result.checkpoint_path
            
            # Stage 3: PPO
            with patch.object(ppo_trainer, 'compute_reward') as mock_compute_reward:
                mock_compute_reward.return_value = [0.5]
                ppo_trainer.tokenizer = Mock()
                ppo_trainer.tokenizer.batch_decode.return_value = ['response']
                
                ppo_result = ppo_trainer.train(sft_checkpoint, reward_checkpoint)
                assert ppo_result.success is True
            
            # Verify pipeline completion
            assert sft_result.checkpoint_path is not None
            assert reward_result.checkpoint_path is not None
            assert ppo_result.checkpoint_path is not None
            
            # Verify all stages were executed
            assert mock_components['experiment_tracker'].start_run.call_count == 3
    
    def test_error_handling_integration(
        self, 
        test_config, 
        mock_components
    ):
        """Test error handling across training stages."""
        # Create SFT trainer
        sft_trainer = create_sft_trainer(
            config=test_config,
            model_manager=mock_components['model_manager'],
            dataset_manager=mock_components['dataset_manager'],
            checkpoint_manager=mock_components['checkpoint_manager'],
            experiment_tracker=mock_components['experiment_tracker']
        )
        
        # Test SFT training failure
        mock_components['model_manager'].load_base_model.side_effect = Exception("Model loading failed")
        
        result = sft_trainer.train()
        
        # Verify error handling
        assert result.success is False
        assert result.error_message is not None
        assert "Model loading failed" in result.error_message
        assert result.duration_seconds is not None
    
    def test_checkpoint_persistence_integration(
        self, 
        test_config, 
        toy_sft_dataset, 
        mock_components,
        temp_dir
    ):
        """Test checkpoint persistence across training stages."""
        # Setup dataset manager mock
        mock_components['dataset_manager'].load_sft_dataset.return_value = toy_sft_dataset
        mock_components['dataset_manager'].preprocess_sft_data.return_value = toy_sft_dataset
        
        # Create SFT trainer
        sft_trainer = create_sft_trainer(
            config=test_config,
            model_manager=mock_components['model_manager'],
            dataset_manager=mock_components['dataset_manager'],
            checkpoint_manager=mock_components['checkpoint_manager'],
            experiment_tracker=mock_components['experiment_tracker']
        )
        
        # Mock training
        with patch.object(sft_trainer, '_run_training') as mock_training:
            mock_training.return_value = Mock(
                metrics={'train_loss': 0.5},
                global_step=10
            )
            
            # Execute training
            result = sft_trainer.train()
            
            # Verify checkpoint was saved
            mock_components['checkpoint_manager'].save_checkpoint.assert_called()
            
            # Verify checkpoint path is returned
            assert result.checkpoint_path is not None
            
            # Test checkpoint loading for next stage
            mock_components['checkpoint_manager'].load_checkpoint.assert_called()
    
    def test_memory_monitoring_integration(
        self, 
        test_config, 
        toy_sft_dataset, 
        mock_components
    ):
        """Test memory monitoring across training stages."""
        # Setup dataset manager mock
        mock_components['dataset_manager'].load_sft_dataset.return_value = toy_sft_dataset
        mock_components['dataset_manager'].preprocess_sft_data.return_value = toy_sft_dataset
        
        # Create SFT trainer
        sft_trainer = create_sft_trainer(
            config=test_config,
            model_manager=mock_components['model_manager'],
            dataset_manager=mock_components['dataset_manager'],
            checkpoint_manager=mock_components['checkpoint_manager'],
            experiment_tracker=mock_components['experiment_tracker']
        )
        
        # Mock training with memory monitoring
        with patch.object(sft_trainer, '_run_training') as mock_training, \
             patch('torch.cuda.is_available') as mock_cuda_available, \
             patch('torch.cuda.max_memory_allocated') as mock_memory_allocated:
            
            mock_cuda_available.return_value = True
            mock_memory_allocated.return_value = 1024**3  # 1GB
            mock_training.return_value = Mock(
                metrics={'train_loss': 0.5},
                global_step=10
            )
            
            # Execute training
            result = sft_trainer.train()
            
            # Verify memory monitoring
            assert result.memory_peak_gb is not None
            assert result.memory_peak_gb > 0
    
    @pytest.mark.parametrize("stage", ["sft", "reward", "ppo"])
    def test_experiment_tracking_integration(
        self, 
        stage,
        test_config, 
        toy_sft_dataset, 
        toy_preference_dataset,
        mock_components,
        temp_dir
    ):
        """Test experiment tracking integration for each stage."""
        # Setup dataset mocks
        mock_components['dataset_manager'].load_sft_dataset.return_value = toy_sft_dataset
        mock_components['dataset_manager'].preprocess_sft_data.return_value = toy_sft_dataset
        mock_components['dataset_manager'].load_preference_dataset.return_value = toy_preference_dataset
        mock_components['dataset_manager'].preprocess_preference_data.return_value = toy_preference_dataset
        
        if stage == "sft":
            trainer = create_sft_trainer(
                config=test_config,
                model_manager=mock_components['model_manager'],
                dataset_manager=mock_components['dataset_manager'],
                checkpoint_manager=mock_components['checkpoint_manager'],
                experiment_tracker=mock_components['experiment_tracker']
            )
            
            with patch.object(trainer, '_run_training') as mock_training:
                mock_training.return_value = Mock(metrics={'train_loss': 0.5}, global_step=10)
                trainer.train()
        
        elif stage == "reward":
            trainer = create_reward_trainer(
                config=test_config,
                model_manager=mock_components['model_manager'],
                dataset_manager=mock_components['dataset_manager'],
                checkpoint_manager=mock_components['checkpoint_manager'],
                experiment_tracker=mock_components['experiment_tracker']
            )
            
            with patch.object(trainer, 'prepare_reward_model'), \
                 patch.object(trainer, 'prepare_datasets'), \
                 patch('rlhf_phi3.training.reward_trainer.Trainer') as mock_trainer_class:
                
                mock_trainer = Mock()
                mock_trainer.train.return_value = Mock(global_step=20)
                mock_trainer.evaluate.return_value = {'eval_accuracy': 0.8}
                mock_trainer_class.return_value = mock_trainer
                
                trainer.train(str(Path(temp_dir) / 'sft_checkpoint'))
        
        elif stage == "ppo":
            trainer = create_ppo_trainer(
                config=test_config,
                model_manager=mock_components['model_manager'],
                dataset_manager=mock_components['dataset_manager'],
                checkpoint_manager=mock_components['checkpoint_manager'],
                experiment_tracker=mock_components['experiment_tracker']
            )
            
            with patch.object(trainer, 'prepare_policy_model'), \
                 patch.object(trainer, 'prepare_reward_model'), \
                 patch.object(trainer, 'prepare_dataset'), \
                 patch('rlhf_phi3.training.ppo_trainer.PPOTrainer') as mock_ppo_trainer_class, \
                 patch.object(trainer, 'compute_reward') as mock_compute_reward:
                
                mock_ppo_trainer = Mock()
                mock_ppo_trainer.dataloader = [{'input_ids': torch.tensor([[1]])}]
                mock_ppo_trainer.generate.return_value = torch.tensor([[2]])
                mock_ppo_trainer.step.return_value = {'env/reward_mean': 0.5}
                mock_ppo_trainer_class.return_value = mock_ppo_trainer
                
                trainer.tokenizer = Mock()
                trainer.tokenizer.batch_decode.return_value = ['response']
                mock_compute_reward.return_value = [0.5]
                
                trainer.train(
                    str(Path(temp_dir) / 'sft_checkpoint'),
                    str(Path(temp_dir) / 'reward_checkpoint')
                )
        
        # Verify experiment tracking was called
        mock_components['experiment_tracker'].start_run.assert_called()
        mock_components['experiment_tracker'].log_metrics.assert_called()
        mock_components['experiment_tracker'].finish_run.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])