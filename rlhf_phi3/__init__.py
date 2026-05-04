"""
RLHF Phi-3 Pipeline

A production-grade Reinforcement Learning from Human Feedback (RLHF) pipeline 
for Microsoft Phi-3 Mini (3.8B parameters). Designed for Google Colab's T4 GPU 
constraints with comprehensive checkpoint management and experiment tracking.

The pipeline implements a three-stage training process:
1. Supervised Fine-Tuning (SFT)
2. Reward Model Training  
3. Proximal Policy Optimization (PPO)

Key Features:
- Memory-efficient training with PEFT/LoRA
- Google Drive checkpoint persistence
- Weights & Biases experiment tracking
- MT-Bench evaluation protocol
- HuggingFace Hub model publishing
"""

__version__ = "0.1.0"
__author__ = "RLHF Pipeline Team"
__email__ = "contact@rlhf-pipeline.com"

# Core component imports
from .config import Config
from .data import DatasetManager
from .models import ModelManager
from .training import TrainingOrchestrator
from .checkpoints import CheckpointManager
from .tracking import ExperimentTracker
from .evaluation import EvaluationEngine

__all__ = [
    "Config",
    "DatasetManager", 
    "ModelManager",
    "TrainingOrchestrator",
    "CheckpointManager",
    "ExperimentTracker",
    "EvaluationEngine",
]