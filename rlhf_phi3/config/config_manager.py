"""
Configuration Manager for RLHF Phi-3 Pipeline

This module provides centralized configuration management for all hyperparameters,
paths, and environment settings. It supports validation, serialization, and 
stage-specific configuration subsets as required by the RLHF pipeline.

Requirements satisfied:
- 8.1: Serializable configuration object
- 8.2: Configuration consistency validation  
- 8.3: Stage-specific configuration subsets
- 8.4: Configuration serialization for reproducibility
- 8.5: Parameter bounds enforcement
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from copy import deepcopy

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration parameters."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def validate(self) -> List[str]:
        """Validate LoRA configuration parameters."""
        errors = []
        
        if self.r <= 0:
            errors.append("LoRA rank (r) must be positive")
        if self.r > 256:
            errors.append("LoRA rank (r) should not exceed 256 for efficiency")
            
        if self.alpha <= 0:
            errors.append("LoRA alpha must be positive")
            
        if not (0.0 <= self.dropout <= 1.0):
            errors.append("LoRA dropout must be between 0.0 and 1.0")
            
        if self.bias not in ["none", "all", "lora_only"]:
            errors.append("LoRA bias must be one of: 'none', 'all', 'lora_only'")
            
        if self.task_type not in ["CAUSAL_LM", "SEQ_CLS", "TOKEN_CLS"]:
            errors.append("LoRA task_type must be one of: 'CAUSAL_LM', 'SEQ_CLS', 'TOKEN_CLS'")
            
        if not self.target_modules:
            errors.append("LoRA target_modules cannot be empty")
            
        return errors


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_length: int = 2048
    device: str = "auto"
    
    def validate(self) -> List[str]:
        """Validate model configuration parameters."""
        errors = []
        
        if not self.name:
            errors.append("Model name cannot be empty")
            
        if self.max_length <= 0:
            errors.append("Model max_length must be positive")
        if self.max_length > 32768:
            errors.append("Model max_length should not exceed 32768 for memory efficiency")
            
        if self.device not in ["auto", "cpu", "cuda", "mps"]:
            errors.append("Device must be one of: 'auto', 'cpu', 'cuda', 'mps'")
            
        return errors


@dataclass
class StageTrainingConfig:
    """Training configuration for a specific stage."""
    epochs: int = 1
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_steps: int = 1000
    
    def validate(self) -> List[str]:
        """Validate stage training configuration parameters."""
        errors = []
        
        if self.epochs <= 0:
            errors.append("Epochs must be positive")
            
        if not (1e-6 <= self.learning_rate <= 1e-2):
            errors.append("Learning rate must be between 1e-6 and 1e-2")
            
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        if self.batch_size > 64:
            errors.append("Batch size should not exceed 64 for memory efficiency")
            
        if self.gradient_accumulation_steps <= 0:
            errors.append("Gradient accumulation steps must be positive")
            
        if self.warmup_steps < 0:
            errors.append("Warmup steps cannot be negative")
            
        if self.max_steps <= 0:
            errors.append("Max steps must be positive")
            
        return errors


@dataclass
class PPOTrainingConfig:
    """PPO-specific training configuration."""
    learning_rate: float = 1e-5
    batch_size: int = 1
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    ppo_epochs: int = 4
    max_steps: int = 1000
    
    def validate(self) -> List[str]:
        """Validate PPO training configuration parameters."""
        errors = []
        
        if not (1e-6 <= self.learning_rate <= 1e-2):
            errors.append("PPO learning rate must be between 1e-6 and 1e-2")
            
        if self.batch_size <= 0:
            errors.append("PPO batch size must be positive")
            
        if self.mini_batch_size <= 0:
            errors.append("PPO mini batch size must be positive")
        if self.mini_batch_size > self.batch_size:
            errors.append("PPO mini batch size cannot exceed batch size")
            
        if self.gradient_accumulation_steps <= 0:
            errors.append("PPO gradient accumulation steps must be positive")
            
        if self.ppo_epochs <= 0:
            errors.append("PPO epochs must be positive")
            
        if self.max_steps <= 0:
            errors.append("PPO max steps must be positive")
            
        return errors


@dataclass
class TrainingConfig:
    """Complete training configuration for all stages."""
    sft: StageTrainingConfig = field(default_factory=StageTrainingConfig)
    reward: StageTrainingConfig = field(default_factory=lambda: StageTrainingConfig(
        epochs=1, learning_rate=1e-4, batch_size=2, gradient_accumulation_steps=8,
        warmup_steps=50, max_steps=500
    ))
    ppo: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)
    
    def validate(self) -> List[str]:
        """Validate training configuration for all stages."""
        errors = []
        errors.extend([f"SFT: {e}" for e in self.sft.validate()])
        errors.extend([f"Reward: {e}" for e in self.reward.validate()])
        errors.extend([f"PPO: {e}" for e in self.ppo.validate()])
        return errors


@dataclass
class OptimizationConfig:
    """Optimization settings for training."""
    optimizer_type: str = "adamw_torch"
    scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    
    def validate(self) -> List[str]:
        """Validate optimization configuration parameters."""
        errors = []
        
        valid_optimizers = ["adamw_torch", "adamw_hf", "sgd", "adafactor"]
        if self.optimizer_type not in valid_optimizers:
            errors.append(f"Optimizer type must be one of: {valid_optimizers}")
            
        valid_schedulers = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"]
        if self.scheduler_type not in valid_schedulers:
            errors.append(f"Scheduler type must be one of: {valid_schedulers}")
            
        if not (0.0 <= self.weight_decay <= 1.0):
            errors.append("Weight decay must be between 0.0 and 1.0")
            
        if self.max_grad_norm <= 0:
            errors.append("Max gradient norm must be positive")
            
        if self.dataloader_num_workers < 0:
            errors.append("Dataloader num workers cannot be negative")
            
        return errors


@dataclass
class PathsConfig:
    """Path configuration for outputs and storage."""
    base_output_dir: str = "/content/drive/MyDrive/rlhf-phi3"
    cache_dir: str = "/content/cache"
    logs_dir: str = "/content/logs"
    
    def validate(self) -> List[str]:
        """Validate path configuration parameters."""
        errors = []
        
        if not self.base_output_dir:
            errors.append("Base output directory cannot be empty")
        if not self.cache_dir:
            errors.append("Cache directory cannot be empty")
        if not self.logs_dir:
            errors.append("Logs directory cannot be empty")
            
        return errors


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""
    project: str = "rlhf-phi3-pipeline"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["phi3", "rlhf", "colab"])
    
    def validate(self) -> List[str]:
        """Validate WandB configuration parameters."""
        errors = []
        
        if not self.project:
            errors.append("WandB project name cannot be empty")
            
        return errors


@dataclass
class DatasetConfig:
    """Dataset configuration for a specific dataset."""
    name: str
    split: str
    max_samples: int
    
    def validate(self) -> List[str]:
        """Validate dataset configuration parameters."""
        errors = []
        
        if not self.name:
            errors.append("Dataset name cannot be empty")
        if not self.split:
            errors.append("Dataset split cannot be empty")
        if self.max_samples <= 0:
            errors.append("Max samples must be positive")
            
        return errors


@dataclass
class DatasetsConfig:
    """Configuration for all datasets."""
    sft: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        name="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        max_samples=10000
    ))
    preference: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        name="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs", 
        max_samples=5000
    ))
    
    def validate(self) -> List[str]:
        """Validate datasets configuration."""
        errors = []
        errors.extend([f"SFT dataset: {e}" for e in self.sft.validate()])
        errors.extend([f"Preference dataset: {e}" for e in self.preference.validate()])
        return errors


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    mt_bench: Dict[str, Any] = field(default_factory=lambda: {
        "num_samples": 100,
        "temperature": 0.7,
        "max_new_tokens": 512
    })
    
    def validate(self) -> List[str]:
        """Validate evaluation configuration parameters."""
        errors = []
        
        if "num_samples" in self.mt_bench:
            if self.mt_bench["num_samples"] <= 0:
                errors.append("MT-Bench num_samples must be positive")
                
        if "temperature" in self.mt_bench:
            if not (0.0 <= self.mt_bench["temperature"] <= 2.0):
                errors.append("MT-Bench temperature must be between 0.0 and 2.0")
                
        if "max_new_tokens" in self.mt_bench:
            if self.mt_bench["max_new_tokens"] <= 0:
                errors.append("MT-Bench max_new_tokens must be positive")
                
        return errors


@dataclass
class CheckpointingConfig:
    """Checkpointing configuration."""
    save_steps: int = 100
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate checkpointing configuration parameters."""
        errors = []
        
        if self.save_steps <= 0:
            errors.append("Save steps must be positive")
            
        if self.save_total_limit <= 0:
            errors.append("Save total limit must be positive")
            
        return errors


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_steps: int = 10
    eval_steps: int = 200
    
    def validate(self) -> List[str]:
        """Validate logging configuration parameters."""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            errors.append(f"Logging level must be one of: {valid_levels}")
            
        if self.log_steps <= 0:
            errors.append("Log steps must be positive")
            
        if self.eval_steps <= 0:
            errors.append("Eval steps must be positive")
            
        return errors


@dataclass
class Config:
    """
    Main configuration class for the RLHF Phi-3 pipeline.
    
    This class centralizes all hyperparameters, paths, and environment settings
    with built-in validation, serialization, and stage-specific subsetting capabilities.
    
    Requirements satisfied:
    - 8.1: Serializable configuration object
    - 8.2: Configuration consistency validation
    - 8.3: Stage-specific configuration subsets  
    - 8.4: Configuration serialization for reproducibility
    - 8.5: Parameter bounds enforcement
    """
    
    # Core configuration sections
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate_config(self) -> bool:
        """
        Validate the complete configuration for consistency and parameter bounds.
        
        Returns:
            bool: True if configuration is valid, False otherwise
            
        Requirement 8.2: Configuration consistency validation
        Requirement 8.5: Parameter bounds enforcement
        """
        errors = self.get_validation_errors()
        
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
            
        logger.info("Configuration validation passed")
        return True
    
    def get_validation_errors(self) -> List[str]:
        """
        Get all validation errors for the configuration.
        
        Returns:
            List[str]: List of validation error messages
        """
        errors = []
        
        # Validate each configuration section
        errors.extend([f"Model: {e}" for e in self.model.validate()])
        errors.extend([f"LoRA: {e}" for e in self.lora.validate()])
        errors.extend(self.training.validate())
        errors.extend([f"Optimization: {e}" for e in self.optimization.validate()])
        errors.extend([f"Paths: {e}" for e in self.paths.validate()])
        errors.extend([f"WandB: {e}" for e in self.wandb.validate()])
        errors.extend(self.datasets.validate())
        errors.extend([f"Evaluation: {e}" for e in self.evaluation.validate()])
        errors.extend([f"Checkpointing: {e}" for e in self.checkpointing.validate()])
        errors.extend([f"Logging: {e}" for e in self.logging.validate()])
        
        # Cross-section validation
        errors.extend(self._validate_cross_section_consistency())
        
        return errors
    
    def _validate_cross_section_consistency(self) -> List[str]:
        """Validate consistency across configuration sections."""
        errors = []
        
        # Check that batch sizes are compatible with memory constraints
        total_sft_batch = self.training.sft.batch_size * self.training.sft.gradient_accumulation_steps
        total_reward_batch = self.training.reward.batch_size * self.training.reward.gradient_accumulation_steps
        total_ppo_batch = self.training.ppo.batch_size * self.training.ppo.gradient_accumulation_steps
        
        if total_sft_batch > 128:
            errors.append("SFT effective batch size (batch_size * gradient_accumulation_steps) should not exceed 128")
        if total_reward_batch > 128:
            errors.append("Reward effective batch size should not exceed 128")
        if total_ppo_batch > 128:
            errors.append("PPO effective batch size should not exceed 128")
            
        # Check that max_length is consistent with model capabilities
        if "phi-3" in self.model.name.lower() and self.model.max_length > 4096:
            errors.append("Phi-3 models typically support max_length up to 4096 tokens")
            
        # Check that checkpointing frequency is reasonable
        if self.checkpointing.save_steps > min(
            self.training.sft.max_steps,
            self.training.reward.max_steps, 
            self.training.ppo.max_steps
        ):
            errors.append("Checkpoint save_steps should be less than the minimum max_steps across stages")
            
        return errors
    
    def get_stage_config(self, stage: str) -> Dict[str, Any]:
        """
        Get configuration subset for a specific training stage.
        
        Args:
            stage: Training stage ('sft', 'reward', or 'ppo')
            
        Returns:
            Dict containing stage-specific configuration
            
        Requirement 8.3: Stage-specific configuration subsets
        """
        if stage not in ["sft", "reward", "ppo"]:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: sft, reward, ppo")
        
        # Base configuration shared across all stages
        base_config = {
            "model": asdict(self.model),
            "lora": asdict(self.lora),
            "optimization": asdict(self.optimization),
            "paths": asdict(self.paths),
            "checkpointing": asdict(self.checkpointing),
            "logging": asdict(self.logging)
        }
        
        # Add stage-specific training configuration
        if stage == "ppo":
            base_config["training"] = asdict(self.training.ppo)
        else:
            stage_training = getattr(self.training, stage)
            base_config["training"] = asdict(stage_training)
        
        # Add stage-specific dataset configuration
        if stage == "sft":
            base_config["dataset"] = asdict(self.datasets.sft)
        elif stage == "reward":
            base_config["dataset"] = asdict(self.datasets.preference)
        else:  # ppo
            # PPO uses both datasets
            base_config["datasets"] = {
                "sft": asdict(self.datasets.sft),
                "preference": asdict(self.datasets.preference)
            }
        
        # Add experiment tracking for all stages
        base_config["wandb"] = asdict(self.wandb)
        
        return base_config
    
    def save_config(self, path: Union[str, Path], format: str = None) -> None:
        """
        Save configuration to file in specified format.
        
        Args:
            path: File path to save configuration
            format: File format ('yaml' or 'json'). If None, infers from file extension
                   or defaults to 'json' if YAML is not available
            
        Requirement 8.4: Configuration serialization for reproducibility
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format if not specified
        if format is None:
            if path.suffix.lower() in [".yaml", ".yml"]:
                format = "yaml"
            elif path.suffix.lower() == ".json":
                format = "json"
            else:
                # Default to JSON if YAML not available, otherwise YAML
                format = "json" if not YAML_AVAILABLE else "yaml"
        
        config_dict = asdict(self)
        
        if format.lower() == "yaml":
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'yaml' or 'json'")
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load_config(cls, path: Union[str, Path]) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            path: File path to load configuration from
            
        Returns:
            Config instance loaded from file
            
        Requirement 8.4: Configuration serialization for reproducibility
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
                config_dict = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Convert nested dictionaries back to dataclass instances
        config = cls._dict_to_config(config_dict)
        
        logger.info(f"Configuration loaded from {path}")
        return config
    
    @classmethod
    def _dict_to_config(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Convert dictionary to Config instance with proper dataclass nesting."""
        
        # Helper function to create dataclass instances from dictionaries
        def create_dataclass_instance(dataclass_type, data):
            if isinstance(data, dict):
                # Filter out keys that don't exist in the dataclass
                field_names = {f.name for f in dataclass_type.__dataclass_fields__.values()}
                filtered_data = {k: v for k, v in data.items() if k in field_names}
                return dataclass_type(**filtered_data)
            return data
        
        # Create nested dataclass instances
        model_config = create_dataclass_instance(ModelConfig, config_dict.get("model", {}))
        lora_config = create_dataclass_instance(LoRAConfig, config_dict.get("lora", {}))
        
        # Handle training config with nested stages
        training_dict = config_dict.get("training", {})
        sft_config = create_dataclass_instance(StageTrainingConfig, training_dict.get("sft", {}))
        reward_config = create_dataclass_instance(StageTrainingConfig, training_dict.get("reward", {}))
        ppo_config = create_dataclass_instance(PPOTrainingConfig, training_dict.get("ppo", {}))
        training_config = TrainingConfig(sft=sft_config, reward=reward_config, ppo=ppo_config)
        
        optimization_config = create_dataclass_instance(OptimizationConfig, config_dict.get("optimization", {}))
        paths_config = create_dataclass_instance(PathsConfig, config_dict.get("paths", {}))
        wandb_config = create_dataclass_instance(WandBConfig, config_dict.get("wandb", {}))
        
        # Handle datasets config
        datasets_dict = config_dict.get("datasets", {})
        sft_dataset = create_dataclass_instance(DatasetConfig, datasets_dict.get("sft", {}))
        preference_dataset = create_dataclass_instance(DatasetConfig, datasets_dict.get("preference", {}))
        datasets_config = DatasetsConfig(sft=sft_dataset, preference=preference_dataset)
        
        evaluation_config = create_dataclass_instance(EvaluationConfig, config_dict.get("evaluation", {}))
        checkpointing_config = create_dataclass_instance(CheckpointingConfig, config_dict.get("checkpointing", {}))
        logging_config = create_dataclass_instance(LoggingConfig, config_dict.get("logging", {}))
        
        return cls(
            model=model_config,
            lora=lora_config,
            training=training_config,
            optimization=optimization_config,
            paths=paths_config,
            wandb=wandb_config,
            datasets=datasets_config,
            evaluation=evaluation_config,
            checkpointing=checkpointing_config,
            logging=logging_config
        )
    
    def copy(self) -> 'Config':
        """Create a deep copy of the configuration."""
        return deepcopy(self)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> 'Config':
        """
        Create a new configuration with updates from a dictionary.
        
        Args:
            updates: Dictionary of updates to apply
            
        Returns:
            New Config instance with updates applied
        """
        config_dict = asdict(self)
        
        # Deep merge the updates
        def deep_merge(base: Dict, updates: Dict) -> Dict:
            result = base.copy()
            for key, value in updates.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(config_dict, updates)
        return self._dict_to_config(merged_dict)
    
    def create_checkpoint_snapshot(self, checkpoint_id: str, 
                                 training_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a complete configuration snapshot for checkpoint storage.
        
        This method creates a comprehensive snapshot of the configuration state
        that can be saved with checkpoints for full reproducibility.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            training_metadata: Optional training metadata to include
            
        Returns:
            Complete configuration snapshot dictionary
            
        Requirement 15.1: Configuration snapshots with checkpoints
        """
        from datetime import datetime
        import hashlib
        
        # Create base snapshot
        snapshot = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self),
            "config_version": "1.0",
            "snapshot_type": "checkpoint"
        }
        
        # Add training metadata if provided
        if training_metadata:
            snapshot["training_metadata"] = training_metadata
        
        # Create configuration hash for integrity verification
        config_str = json.dumps(snapshot["config"], sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        snapshot["config_hash"] = config_hash
        
        # Add reproducibility information
        try:
            from ..utils.reproducibility import ReproducibilityManager
            repro_manager = ReproducibilityManager()
            snapshot["reproducibility"] = repro_manager.get_reproducibility_summary()
        except ImportError:
            logger.warning("Reproducibility manager not available for snapshot")
            snapshot["reproducibility"] = None
        
        logger.info(f"Created configuration snapshot for checkpoint: {checkpoint_id}")
        return snapshot
    
    def save_checkpoint_snapshot(self, checkpoint_path: Union[str, Path], 
                               checkpoint_id: str,
                               training_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save configuration snapshot to checkpoint directory.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            checkpoint_id: Unique identifier for the checkpoint
            training_metadata: Optional training metadata to include
            
        Requirement 15.1: Configuration snapshots with checkpoints
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Create snapshot
        snapshot = self.create_checkpoint_snapshot(checkpoint_id, training_metadata)
        
        # Save snapshot
        snapshot_file = checkpoint_path / "config_snapshot.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        logger.info(f"Configuration snapshot saved to {snapshot_file}")
    
    @classmethod
    def load_checkpoint_snapshot(cls, checkpoint_path: Union[str, Path]) -> Tuple['Config', Dict[str, Any]]:
        """
        Load configuration from checkpoint snapshot.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Tuple of (Config instance, snapshot metadata)
            
        Requirement 15.1: Configuration snapshots with checkpoints
        """
        checkpoint_path = Path(checkpoint_path)
        snapshot_file = checkpoint_path / "config_snapshot.json"
        
        if not snapshot_file.exists():
            raise FileNotFoundError(f"Configuration snapshot not found: {snapshot_file}")
        
        with open(snapshot_file, "r") as f:
            snapshot = json.load(f)
        
        # Verify snapshot integrity
        config_dict = snapshot["config"]
        expected_hash = snapshot.get("config_hash")
        
        if expected_hash:
            config_str = json.dumps(config_dict, sort_keys=True)
            actual_hash = hashlib.sha256(config_str.encode()).hexdigest()
            
            if actual_hash != expected_hash:
                logger.warning("Configuration snapshot integrity check failed")
        
        # Create config instance
        config = cls._dict_to_config(config_dict)
        
        logger.info(f"Configuration loaded from checkpoint snapshot: {snapshot_file}")
        return config, snapshotcls._dict_to_config(config_dict)
        
        # Return config and metadata
        metadata = {k: v for k, v in snapshot.items() if k != "config"}
        
        logger.info(f"Configuration loaded from snapshot: {snapshot.get('checkpoint_id', 'unknown')}")
        return config, metadata


# Convenience functions for common configuration scenarios

def load_default_config() -> Config:
    """Load the default configuration."""
    return Config()

def load_colab_config() -> Config:
    """Load Colab-optimized configuration."""
    config = Config()
    
    # Apply Colab-specific optimizations
    config.model.max_length = 1024  # Reduced for memory
    config.lora.r = 8  # Smaller rank
    config.lora.alpha = 16
    
    # Smaller batch sizes and more aggressive accumulation
    config.training.sft.epochs = 2
    config.training.sft.batch_size = 2
    config.training.sft.gradient_accumulation_steps = 8
    config.training.sft.max_steps = 500
    
    config.training.reward.batch_size = 1
    config.training.reward.gradient_accumulation_steps = 16
    config.training.reward.max_steps = 250
    
    config.training.ppo.learning_rate = 5e-6
    config.training.ppo.gradient_accumulation_steps = 32
    config.training.ppo.ppo_epochs = 2
    config.training.ppo.max_steps = 200
    
    # Memory optimizations
    config.optimization.max_grad_norm = 0.5
    config.optimization.dataloader_num_workers = 0
    
    # Smaller datasets
    config.datasets.sft.max_samples = 2000
    config.datasets.preference.max_samples = 1000
    
    # More frequent checkpointing
    config.checkpointing.save_steps = 50
    config.checkpointing.save_total_limit = 2
    
    # Less frequent logging
    config.logging.log_steps = 25
    config.logging.eval_steps = 100
    
    return config