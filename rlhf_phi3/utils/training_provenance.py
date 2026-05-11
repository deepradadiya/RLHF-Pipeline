"""
Training Provenance Utilities for RLHF Phi-3 Pipeline

This module provides utilities for tracking and managing training provenance
throughout the RLHF pipeline. It integrates with existing components to ensure
comprehensive metadata tracking and reproducibility.

Requirements satisfied:
- 15.1: Configuration snapshots with checkpoints
- 15.4: Training provenance in model metadata
- 15.5: Reproducibility scripts and environment recreation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict

from .reproducibility import ReproducibilityManager, create_training_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class TrainingStageProvenance:
    """Provenance information for a single training stage."""
    stage_name: str
    start_time: str
    end_time: Optional[str] = None
    total_steps: int = 0
    final_loss: Optional[float] = None
    best_metric: Optional[float] = None
    checkpoint_path: Optional[str] = None
    config_hash: Optional[str] = None
    metrics_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []
    
    def add_metric(self, step: int, metrics: Dict[str, float]) -> None:
        """Add metrics for a training step."""
        self.metrics_history.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
    
    def finalize_stage(self, end_time: Optional[str] = None, 
                      final_loss: Optional[float] = None,
                      checkpoint_path: Optional[str] = None) -> None:
        """Finalize the training stage with completion information."""
        self.end_time = end_time or datetime.now().isoformat()
        if final_loss is not None:
            self.final_loss = final_loss
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        
        # Calculate best metric if available
        if self.metrics_history:
            losses = [m["metrics"].get("loss") for m in self.metrics_history if "loss" in m["metrics"]]
            if losses:
                self.best_metric = min(losses)


@dataclass
class TrainingProvenance:
    """Complete training provenance for the RLHF pipeline."""
    pipeline_id: str
    start_time: str
    end_time: Optional[str] = None
    seed: Optional[int] = None
    reproducibility_hash: Optional[str] = None
    config_hash: Optional[str] = None
    
    # Environment information
    environment_info: Optional[Dict[str, Any]] = None
    
    # Training stages
    stages: List[TrainingStageProvenance] = None
    
    # Final model information
    final_model_path: Optional[str] = None
    model_card_path: Optional[str] = None
    
    # Evaluation results
    evaluation_results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = []
    
    def add_stage(self, stage: TrainingStageProvenance) -> None:
        """Add a training stage to the provenance."""
        self.stages.append(stage)
        logger.info(f"Added training stage to provenance: {stage.stage_name}")
    
    def get_stage(self, stage_name: str) -> Optional[TrainingStageProvenance]:
        """Get a training stage by name."""
        for stage in self.stages:
            if stage.stage_name == stage_name:
                return stage
        return None
    
    def finalize_training(self, end_time: Optional[str] = None,
                         final_model_path: Optional[str] = None,
                         evaluation_results: Optional[Dict[str, Any]] = None) -> None:
        """Finalize the complete training provenance."""
        self.end_time = end_time or datetime.now().isoformat()
        if final_model_path is not None:
            self.final_model_path = final_model_path
        if evaluation_results is not None:
            self.evaluation_results = evaluation_results
        
        logger.info("Training provenance finalized")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingProvenance':
        """Create from dictionary."""
        # Convert stage dictionaries back to TrainingStageProvenance objects
        stages_data = data.pop('stages', [])
        stages = [TrainingStageProvenance(**stage_data) for stage_data in stages_data]
        
        provenance = cls(**data)
        provenance.stages = stages
        return provenance
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save provenance to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Training provenance saved to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'TrainingProvenance':
        """Load provenance from JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Provenance file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


class TrainingProvenanceManager:
    """
    Manager for tracking training provenance throughout the RLHF pipeline.
    
    This class integrates with existing components to provide comprehensive
    provenance tracking and metadata management.
    """
    
    def __init__(self, config: Optional[Any] = None, 
                 pipeline_id: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize training provenance manager.
        
        Args:
            config: Training configuration object
            pipeline_id: Unique identifier for this training run
            seed: Random seed for reproducibility
        """
        self.config = config
        self.pipeline_id = pipeline_id or self._generate_pipeline_id()
        self.seed = seed
        
        # Initialize reproducibility manager
        self.repro_manager = ReproducibilityManager(seed=seed)
        
        # Initialize provenance
        self.provenance = TrainingProvenance(
            pipeline_id=self.pipeline_id,
            start_time=datetime.now().isoformat(),
            seed=seed
        )
        
        # Capture initial environment information
        self._capture_environment_info()
        
        logger.info(f"Training provenance manager initialized: {self.pipeline_id}")
    
    def _generate_pipeline_id(self) -> str:
        """Generate a unique pipeline identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"rlhf_phi3_{timestamp}"
    
    def _capture_environment_info(self) -> None:
        """Capture comprehensive environment information."""
        try:
            self.provenance.environment_info = self.repro_manager.log_environment_info()
            self.provenance.reproducibility_hash = self.repro_manager.create_reproducibility_hash()
            
            # Capture configuration hash if available
            if self.config is not None:
                if hasattr(self.config, 'create_checkpoint_snapshot'):
                    snapshot = self.config.create_checkpoint_snapshot(
                        self.pipeline_id, {"stage": "initialization"}
                    )
                    self.provenance.config_hash = snapshot.get("config_hash")
                
            logger.info("Environment information captured for provenance")
            
        except Exception as e:
            logger.warning(f"Failed to capture environment information: {e}")
    
    def start_stage(self, stage_name: str, config_hash: Optional[str] = None) -> TrainingStageProvenance:
        """
        Start tracking a new training stage.
        
        Args:
            stage_name: Name of the training stage (e.g., "sft", "reward", "ppo")
            config_hash: Optional configuration hash for this stage
            
        Returns:
            TrainingStageProvenance object for this stage
        """
        stage = TrainingStageProvenance(
            stage_name=stage_name,
            start_time=datetime.now().isoformat(),
            config_hash=config_hash
        )
        
        self.provenance.add_stage(stage)
        logger.info(f"Started tracking training stage: {stage_name}")
        
        return stage
    
    def update_stage_metrics(self, stage_name: str, step: int, 
                           metrics: Dict[str, float]) -> None:
        """
        Update metrics for a training stage.
        
        Args:
            stage_name: Name of the training stage
            step: Current training step
            metrics: Dictionary of metrics to record
        """
        stage = self.provenance.get_stage(stage_name)
        if stage is not None:
            stage.add_metric(step, metrics)
            stage.total_steps = max(stage.total_steps, step)
        else:
            logger.warning(f"Stage not found for metrics update: {stage_name}")
    
    def finalize_stage(self, stage_name: str, final_loss: Optional[float] = None,
                      checkpoint_path: Optional[str] = None) -> None:
        """
        Finalize a training stage.
        
        Args:
            stage_name: Name of the training stage
            final_loss: Final loss value for the stage
            checkpoint_path: Path to the final checkpoint for this stage
        """
        stage = self.provenance.get_stage(stage_name)
        if stage is not None:
            stage.finalize_stage(
                final_loss=final_loss,
                checkpoint_path=checkpoint_path
            )
            logger.info(f"Finalized training stage: {stage_name}")
        else:
            logger.warning(f"Stage not found for finalization: {stage_name}")
    
    def add_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        Add evaluation results to the provenance.
        
        Args:
            results: Dictionary containing evaluation results
        """
        self.provenance.evaluation_results = results
        logger.info("Evaluation results added to provenance")
    
    def finalize_training(self, final_model_path: Optional[str] = None) -> None:
        """
        Finalize the complete training provenance.
        
        Args:
            final_model_path: Path to the final trained model
        """
        self.provenance.finalize_training(
            final_model_path=final_model_path,
            evaluation_results=self.provenance.evaluation_results
        )
        logger.info("Training provenance finalized")
    
    def save_provenance(self, output_dir: Union[str, Path]) -> Path:
        """
        Save complete training provenance to file.
        
        Args:
            output_dir: Directory to save provenance files
            
        Returns:
            Path to the saved provenance file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main provenance file
        provenance_file = output_dir / f"{self.pipeline_id}_training_provenance.json"
        self.provenance.save_to_file(provenance_file)
        
        # Save environment information separately
        if self.provenance.environment_info:
            env_file = output_dir / f"{self.pipeline_id}_environment_info.json"
            with open(env_file, 'w') as f:
                json.dump(self.provenance.environment_info, f, indent=2, default=str)
            logger.info(f"Environment information saved to {env_file}")
        
        # Save configuration snapshot if available
        if self.config is not None and hasattr(self.config, 'save_checkpoint_snapshot'):
            config_file = output_dir / f"{self.pipeline_id}_config_snapshot.json"
            try:
                self.config.save_checkpoint_snapshot(
                    output_dir, self.pipeline_id, 
                    {"provenance_id": self.pipeline_id}
                )
                logger.info(f"Configuration snapshot saved to {config_file}")
            except Exception as e:
                logger.warning(f"Failed to save configuration snapshot: {e}")
        
        return provenance_file
    
    def create_training_fingerprint(self) -> str:
        """
        Create a unique fingerprint for this training run.
        
        Returns:
            SHA-256 hash representing the training fingerprint
        """
        if self.config is not None:
            try:
                from dataclasses import asdict
                config_dict = asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else self.config.__dict__
                return create_training_fingerprint(self.seed or 42, config_dict)
            except Exception as e:
                logger.warning(f"Failed to create training fingerprint: {e}")
        
        # Fallback fingerprint
        return self.repro_manager.create_reproducibility_hash()
    
    def get_provenance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training provenance.
        
        Returns:
            Dictionary containing provenance summary
        """
        summary = {
            "pipeline_id": self.pipeline_id,
            "start_time": self.provenance.start_time,
            "end_time": self.provenance.end_time,
            "seed": self.seed,
            "reproducibility_hash": self.provenance.reproducibility_hash,
            "config_hash": self.provenance.config_hash,
            "stages_completed": len(self.provenance.stages),
            "stage_names": [stage.stage_name for stage in self.provenance.stages],
            "has_evaluation_results": self.provenance.evaluation_results is not None,
            "final_model_path": self.provenance.final_model_path,
            "training_fingerprint": self.create_training_fingerprint()
        }
        
        # Add stage summaries
        stage_summaries = []
        for stage in self.provenance.stages:
            stage_summary = {
                "name": stage.stage_name,
                "total_steps": stage.total_steps,
                "final_loss": stage.final_loss,
                "completed": stage.end_time is not None,
                "checkpoint_available": stage.checkpoint_path is not None
            }
            stage_summaries.append(stage_summary)
        
        summary["stages"] = stage_summaries
        
        return summary
    
    def integrate_with_checkpoint(self, checkpoint_metadata: Any) -> None:
        """
        Integrate provenance information with checkpoint metadata.
        
        Args:
            checkpoint_metadata: Checkpoint metadata object to enhance
        """
        if hasattr(checkpoint_metadata, 'add_training_provenance'):
            try:
                # Create training metadata for checkpoint
                training_metadata = {
                    "pipeline_id": self.pipeline_id,
                    "training_fingerprint": self.create_training_fingerprint(),
                    "stages_completed": [stage.stage_name for stage in self.provenance.stages],
                    "provenance_summary": self.get_provenance_summary()
                }
                
                checkpoint_metadata.add_training_provenance(
                    config=self.config,
                    training_metadata=training_metadata,
                    include_environment=True
                )
                
                logger.info("Provenance integrated with checkpoint metadata")
                
            except Exception as e:
                logger.warning(f"Failed to integrate provenance with checkpoint: {e}")
        else:
            logger.warning("Checkpoint metadata does not support provenance integration")


# Convenience functions for easy integration

def create_provenance_manager(config: Any, seed: Optional[int] = None) -> TrainingProvenanceManager:
    """
    Create a training provenance manager with configuration.
    
    Args:
        config: Training configuration object
        seed: Optional random seed for reproducibility
        
    Returns:
        TrainingProvenanceManager instance
    """
    return TrainingProvenanceManager(config=config, seed=seed)


def track_training_stage(provenance_manager: TrainingProvenanceManager,
                        stage_name: str) -> TrainingStageProvenance:
    """
    Convenience function to start tracking a training stage.
    
    Args:
        provenance_manager: TrainingProvenanceManager instance
        stage_name: Name of the training stage
        
    Returns:
        TrainingStageProvenance object for the stage
    """
    return provenance_manager.start_stage(stage_name)


def save_training_provenance(provenance_manager: TrainingProvenanceManager,
                           output_dir: Union[str, Path]) -> Path:
    """
    Convenience function to save training provenance.
    
    Args:
        provenance_manager: TrainingProvenanceManager instance
        output_dir: Directory to save provenance files
        
    Returns:
        Path to the saved provenance file
    """
    return provenance_manager.save_provenance(output_dir)