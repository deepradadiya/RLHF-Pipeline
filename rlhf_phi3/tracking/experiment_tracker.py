"""
Experiment Tracker for RLHF Phi-3 Pipeline

This module provides comprehensive experiment tracking and visualization using
Weights & Biases (WandB). It handles metric logging, hyperparameter tracking,
artifact management, and training visualization across all training stages.

Requirements satisfied:
- 6.1: Training metrics logging to Weights & Biases
- 6.2: Hyperparameter and configuration tracking for reproducibility
- 6.3: Training visualization plots for loss curves and metric trends
- 6.4: Model checkpoint artifact logging for version control
- 6.5: Experiment comparison and analysis capabilities
- 15.1: Configuration snapshot saving with checkpoints
- 15.3: Environment and library version logging
"""

import os
import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import asdict

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from packaging import version

from ..config.config_manager import Config

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Comprehensive experiment tracking using Weights & Biases.
    
    This class provides experiment tracking capabilities including metric logging,
    hyperparameter tracking, artifact management, and visualization generation.
    It integrates with the RLHF pipeline to track training across all stages.
    
    Requirements satisfied:
    - 6.1: Training metrics logging
    - 6.2: Hyperparameter and configuration tracking
    - 6.3: Training visualization generation
    - 6.4: Model checkpoint artifact logging
    - 6.5: Experiment comparison capabilities
    """
    
    def __init__(self, project_name: str, config: Config, entity: Optional[str] = None):
        """
        Initialize the experiment tracker.
        
        Args:
            project_name: WandB project name
            config: Configuration object containing all hyperparameters
            entity: WandB entity (team/user), uses default if None
            
        Requirement 6.2: Hyperparameter and configuration tracking
        """
        self.project_name = project_name
        self.config = config
        self.entity = entity or config.wandb.entity
        self.current_run = None
        self.current_stage = None
        self.metrics_history = []
        
        # Set up matplotlib for headless operation (Colab compatibility)
        plt.switch_backend('Agg')
        sns.set_style("whitegrid")
        
        # Initialize WandB if not already done
        if not wandb.run:
            self._initialize_wandb()
        
        logger.info(f"ExperimentTracker initialized for project: {project_name}")
    
    def _initialize_wandb(self) -> None:
        """Initialize WandB with proper configuration."""
        try:
            # Check if WandB is properly configured
            if not os.getenv("WANDB_API_KEY"):
                logger.warning("WANDB_API_KEY not found. Run 'wandb login' to authenticate.")
            
            # Set WandB mode based on environment
            wandb_mode = os.getenv("WANDB_MODE", "online")
            
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                mode=wandb_mode,
                tags=self.config.wandb.tags,
                save_code=True
            )
            
            logger.info("WandB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            logger.warning("Continuing without WandB tracking")
    
    def start_run(self, stage: str, run_name: str, resume: bool = False) -> None:
        """
        Start a new experiment run for a specific training stage.
        
        Args:
            stage: Training stage ('sft', 'reward', or 'ppo')
            run_name: Name for the run
            resume: Whether to resume an existing run
            
        Requirement 6.1: Training metrics logging
        Requirement 6.2: Hyperparameter and configuration tracking
        Requirement 15.3: Environment logging
        """
        if stage not in ["sft", "reward", "ppo"]:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: sft, reward, ppo")
        
        self.current_stage = stage
        
        try:
            # Create run name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_run_name = f"{stage}_{run_name}_{timestamp}"
            
            # Start or resume WandB run
            if wandb.run and not resume:
                wandb.finish()
            
            if not wandb.run or not resume:
                wandb.init(
                    project=self.project_name,
                    entity=self.entity,
                    name=full_run_name,
                    tags=self.config.wandb.tags + [stage],
                    reinit=True,
                    resume="allow" if resume else False
                )
            
            # Log configuration and environment information
            self._log_configuration(stage)
            self._log_environment_info()
            
            self.current_run = wandb.run
            logger.info(f"Started experiment run: {full_run_name}")
            
        except Exception as e:
            logger.error(f"Failed to start experiment run: {e}")
            self.current_run = None
    
    def _log_configuration(self, stage: str) -> None:
        """
        Log configuration and hyperparameters for the current stage.
        
        Args:
            stage: Training stage to log configuration for
            
        Requirement 6.2: Hyperparameter and configuration tracking
        Requirement 15.1: Configuration snapshot completeness
        """
        try:
            # Get stage-specific configuration
            stage_config = self.config.get_stage_config(stage)
            
            # Log all configuration as WandB config
            wandb.config.update(stage_config)
            
            # Also log the complete configuration as an artifact
            config_dict = asdict(self.config)
            config_artifact = wandb.Artifact(
                name=f"config_{stage}",
                type="configuration",
                description=f"Complete configuration for {stage} stage"
            )
            
            # Save configuration to temporary file and add to artifact
            config_path = Path(f"/tmp/config_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            config_artifact.add_file(str(config_path))
            wandb.log_artifact(config_artifact)
            
            # Clean up temporary file
            config_path.unlink(missing_ok=True)
            
            logger.info(f"Configuration logged for stage: {stage}")
            
        except Exception as e:
            logger.error(f"Failed to log configuration: {e}")
    
    def _log_environment_info(self) -> None:
        """
        Log environment and library version information.
        
        Requirement 15.3: Environment logging completeness
        """
        try:
            env_info = {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "architecture": platform.architecture()[0],
                "hostname": platform.node(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Get library versions
            libraries = [
                "torch", "transformers", "peft", "trl", "datasets",
                "accelerate", "wandb", "matplotlib", "seaborn", "numpy", "pandas"
            ]
            
            library_versions = {}
            for lib in libraries:
                try:
                    if lib == "torch":
                        import torch
                        library_versions[lib] = torch.__version__
                    elif lib == "transformers":
                        import transformers
                        library_versions[lib] = transformers.__version__
                    elif lib == "peft":
                        import peft
                        library_versions[lib] = peft.__version__
                    elif lib == "trl":
                        import trl
                        library_versions[lib] = trl.__version__
                    elif lib == "datasets":
                        import datasets
                        library_versions[lib] = datasets.__version__
                    elif lib == "accelerate":
                        import accelerate
                        library_versions[lib] = accelerate.__version__
                    elif lib == "wandb":
                        library_versions[lib] = wandb.__version__
                    elif lib == "matplotlib":
                        library_versions[lib] = plt.matplotlib.__version__
                    elif lib == "seaborn":
                        library_versions[lib] = sns.__version__
                    elif lib == "numpy":
                        library_versions[lib] = np.__version__
                    elif lib == "pandas":
                        library_versions[lib] = pd.__version__
                except ImportError:
                    library_versions[lib] = "not_installed"
                except AttributeError:
                    library_versions[lib] = "version_unknown"
            
            env_info["library_versions"] = library_versions
            
            # Log GPU information if available
            try:
                import torch
                if torch.cuda.is_available():
                    env_info["gpu_count"] = torch.cuda.device_count()
                    env_info["gpu_name"] = torch.cuda.get_device_name(0)
                    env_info["cuda_version"] = torch.version.cuda
                else:
                    env_info["gpu_available"] = False
            except Exception:
                env_info["gpu_info"] = "unavailable"
            
            # Log environment info to WandB
            wandb.config.update({"environment": env_info})
            
            logger.info("Environment information logged")
            
        except Exception as e:
            logger.error(f"Failed to log environment info: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int, stage: Optional[str] = None) -> None:
        """
        Log training metrics to WandB and store in history.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step number
            stage: Training stage (uses current stage if None)
            
        Requirement 6.1: Training metrics logging
        """
        if not self.current_run:
            logger.warning("No active run. Metrics not logged to WandB.")
            return
        
        stage = stage or self.current_stage
        
        try:
            # Prefix metrics with stage name for clarity
            prefixed_metrics = {}
            for key, value in metrics.items():
                if not key.startswith(f"{stage}/"):
                    prefixed_metrics[f"{stage}/{key}"] = value
                else:
                    prefixed_metrics[key] = value
            
            # Add step information
            prefixed_metrics["step"] = step
            prefixed_metrics["stage"] = stage
            
            # Log to WandB
            wandb.log(prefixed_metrics, step=step)
            
            # Store in local history
            metric_entry = {
                "step": step,
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
                **metrics
            }
            self.metrics_history.append(metric_entry)
            
            logger.debug(f"Logged metrics for step {step}: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model_checkpoint(self, checkpoint_path: str, stage: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log model checkpoint as WandB artifact.
        
        Args:
            checkpoint_path: Path to the checkpoint file/directory
            stage: Training stage
            metadata: Additional metadata for the checkpoint
            
        Requirement 6.4: Model checkpoint artifact logging
        """
        if not self.current_run:
            logger.warning("No active run. Checkpoint not logged to WandB.")
            return
        
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
                return
            
            # Create artifact name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifact_name = f"model_checkpoint_{stage}_{timestamp}"
            
            # Create WandB artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model_checkpoint",
                description=f"Model checkpoint for {stage} stage",
                metadata=metadata or {}
            )
            
            # Add checkpoint files to artifact
            if checkpoint_path.is_file():
                artifact.add_file(str(checkpoint_path))
            else:
                artifact.add_dir(str(checkpoint_path))
            
            # Log artifact
            wandb.log_artifact(artifact)
            
            logger.info(f"Checkpoint logged as artifact: {artifact_name}")
            
        except Exception as e:
            logger.error(f"Failed to log checkpoint: {e}")
    
    def log_evaluation_results(self, results: Dict[str, Any], stage: Optional[str] = None) -> None:
        """
        Log evaluation results and metrics.
        
        Args:
            results: Dictionary containing evaluation results
            stage: Training stage (uses current stage if None)
            
        Requirement 6.1: Training metrics logging
        """
        if not self.current_run:
            logger.warning("No active run. Evaluation results not logged to WandB.")
            return
        
        stage = stage or self.current_stage
        
        try:
            # Prefix evaluation metrics
            eval_metrics = {}
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    eval_metrics[f"{stage}/eval_{key}"] = value
            
            # Log evaluation metrics
            wandb.log(eval_metrics)
            
            # Create evaluation artifact for detailed results
            eval_artifact = wandb.Artifact(
                name=f"evaluation_results_{stage}",
                type="evaluation",
                description=f"Detailed evaluation results for {stage} stage"
            )
            
            # Save detailed results to temporary file
            results_path = Path(f"/tmp/eval_results_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            eval_artifact.add_file(str(results_path))
            wandb.log_artifact(eval_artifact)
            
            # Clean up temporary file
            results_path.unlink(missing_ok=True)
            
            logger.info(f"Evaluation results logged for stage: {stage}")
            
        except Exception as e:
            logger.error(f"Failed to log evaluation results: {e}")
    
    def create_training_plots(self, metrics_history: Optional[List[Dict]] = None, 
                            save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Generate training visualization plots for loss curves and metric trends.
        
        Args:
            metrics_history: List of metric dictionaries (uses stored history if None)
            save_path: Directory to save plots (optional)
            
        Returns:
            Dictionary mapping plot names to file paths
            
        Requirement 6.3: Training visualization generation
        """
        metrics_history = metrics_history or self.metrics_history
        
        if not metrics_history:
            logger.warning("No metrics history available for plotting")
            return {}
        
        try:
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(metrics_history)
            
            if df.empty:
                logger.warning("Empty metrics DataFrame")
                return {}
            
            plot_paths = {}
            
            # Create save directory if specified
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot 1: Loss curves by stage
            if 'loss' in df.columns or any('loss' in col for col in df.columns):
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for stage in df['stage'].unique():
                    stage_df = df[df['stage'] == stage]
                    
                    # Find loss columns for this stage
                    loss_cols = [col for col in stage_df.columns if 'loss' in col.lower()]
                    
                    for loss_col in loss_cols:
                        if stage_df[loss_col].notna().any():
                            ax.plot(stage_df['step'], stage_df[loss_col], 
                                   label=f"{stage}_{loss_col}", marker='o', markersize=3)
                
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss Curves')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                loss_plot_path = self._save_plot(fig, "loss_curves", save_path)
                plot_paths["loss_curves"] = loss_plot_path
                plt.close(fig)
            
            # Plot 2: Learning rate schedule
            if 'learning_rate' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for stage in df['stage'].unique():
                    stage_df = df[df['stage'] == stage]
                    if stage_df['learning_rate'].notna().any():
                        ax.plot(stage_df['step'], stage_df['learning_rate'], 
                               label=f"{stage}", marker='o', markersize=3)
                
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Learning Rate')
                ax.set_title('Learning Rate Schedule')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                
                lr_plot_path = self._save_plot(fig, "learning_rate", save_path)
                plot_paths["learning_rate"] = lr_plot_path
                plt.close(fig)
            
            # Plot 3: Gradient norm
            if 'grad_norm' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for stage in df['stage'].unique():
                    stage_df = df[df['stage'] == stage]
                    if stage_df['grad_norm'].notna().any():
                        ax.plot(stage_df['step'], stage_df['grad_norm'], 
                               label=f"{stage}", marker='o', markersize=3)
                
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Gradient Norm')
                ax.set_title('Gradient Norm Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                grad_plot_path = self._save_plot(fig, "gradient_norm", save_path)
                plot_paths["gradient_norm"] = grad_plot_path
                plt.close(fig)
            
            # Plot 4: Stage comparison summary
            if len(df['stage'].unique()) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                # Plot metrics by stage
                metrics_to_plot = ['loss', 'learning_rate', 'grad_norm']
                available_metrics = [m for m in metrics_to_plot if m in df.columns]
                
                for i, metric in enumerate(available_metrics[:4]):
                    ax = axes[i]
                    
                    for stage in df['stage'].unique():
                        stage_df = df[df['stage'] == stage]
                        if stage_df[metric].notna().any():
                            ax.plot(stage_df['step'], stage_df[metric], 
                                   label=f"{stage}", marker='o', markersize=2)
                    
                    ax.set_xlabel('Training Step')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_title(f'{metric.replace("_", " ").title()} by Stage')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    if metric == 'learning_rate':
                        ax.set_yscale('log')
                
                # Hide unused subplots
                for i in range(len(available_metrics), 4):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                summary_plot_path = self._save_plot(fig, "stage_comparison", save_path)
                plot_paths["stage_comparison"] = summary_plot_path
                plt.close(fig)
            
            # Log plots to WandB if run is active
            if self.current_run:
                for plot_name, plot_path in plot_paths.items():
                    if Path(plot_path).exists():
                        wandb.log({f"plots/{plot_name}": wandb.Image(plot_path)})
            
            logger.info(f"Generated {len(plot_paths)} training plots")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Failed to create training plots: {e}")
            return {}
    
    def _save_plot(self, fig, name: str, save_path: Optional[str] = None) -> str:
        """Save a matplotlib figure and return the path."""
        if save_path:
            plot_path = Path(save_path) / f"{name}.png"
        else:
            plot_path = Path(f"/tmp/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        return str(plot_path)
    
    def finish_run(self) -> None:
        """
        Finish the current experiment run.
        
        Requirement 6.1: Training metrics logging
        """
        try:
            if self.current_run:
                # Generate final plots before finishing
                if self.metrics_history:
                    self.create_training_plots()
                
                # Log final summary
                wandb.log({
                    "run_finished": True,
                    "total_metrics_logged": len(self.metrics_history),
                    "final_timestamp": datetime.now().isoformat()
                })
                
                wandb.finish()
                logger.info("Experiment run finished")
            
            self.current_run = None
            self.current_stage = None
            
        except Exception as e:
            logger.error(f"Failed to finish run: {e}")
    
    def get_run_comparison(self, run_ids: List[str], 
                          metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple experiment runs.
        
        Args:
            run_ids: List of WandB run IDs to compare
            metrics: List of metrics to compare (compares all if None)
            
        Returns:
            Dictionary containing comparison results
            
        Requirement 6.5: Run comparison capability
        """
        try:
            api = wandb.Api()
            
            comparison_data = {
                "runs": [],
                "metrics_comparison": {},
                "config_comparison": {}
            }
            
            for run_id in run_ids:
                try:
                    run = api.run(f"{self.entity}/{self.project_name}/{run_id}")
                    
                    run_data = {
                        "id": run_id,
                        "name": run.name,
                        "state": run.state,
                        "created_at": run.created_at,
                        "config": dict(run.config),
                        "summary": dict(run.summary)
                    }
                    
                    comparison_data["runs"].append(run_data)
                    
                except Exception as e:
                    logger.error(f"Failed to fetch run {run_id}: {e}")
                    continue
            
            if not comparison_data["runs"]:
                logger.warning("No valid runs found for comparison")
                return comparison_data
            
            # Compare metrics
            if metrics:
                for metric in metrics:
                    metric_values = {}
                    for run_data in comparison_data["runs"]:
                        if metric in run_data["summary"]:
                            metric_values[run_data["name"]] = run_data["summary"][metric]
                    
                    if metric_values:
                        comparison_data["metrics_comparison"][metric] = metric_values
            
            # Compare configurations
            config_keys = set()
            for run_data in comparison_data["runs"]:
                config_keys.update(run_data["config"].keys())
            
            for key in config_keys:
                config_values = {}
                for run_data in comparison_data["runs"]:
                    if key in run_data["config"]:
                        config_values[run_data["name"]] = run_data["config"][key]
                
                if len(set(config_values.values())) > 1:  # Only include if values differ
                    comparison_data["config_comparison"][key] = config_values
            
            logger.info(f"Compared {len(comparison_data['runs'])} runs")
            return comparison_data
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return {"error": str(e)}
    
    def create_comparison_plots(self, run_ids: List[str], 
                              metrics: List[str],
                              save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Create comparison plots for multiple runs.
        
        Args:
            run_ids: List of WandB run IDs to compare
            metrics: List of metrics to plot
            save_path: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
            
        Requirement 6.5: Run comparison capability
        """
        try:
            api = wandb.Api()
            plot_paths = {}
            
            # Create save directory if specified
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Fetch run data
            runs_data = []
            for run_id in run_ids:
                try:
                    run = api.run(f"{self.entity}/{self.project_name}/{run_id}")
                    history = run.history()
                    
                    runs_data.append({
                        "name": run.name,
                        "history": history,
                        "config": dict(run.config)
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to fetch run {run_id}: {e}")
                    continue
            
            if not runs_data:
                logger.warning("No valid run data found for plotting")
                return {}
            
            # Create comparison plots for each metric
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for run_data in runs_data:
                    history = run_data["history"]
                    if metric in history.columns:
                        # Filter out NaN values
                        valid_data = history[history[metric].notna()]
                        if not valid_data.empty:
                            ax.plot(valid_data.index, valid_data[metric], 
                                   label=run_data["name"], marker='o', markersize=3)
                
                ax.set_xlabel('Step')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_path = self._save_plot(fig, f"comparison_{metric}", save_path)
                plot_paths[f"comparison_{metric}"] = plot_path
                plt.close(fig)
            
            logger.info(f"Generated {len(plot_paths)} comparison plots")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Failed to create comparison plots: {e}")
            return {}
    
    def export_metrics_history(self, format: str = "csv", 
                             save_path: Optional[str] = None) -> str:
        """
        Export metrics history to file.
        
        Args:
            format: Export format ('csv', 'json', or 'parquet')
            save_path: File path to save (auto-generated if None)
            
        Returns:
            Path to the exported file
        """
        if not self.metrics_history:
            logger.warning("No metrics history to export")
            return ""
        
        try:
            df = pd.DataFrame(self.metrics_history)
            
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"/tmp/metrics_history_{timestamp}.{format}"
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                df.to_csv(save_path, index=False)
            elif format.lower() == "json":
                df.to_json(save_path, orient="records", indent=2)
            elif format.lower() == "parquet":
                df.to_parquet(save_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Metrics history exported to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to export metrics history: {e}")
            return ""
    
    def create_experiment_documentation(self, 
                                      experiment_name: str,
                                      description: str,
                                      results_summary: Dict[str, Any],
                                      save_path: Optional[str] = None) -> str:
        """
        Create comprehensive experiment documentation.
        
        Args:
            experiment_name: Name of the experiment
            description: Description of the experiment
            results_summary: Summary of key results
            save_path: Path to save documentation (auto-generated if None)
            
        Returns:
            Path to the documentation file
            
        Requirement 6.4: Comprehensive experiment documentation
        """
        try:
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"/tmp/experiment_doc_{timestamp}.md"
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate documentation content
            doc_content = self._generate_experiment_doc(
                experiment_name, description, results_summary
            )
            
            with open(save_path, 'w') as f:
                f.write(doc_content)
            
            # Log documentation as artifact if run is active
            if self.current_run:
                doc_artifact = wandb.Artifact(
                    name=f"experiment_documentation",
                    type="documentation",
                    description=f"Comprehensive documentation for {experiment_name}"
                )
                doc_artifact.add_file(str(save_path))
                wandb.log_artifact(doc_artifact)
            
            logger.info(f"Experiment documentation created: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to create experiment documentation: {e}")
            return ""
    
    def _generate_experiment_doc(self, name: str, description: str, 
                               results: Dict[str, Any]) -> str:
        """Generate markdown documentation for an experiment."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        doc = f"""# Experiment Documentation: {name}

**Generated:** {timestamp}
**Project:** {self.project_name}

## Description

{description}

## Configuration

### Model Configuration
- **Model:** {self.config.model.name}
- **Max Length:** {self.config.model.max_length}
- **Device:** {self.config.model.device}

### LoRA Configuration
- **Rank (r):** {self.config.lora.r}
- **Alpha:** {self.config.lora.alpha}
- **Dropout:** {self.config.lora.dropout}
- **Target Modules:** {', '.join(self.config.lora.target_modules)}

### Training Configuration

#### SFT Stage
- **Epochs:** {self.config.training.sft.epochs}
- **Learning Rate:** {self.config.training.sft.learning_rate}
- **Batch Size:** {self.config.training.sft.batch_size}
- **Gradient Accumulation Steps:** {self.config.training.sft.gradient_accumulation_steps}
- **Max Steps:** {self.config.training.sft.max_steps}

#### Reward Model Stage
- **Epochs:** {self.config.training.reward.epochs}
- **Learning Rate:** {self.config.training.reward.learning_rate}
- **Batch Size:** {self.config.training.reward.batch_size}
- **Gradient Accumulation Steps:** {self.config.training.reward.gradient_accumulation_steps}
- **Max Steps:** {self.config.training.reward.max_steps}

#### PPO Stage
- **Learning Rate:** {self.config.training.ppo.learning_rate}
- **Batch Size:** {self.config.training.ppo.batch_size}
- **Mini Batch Size:** {self.config.training.ppo.mini_batch_size}
- **PPO Epochs:** {self.config.training.ppo.ppo_epochs}
- **Max Steps:** {self.config.training.ppo.max_steps}

### Datasets
- **SFT Dataset:** {self.config.datasets.sft.name} ({self.config.datasets.sft.max_samples} samples)
- **Preference Dataset:** {self.config.datasets.preference.name} ({self.config.datasets.preference.max_samples} samples)

## Results Summary

"""
        
        # Add results section
        for key, value in results.items():
            if isinstance(value, dict):
                doc += f"### {key.replace('_', ' ').title()}\n\n"
                for subkey, subvalue in value.items():
                    doc += f"- **{subkey.replace('_', ' ').title()}:** {subvalue}\n"
                doc += "\n"
            else:
                doc += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        # Add metrics history summary if available
        if self.metrics_history:
            doc += "\n## Training Metrics Summary\n\n"
            df = pd.DataFrame(self.metrics_history)
            
            # Group by stage and get final metrics
            for stage in df['stage'].unique():
                stage_df = df[df['stage'] == stage]
                if not stage_df.empty:
                    doc += f"### {stage.upper()} Stage\n\n"
                    
                    # Get final values for key metrics
                    final_row = stage_df.iloc[-1]
                    for col in ['loss', 'learning_rate', 'grad_norm']:
                        if col in final_row and pd.notna(final_row[col]):
                            doc += f"- **Final {col.replace('_', ' ').title()}:** {final_row[col]:.6f}\n"
                    
                    doc += f"- **Total Steps:** {final_row['step']}\n"
                    doc += "\n"
        
        doc += """
## Environment Information

"""
        
        # Add environment info if available from WandB config
        if self.current_run and hasattr(wandb.config, 'environment'):
            env_info = wandb.config.environment
            doc += f"- **Python Version:** {env_info.get('python_version', 'Unknown')}\n"
            doc += f"- **Platform:** {env_info.get('platform', 'Unknown')}\n"
            
            if 'library_versions' in env_info:
                doc += "\n### Library Versions\n\n"
                for lib, version in env_info['library_versions'].items():
                    doc += f"- **{lib}:** {version}\n"
        
        doc += """
## Reproducibility

To reproduce this experiment:

1. Use the exact configuration shown above
2. Ensure the same library versions are installed
3. Use the same datasets with identical preprocessing
4. Set the same random seeds if deterministic training was used

## Notes

- All metrics and artifacts are logged to Weights & Biases
- Model checkpoints are saved as WandB artifacts
- Training plots and visualizations are automatically generated

---

*This documentation was automatically generated by the RLHF Phi-3 Pipeline Experiment Tracker.*
"""
        
        return doc
    
    def analyze_training_stability(self, window_size: int = 50) -> Dict[str, Any]:
        """
        Analyze training stability across different stages.
        
        Args:
            window_size: Window size for rolling statistics
            
        Returns:
            Dictionary containing stability analysis results
            
        Requirement 6.5: Experiment comparison and analysis
        """
        if not self.metrics_history:
            logger.warning("No metrics history available for stability analysis")
            return {}
        
        try:
            df = pd.DataFrame(self.metrics_history)
            analysis = {}
            
            for stage in df['stage'].unique():
                stage_df = df[df['stage'] == stage].copy()
                stage_analysis = {}
                
                # Analyze loss stability
                if 'loss' in stage_df.columns:
                    loss_data = stage_df['loss'].dropna()
                    if len(loss_data) > window_size:
                        # Calculate rolling statistics
                        rolling_mean = loss_data.rolling(window=window_size).mean()
                        rolling_std = loss_data.rolling(window=window_size).std()
                        
                        # Stability metrics
                        stage_analysis['loss_stability'] = {
                            'final_mean': rolling_mean.iloc[-1],
                            'final_std': rolling_std.iloc[-1],
                            'coefficient_of_variation': rolling_std.iloc[-1] / rolling_mean.iloc[-1],
                            'trend_slope': self._calculate_trend_slope(loss_data),
                            'convergence_step': self._find_convergence_step(loss_data, window_size)
                        }
                
                # Analyze gradient norm stability
                if 'grad_norm' in stage_df.columns:
                    grad_data = stage_df['grad_norm'].dropna()
                    if len(grad_data) > window_size:
                        rolling_std = grad_data.rolling(window=window_size).std()
                        
                        stage_analysis['gradient_stability'] = {
                            'final_grad_norm': grad_data.iloc[-1],
                            'grad_norm_std': rolling_std.iloc[-1],
                            'exploding_gradients': (grad_data > 10.0).sum(),
                            'vanishing_gradients': (grad_data < 1e-6).sum()
                        }
                
                # Analyze learning rate adaptation
                if 'learning_rate' in stage_df.columns:
                    lr_data = stage_df['learning_rate'].dropna()
                    if len(lr_data) > 1:
                        stage_analysis['learning_rate_schedule'] = {
                            'initial_lr': lr_data.iloc[0],
                            'final_lr': lr_data.iloc[-1],
                            'lr_decay_ratio': lr_data.iloc[-1] / lr_data.iloc[0],
                            'lr_changes': (lr_data.diff().abs() > 1e-8).sum()
                        }
                
                analysis[stage] = stage_analysis
            
            logger.info("Training stability analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze training stability: {e}")
            return {}
    
    def _calculate_trend_slope(self, data: pd.Series) -> float:
        """Calculate the trend slope using linear regression."""
        try:
            from sklearn.linear_model import LinearRegression
            
            x = np.arange(len(data)).reshape(-1, 1)
            y = data.values
            
            model = LinearRegression()
            model.fit(x, y)
            
            return model.coef_[0]
            
        except Exception:
            # Fallback to simple slope calculation
            if len(data) < 2:
                return 0.0
            return (data.iloc[-1] - data.iloc[0]) / len(data)
    
    def _find_convergence_step(self, data: pd.Series, window_size: int) -> Optional[int]:
        """Find the step where training appears to converge."""
        try:
            if len(data) < window_size * 2:
                return None
            
            rolling_std = data.rolling(window=window_size).std()
            
            # Find where rolling std becomes consistently small
            threshold = rolling_std.mean() * 0.1  # 10% of average std
            
            for i in range(window_size, len(rolling_std) - window_size):
                if all(rolling_std.iloc[i:i+window_size] < threshold):
                    return data.index[i]
            
            return None
            
        except Exception:
            return None
    
    def generate_performance_report(self, baseline_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            Dictionary containing performance analysis
            
        Requirement 6.5: Experiment comparison and analysis
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "project": self.project_name,
                "total_runs": len(self.metrics_history) if self.metrics_history else 0
            }
            
            if not self.metrics_history:
                report["status"] = "no_data"
                return report
            
            df = pd.DataFrame(self.metrics_history)
            
            # Overall training summary
            report["training_summary"] = {
                "stages_completed": df['stage'].nunique(),
                "total_steps": df['step'].max(),
                "training_duration": len(df),
                "stages": list(df['stage'].unique())
            }
            
            # Performance by stage
            stage_performance = {}
            for stage in df['stage'].unique():
                stage_df = df[df['stage'] == stage]
                
                stage_perf = {
                    "steps": len(stage_df),
                    "final_step": stage_df['step'].max()
                }
                
                # Add final metrics
                final_metrics = stage_df.iloc[-1]
                for metric in ['loss', 'learning_rate', 'grad_norm']:
                    if metric in final_metrics and pd.notna(final_metrics[metric]):
                        stage_perf[f"final_{metric}"] = final_metrics[metric]
                
                # Calculate improvement
                if 'loss' in stage_df.columns:
                    loss_data = stage_df['loss'].dropna()
                    if len(loss_data) > 1:
                        improvement = (loss_data.iloc[0] - loss_data.iloc[-1]) / loss_data.iloc[0]
                        stage_perf["loss_improvement_pct"] = improvement * 100
                
                stage_performance[stage] = stage_perf
            
            report["stage_performance"] = stage_performance
            
            # Comparison with baseline if provided
            if baseline_metrics:
                comparison = {}
                final_row = df.iloc[-1]
                
                for metric, baseline_value in baseline_metrics.items():
                    if metric in final_row and pd.notna(final_row[metric]):
                        current_value = final_row[metric]
                        improvement = (baseline_value - current_value) / baseline_value
                        comparison[metric] = {
                            "baseline": baseline_value,
                            "current": current_value,
                            "improvement_pct": improvement * 100,
                            "better": improvement > 0
                        }
                
                report["baseline_comparison"] = comparison
            
            # Training efficiency metrics
            if 'loss' in df.columns:
                loss_data = df['loss'].dropna()
                if len(loss_data) > 1:
                    report["efficiency_metrics"] = {
                        "convergence_rate": abs(loss_data.iloc[-1] - loss_data.iloc[0]) / len(loss_data),
                        "stability_score": 1.0 / (1.0 + loss_data.std()),
                        "final_loss": loss_data.iloc[-1],
                        "best_loss": loss_data.min()
                    }
            
            logger.info("Performance report generated")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}
    
    def save_experiment_snapshot(self, snapshot_name: str, 
                               include_artifacts: bool = True) -> str:
        """
        Save a complete snapshot of the current experiment.
        
        Args:
            snapshot_name: Name for the snapshot
            include_artifacts: Whether to include model artifacts
            
        Returns:
            Path to the snapshot directory
            
        Requirement 15.1: Configuration snapshot completeness
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_dir = Path(f"/tmp/experiment_snapshot_{snapshot_name}_{timestamp}")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_path = snapshot_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            # Save metrics history
            if self.metrics_history:
                metrics_path = snapshot_dir / "metrics_history.csv"
                df = pd.DataFrame(self.metrics_history)
                df.to_csv(metrics_path, index=False)
            
            # Save environment info
            env_path = snapshot_dir / "environment.json"
            env_info = {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(env_path, 'w') as f:
                json.dump(env_info, f, indent=2)
            
            # Generate and save plots
            if self.metrics_history:
                plots_dir = snapshot_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                self.create_training_plots(save_path=str(plots_dir))
            
            # Create snapshot manifest
            manifest = {
                "snapshot_name": snapshot_name,
                "timestamp": timestamp,
                "project": self.project_name,
                "files": [
                    "config.json",
                    "metrics_history.csv",
                    "environment.json",
                    "plots/"
                ]
            }
            
            manifest_path = snapshot_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Log snapshot as WandB artifact if run is active
            if self.current_run and include_artifacts:
                snapshot_artifact = wandb.Artifact(
                    name=f"experiment_snapshot_{snapshot_name}",
                    type="experiment_snapshot",
                    description=f"Complete experiment snapshot: {snapshot_name}"
                )
                snapshot_artifact.add_dir(str(snapshot_dir))
                wandb.log_artifact(snapshot_artifact)
            
            logger.info(f"Experiment snapshot saved: {snapshot_dir}")
            return str(snapshot_dir)
            
        except Exception as e:
            logger.error(f"Failed to save experiment snapshot: {e}")
            return ""