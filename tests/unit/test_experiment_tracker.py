"""
Unit tests for Experiment Tracker

This module contains unit tests for the ExperimentTracker component,
testing specific functionality and edge cases.

Tests cover:
- Metric logging and hyperparameter tracking
- Visualization plot generation
- Run comparison functionality
- Configuration tracking
- Environment logging
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import pandas as pd
import numpy as np

from rlhf_phi3.config.config_manager import Config
from rlhf_phi3.tracking.experiment_tracker import ExperimentTracker


class TestExperimentTracker:
    """Unit tests for ExperimentTracker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        self.project_name = "test_project"
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ExperimentTracker initialization."""
        with patch('wandb.init'), patch('wandb.run', None):
            tracker = ExperimentTracker(self.project_name, self.config)
            
            assert tracker.project_name == self.project_name
            assert tracker.config == self.config
            assert tracker.current_run is None
            assert tracker.current_stage is None
            assert tracker.metrics_history == []
    
    def test_initialization_with_entity(self):
        """Test ExperimentTracker initialization with entity."""
        entity = "test_entity"
        
        with patch('wandb.init'), patch('wandb.run', None):
            tracker = ExperimentTracker(self.project_name, self.config, entity=entity)
            
            assert tracker.entity == entity
    
    def test_start_run_valid_stage(self):
        """Test starting a run with valid stage."""
        with patch('wandb.init') as mock_init, \
             patch('wandb.finish') as mock_finish, \
             patch('wandb.config') as mock_config:
            
            mock_run = Mock()
            mock_init.return_value = mock_run
            
            with patch('wandb.run', mock_run):
                tracker = ExperimentTracker(self.project_name, self.config)
                tracker.start_run("sft", "test_run")
                
                assert tracker.current_stage == "sft"
                assert tracker.current_run == mock_run
                
                # Verify WandB initialization was called
                mock_init.assert_called()
    
    def test_start_run_invalid_stage(self):
        """Test starting a run with invalid stage raises error."""
        with patch('wandb.init'), patch('wandb.run', None):
            tracker = ExperimentTracker(self.project_name, self.config)
            
            with pytest.raises(ValueError, match="Invalid stage 'invalid'"):
                tracker.start_run("invalid", "test_run")
    
    def test_log_metrics_with_active_run(self):
        """Test logging metrics with active run."""
        with patch('wandb.init'), patch('wandb.log') as mock_log:
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Set up active run
            mock_run = Mock()
            tracker.current_run = mock_run
            tracker.current_stage = "sft"
            
            metrics = {"loss": 0.5, "learning_rate": 0.001}
            step = 100
            
            tracker.log_metrics(metrics, step)
            
            # Verify metrics were logged to WandB
            expected_metrics = {
                "sft/loss": 0.5,
                "sft/learning_rate": 0.001,
                "step": 100,
                "stage": "sft"
            }
            mock_log.assert_called_once_with(expected_metrics, step=100)
            
            # Verify metrics were stored in history
            assert len(tracker.metrics_history) == 1
            history_entry = tracker.metrics_history[0]
            assert history_entry["step"] == 100
            assert history_entry["stage"] == "sft"
            assert history_entry["loss"] == 0.5
            assert history_entry["learning_rate"] == 0.001
    
    def test_log_metrics_without_active_run(self):
        """Test logging metrics without active run."""
        with patch('wandb.init'), patch('wandb.log') as mock_log:
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # No active run
            tracker.current_run = None
            
            metrics = {"loss": 0.5}
            step = 100
            
            tracker.log_metrics(metrics, step)
            
            # WandB log should not be called
            mock_log.assert_not_called()
            
            # But metrics should still be stored locally
            assert len(tracker.metrics_history) == 1
    
    def test_log_model_checkpoint(self):
        """Test logging model checkpoint as artifact."""
        with patch('wandb.init'), \
             patch('wandb.Artifact') as mock_artifact_class, \
             patch('wandb.log_artifact') as mock_log_artifact:
            
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Set up active run
            mock_run = Mock()
            tracker.current_run = mock_run
            
            # Create temporary checkpoint file
            checkpoint_path = Path(self.temp_dir) / "checkpoint.pt"
            checkpoint_path.write_text("fake checkpoint data")
            
            mock_artifact = Mock()
            mock_artifact_class.return_value = mock_artifact
            
            metadata = {"epoch": 1, "step": 100}
            
            tracker.log_model_checkpoint(str(checkpoint_path), "sft", metadata)
            
            # Verify artifact creation
            mock_artifact_class.assert_called_once()
            call_args = mock_artifact_class.call_args
            assert call_args[1]["type"] == "model_checkpoint"
            assert call_args[1]["metadata"] == metadata
            
            # Verify artifact was logged
            mock_log_artifact.assert_called_once_with(mock_artifact)
            
            # Verify file was added to artifact
            mock_artifact.add_file.assert_called_once_with(str(checkpoint_path))
    
    def test_log_evaluation_results(self):
        """Test logging evaluation results."""
        with patch('wandb.init'), \
             patch('wandb.log') as mock_log, \
             patch('wandb.Artifact') as mock_artifact_class, \
             patch('wandb.log_artifact') as mock_log_artifact:
            
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Set up active run
            mock_run = Mock()
            tracker.current_run = mock_run
            tracker.current_stage = "sft"
            
            mock_artifact = Mock()
            mock_artifact_class.return_value = mock_artifact
            
            results = {
                "mt_bench_score": 7.5,
                "helpfulness": 8.0,
                "tokens_per_second": 25.5,
                "metadata": {"model": "phi-3"}
            }
            
            tracker.log_evaluation_results(results)
            
            # Verify metrics were logged
            expected_metrics = {
                "sft/eval_mt_bench_score": 7.5,
                "sft/eval_helpfulness": 8.0,
                "sft/eval_tokens_per_second": 25.5
            }
            mock_log.assert_called_once_with(expected_metrics)
            
            # Verify artifact was created and logged
            mock_artifact_class.assert_called_once()
            mock_log_artifact.assert_called_once_with(mock_artifact)
    
    def test_create_training_plots_with_data(self):
        """Test creating training plots with metrics data."""
        with patch('wandb.init'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('wandb.log') as mock_wandb_log:
            
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Set up active run
            mock_run = Mock()
            tracker.current_run = mock_run
            
            # Create sample metrics history
            metrics_history = [
                {"step": 10, "stage": "sft", "loss": 2.0, "learning_rate": 0.001, "grad_norm": 1.5},
                {"step": 20, "stage": "sft", "loss": 1.8, "learning_rate": 0.0009, "grad_norm": 1.2},
                {"step": 30, "stage": "sft", "loss": 1.6, "learning_rate": 0.0008, "grad_norm": 1.0},
                {"step": 40, "stage": "reward", "loss": 0.8, "learning_rate": 0.0005, "grad_norm": 0.8},
                {"step": 50, "stage": "reward", "loss": 0.6, "learning_rate": 0.0004, "grad_norm": 0.6}
            ]
            
            plot_paths = tracker.create_training_plots(metrics_history, self.temp_dir)
            
            # Verify plots were generated
            assert isinstance(plot_paths, dict)
            assert len(plot_paths) > 0
            
            # Should have loss curves, learning rate, gradient norm, and stage comparison
            expected_plots = ["loss_curves", "learning_rate", "gradient_norm", "stage_comparison"]
            for expected_plot in expected_plots:
                assert any(expected_plot in plot_name for plot_name in plot_paths.keys())
            
            # Verify matplotlib savefig was called
            assert mock_savefig.call_count == len(plot_paths)
            
            # Verify plots were logged to WandB
            assert mock_wandb_log.call_count > 0
    
    def test_create_training_plots_empty_data(self):
        """Test creating training plots with empty data."""
        with patch('wandb.init'):
            tracker = ExperimentTracker(self.project_name, self.config)
            
            plot_paths = tracker.create_training_plots([])
            
            # Should return empty dict for empty data
            assert plot_paths == {}
    
    def test_finish_run(self):
        """Test finishing an experiment run."""
        with patch('wandb.init'), \
             patch('wandb.finish') as mock_finish, \
             patch('wandb.log') as mock_log:
            
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Set up active run with some metrics
            mock_run = Mock()
            tracker.current_run = mock_run
            tracker.current_stage = "sft"
            tracker.metrics_history = [
                {"step": 10, "stage": "sft", "loss": 2.0}
            ]
            
            with patch.object(tracker, 'create_training_plots') as mock_create_plots:
                tracker.finish_run()
            
            # Verify final summary was logged
            mock_log.assert_called()
            
            # Verify plots were created
            mock_create_plots.assert_called_once()
            
            # Verify WandB finish was called
            mock_finish.assert_called_once()
            
            # Verify state was reset
            assert tracker.current_run is None
            assert tracker.current_stage is None
    
    def test_get_run_comparison(self):
        """Test comparing multiple experiment runs."""
        with patch('wandb.init'), patch('wandb.Api') as mock_api_class:
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Mock API and runs
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            
            # Create mock runs
            mock_run1 = Mock()
            mock_run1.name = "run_1"
            mock_run1.state = "finished"
            mock_run1.created_at = "2024-01-01T00:00:00Z"
            mock_run1.config = {"learning_rate": 0.001, "batch_size": 4}
            mock_run1.summary = {"loss": 1.5, "accuracy": 0.85}
            
            mock_run2 = Mock()
            mock_run2.name = "run_2"
            mock_run2.state = "finished"
            mock_run2.created_at = "2024-01-02T00:00:00Z"
            mock_run2.config = {"learning_rate": 0.002, "batch_size": 4}
            mock_run2.summary = {"loss": 1.2, "accuracy": 0.88}
            
            def mock_run_getter(run_path):
                if "run1" in run_path:
                    return mock_run1
                elif "run2" in run_path:
                    return mock_run2
                raise Exception("Run not found")
            
            mock_api.run = mock_run_getter
            
            run_ids = ["run1", "run2"]
            metrics = ["loss", "accuracy"]
            
            comparison = tracker.get_run_comparison(run_ids, metrics)
            
            # Verify comparison structure
            assert "runs" in comparison
            assert "metrics_comparison" in comparison
            assert "config_comparison" in comparison
            
            # Verify runs data
            assert len(comparison["runs"]) == 2
            
            # Verify metrics comparison
            assert "loss" in comparison["metrics_comparison"]
            assert "accuracy" in comparison["metrics_comparison"]
            
            # Verify config comparison (should show learning_rate difference)
            assert "learning_rate" in comparison["config_comparison"]
    
    def test_create_comparison_plots(self):
        """Test creating comparison plots for multiple runs."""
        with patch('wandb.init'), \
             patch('wandb.Api') as mock_api_class, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Mock API
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            
            # Create mock run with history
            mock_run = Mock()
            mock_run.name = "test_run"
            mock_run.config = {"learning_rate": 0.001}
            
            # Create mock history DataFrame
            history_data = {
                "loss": [2.0, 1.8, 1.6, 1.4],
                "learning_rate": [0.001, 0.0009, 0.0008, 0.0007]
            }
            mock_history = pd.DataFrame(history_data)
            mock_run.history.return_value = mock_history
            
            mock_api.run.return_value = mock_run
            
            run_ids = ["run1"]
            metrics = ["loss", "learning_rate"]
            
            plot_paths = tracker.create_comparison_plots(run_ids, metrics, self.temp_dir)
            
            # Verify plots were created
            assert isinstance(plot_paths, dict)
            assert len(plot_paths) == len(metrics)
            
            # Verify plot names
            for metric in metrics:
                assert any(f"comparison_{metric}" in plot_name for plot_name in plot_paths.keys())
            
            # Verify matplotlib savefig was called
            assert mock_savefig.call_count == len(metrics)
    
    def test_export_metrics_history_csv(self):
        """Test exporting metrics history to CSV."""
        with patch('wandb.init'):
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Add some metrics history
            tracker.metrics_history = [
                {"step": 10, "stage": "sft", "loss": 2.0, "learning_rate": 0.001},
                {"step": 20, "stage": "sft", "loss": 1.8, "learning_rate": 0.0009}
            ]
            
            export_path = tracker.export_metrics_history("csv", 
                                                       str(Path(self.temp_dir) / "metrics.csv"))
            
            # Verify file was created
            assert Path(export_path).exists()
            
            # Verify content
            df = pd.read_csv(export_path)
            assert len(df) == 2
            assert "step" in df.columns
            assert "stage" in df.columns
            assert "loss" in df.columns
            assert "learning_rate" in df.columns
    
    def test_export_metrics_history_json(self):
        """Test exporting metrics history to JSON."""
        with patch('wandb.init'):
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Add some metrics history
            tracker.metrics_history = [
                {"step": 10, "stage": "sft", "loss": 2.0}
            ]
            
            export_path = tracker.export_metrics_history("json", 
                                                       str(Path(self.temp_dir) / "metrics.json"))
            
            # Verify file was created
            assert Path(export_path).exists()
            
            # Verify content
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 1
            assert data[0]["step"] == 10
            assert data[0]["stage"] == "sft"
            assert data[0]["loss"] == 2.0
    
    def test_export_metrics_history_empty(self):
        """Test exporting empty metrics history."""
        with patch('wandb.init'):
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # No metrics history
            export_path = tracker.export_metrics_history("csv")
            
            # Should return empty string
            assert export_path == ""
    
    def test_analyze_training_stability(self):
        """Test analyzing training stability."""
        with patch('wandb.init'):
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Create metrics history with stability patterns
            metrics_history = []
            for i in range(100):
                # Decreasing loss with some noise
                loss = 2.0 * np.exp(-i/50) + 0.1 * np.random.random()
                grad_norm = 1.0 + 0.2 * np.random.random()
                lr = 0.001 * (0.99 ** i)
                
                metrics_history.append({
                    "step": i * 10,
                    "stage": "sft",
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "learning_rate": lr
                })
            
            tracker.metrics_history = metrics_history
            
            analysis = tracker.analyze_training_stability(window_size=20)
            
            # Verify analysis structure
            assert "sft" in analysis
            sft_analysis = analysis["sft"]
            
            # Verify loss stability analysis
            assert "loss_stability" in sft_analysis
            loss_stability = sft_analysis["loss_stability"]
            assert "final_mean" in loss_stability
            assert "final_std" in loss_stability
            assert "coefficient_of_variation" in loss_stability
            assert "trend_slope" in loss_stability
            
            # Verify gradient stability analysis
            assert "gradient_stability" in sft_analysis
            grad_stability = sft_analysis["gradient_stability"]
            assert "final_grad_norm" in grad_stability
            assert "grad_norm_std" in grad_stability
            
            # Verify learning rate analysis
            assert "learning_rate_schedule" in sft_analysis
            lr_analysis = sft_analysis["learning_rate_schedule"]
            assert "initial_lr" in lr_analysis
            assert "final_lr" in lr_analysis
            assert "lr_decay_ratio" in lr_analysis
    
    def test_generate_performance_report(self):
        """Test generating performance report."""
        with patch('wandb.init'):
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Add metrics history
            tracker.metrics_history = [
                {"step": 10, "stage": "sft", "loss": 2.0, "learning_rate": 0.001},
                {"step": 20, "stage": "sft", "loss": 1.8, "learning_rate": 0.0009},
                {"step": 30, "stage": "reward", "loss": 0.8, "learning_rate": 0.0005},
                {"step": 40, "stage": "reward", "loss": 0.6, "learning_rate": 0.0004}
            ]
            
            baseline_metrics = {"loss": 2.5, "learning_rate": 0.001}
            
            report = tracker.generate_performance_report(baseline_metrics)
            
            # Verify report structure
            assert "timestamp" in report
            assert "project" in report
            assert "training_summary" in report
            assert "stage_performance" in report
            assert "baseline_comparison" in report
            assert "efficiency_metrics" in report
            
            # Verify training summary
            training_summary = report["training_summary"]
            assert training_summary["stages_completed"] == 2
            assert training_summary["total_steps"] == 40
            assert set(training_summary["stages"]) == {"sft", "reward"}
            
            # Verify stage performance
            stage_performance = report["stage_performance"]
            assert "sft" in stage_performance
            assert "reward" in stage_performance
            
            # Verify baseline comparison
            baseline_comparison = report["baseline_comparison"]
            assert "loss" in baseline_comparison
            loss_comparison = baseline_comparison["loss"]
            assert "baseline" in loss_comparison
            assert "current" in loss_comparison
            assert "improvement_pct" in loss_comparison
    
    def test_save_experiment_snapshot(self):
        """Test saving experiment snapshot."""
        with patch('wandb.init'), \
             patch('wandb.log_artifact') as mock_log_artifact:
            
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Set up active run
            mock_run = Mock()
            tracker.current_run = mock_run
            
            # Add some metrics history
            tracker.metrics_history = [
                {"step": 10, "stage": "sft", "loss": 2.0}
            ]
            
            snapshot_name = "test_snapshot"
            
            with patch.object(tracker, 'create_training_plots') as mock_create_plots:
                snapshot_path = tracker.save_experiment_snapshot(snapshot_name)
            
            # Verify snapshot was created
            assert isinstance(snapshot_path, str)
            assert len(snapshot_path) > 0
            
            snapshot_dir = Path(snapshot_path)
            
            if snapshot_dir.exists():
                # Verify required files exist
                assert (snapshot_dir / "config.json").exists()
                assert (snapshot_dir / "metrics_history.csv").exists()
                assert (snapshot_dir / "environment.json").exists()
                assert (snapshot_dir / "manifest.json").exists()
                
                # Verify plots were created
                mock_create_plots.assert_called_once()
                
                # Verify artifact was logged
                mock_log_artifact.assert_called_once()
    
    def test_create_experiment_documentation(self):
        """Test creating experiment documentation."""
        with patch('wandb.init'), \
             patch('wandb.log_artifact') as mock_log_artifact:
            
            tracker = ExperimentTracker(self.project_name, self.config)
            
            # Set up active run
            mock_run = Mock()
            tracker.current_run = mock_run
            
            # Add some metrics history
            tracker.metrics_history = [
                {"step": 10, "stage": "sft", "loss": 2.0, "learning_rate": 0.001},
                {"step": 20, "stage": "sft", "loss": 1.8, "learning_rate": 0.0009}
            ]
            
            experiment_name = "Test Experiment"
            description = "This is a test experiment for unit testing."
            results_summary = {
                "final_loss": 1.8,
                "improvement": "10%",
                "performance_metrics": {
                    "tokens_per_second": 25.5,
                    "memory_usage": "8GB"
                }
            }
            
            doc_path = tracker.create_experiment_documentation(
                experiment_name, description, results_summary,
                str(Path(self.temp_dir) / "experiment_doc.md")
            )
            
            # Verify documentation was created
            assert Path(doc_path).exists()
            
            # Verify content
            with open(doc_path, 'r') as f:
                content = f.read()
            
            assert experiment_name in content
            assert description in content
            assert "final_loss" in content
            assert "Configuration" in content
            assert "Results Summary" in content
            
            # Verify artifact was logged
            mock_log_artifact.assert_called_once()