"""
Unit tests for Google Colab utilities.

Tests specific functionality of Colab integration components including
Google Drive mounting, session management, progress tracking, and error handling.
"""

import pytest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
from typing import Dict, Any

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from notebooks.colab_utils import (
    ColabSessionManager,
    ColabDriveManager,
    ColabProgressTracker,
    ColabEnvironmentSetup,
    setup_colab_session,
    monitor_session_health,
    handle_training_error,
    emergency_checkpoint_save,
    create_session_resume_guide
)


class TestColabSessionManager:
    """Test ColabSessionManager functionality."""
    
    def test_session_initialization(self):
        """Test session manager initialization."""
        session_manager = ColabSessionManager(session_timeout_hours=10.0)
        
        assert session_manager.session_timeout_hours == 10.0
        assert session_manager.drive_mounted is False
        assert session_manager.drive_mount_path == "/content/drive"
        assert isinstance(session_manager.session_start_time, datetime)
    
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    def test_get_memory_usage(self, mock_cuda_available, mock_virtual_memory):
        """Test memory usage reporting."""
        # Mock system memory
        mock_vm = Mock()
        mock_vm.total = 16 * 1024**3  # 16GB
        mock_vm.available = 8 * 1024**3  # 8GB
        mock_vm.used = 8 * 1024**3  # 8GB
        mock_vm.percent = 50.0
        mock_virtual_memory.return_value = mock_vm
        
        # Mock CUDA not available
        mock_cuda_available.return_value = False
        
        session_manager = ColabSessionManager()
        memory_info = session_manager.get_memory_usage()
        
        assert "system_memory" in memory_info
        assert "gpu_memory" in memory_info
        
        sys_mem = memory_info["system_memory"]
        assert sys_mem["total_gb"] == 16.0
        assert sys_mem["available_gb"] == 8.0
        assert sys_mem["used_gb"] == 8.0
        assert sys_mem["percent_used"] == 50.0
        
        # No GPU memory when CUDA not available
        assert len(memory_info["gpu_memory"]) == 0
    
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_get_memory_usage_with_gpu(self, mock_reserved, mock_allocated, 
                                      mock_props, mock_device_count, 
                                      mock_cuda_available, mock_virtual_memory):
        """Test memory usage reporting with GPU."""
        # Mock system memory
        mock_vm = Mock()
        mock_vm.total = 16 * 1024**3
        mock_vm.available = 8 * 1024**3
        mock_vm.used = 8 * 1024**3
        mock_vm.percent = 50.0
        mock_virtual_memory.return_value = mock_vm
        
        # Mock GPU
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        mock_gpu_props = Mock()
        mock_gpu_props.name = "Tesla T4"
        mock_gpu_props.total_memory = 15 * 1024**3  # 15GB
        mock_props.return_value = mock_gpu_props
        
        mock_allocated.return_value = 5 * 1024**3  # 5GB allocated
        mock_reserved.return_value = 6 * 1024**3   # 6GB reserved
        
        session_manager = ColabSessionManager()
        memory_info = session_manager.get_memory_usage()
        
        assert "gpu_0" in memory_info["gpu_memory"]
        gpu_mem = memory_info["gpu_memory"]["gpu_0"]
        
        assert gpu_mem["name"] == "Tesla T4"
        assert gpu_mem["total_gb"] == 15.0
        assert gpu_mem["allocated_gb"] == 5.0
        assert gpu_mem["reserved_gb"] == 6.0
        assert gpu_mem["free_gb"] == 9.0  # 15 - 6
        assert gpu_mem["percent_used"] == 40.0  # 6/15 * 100
    
    def test_check_memory_pressure(self):
        """Test memory pressure detection."""
        session_manager = ColabSessionManager()
        
        # Mock high memory usage
        with patch.object(session_manager, 'get_memory_usage') as mock_memory:
            mock_memory.return_value = {
                "system_memory": {"percent_used": 95.0},
                "gpu_memory": {
                    "gpu_0": {"percent_used": 85.0}
                }
            }
            
            pressure = session_manager.check_memory_pressure(threshold_percent=90.0)
            
            assert pressure["system_pressure"] is True  # 95% > 90%
            assert pressure["gpu_pressure"] is False   # 85% < 90%
            assert pressure["critical_level"] is True  # system_pressure OR gpu_pressure
    
    @patch('gc.collect')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    def test_optimize_memory(self, mock_sync, mock_empty_cache, mock_cuda_available, mock_gc):
        """Test memory optimization."""
        mock_cuda_available.return_value = True
        
        session_manager = ColabSessionManager()
        
        # Mock memory usage before and after
        initial_memory = {
            "system_memory": {"used_gb": 10.0},
            "gpu_memory": {"gpu_0": {"reserved_gb": 8.0}}
        }
        final_memory = {
            "system_memory": {"used_gb": 9.0},
            "gpu_memory": {"gpu_0": {"reserved_gb": 6.0}}
        }
        
        with patch.object(session_manager, 'get_memory_usage', side_effect=[initial_memory, final_memory]):
            result = session_manager.optimize_memory()
            
            # Verify cleanup calls
            assert mock_gc.call_count >= 2  # Called at least twice
            mock_empty_cache.assert_called_once()
            mock_sync.assert_called_once()
            
            # Verify memory freed calculation
            assert result["memory_freed"]["system_gb"] == 1.0  # 10 - 9
            assert result["memory_freed"]["gpu_gb"] == 2.0     # 8 - 6
    
    def test_estimate_time_remaining(self):
        """Test session time estimation."""
        session_manager = ColabSessionManager(session_timeout_hours=12.0)
        
        # Mock session started 10 hours ago
        start_time = datetime.now() - timedelta(hours=10)
        session_manager.session_start_time = start_time
        
        recommendations = session_manager.estimate_time_remaining()
        
        assert abs(recommendations["remaining_hours"] - 2.0) < 0.1  # ~2 hours remaining
        assert recommendations["should_checkpoint"] is True   # < 1 hour threshold
        assert recommendations["should_prepare_resume"] is False  # > 0.5 hour threshold
        assert recommendations["emergency_save"] is False    # > 0.1 hour threshold
    
    def test_save_and_load_session_state(self):
        """Test session state persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_manager = ColabSessionManager()
            
            # Mock drive manager
            drive_manager = ColabDriveManager(temp_dir)
            drive_manager.mounted = True
            
            # Create MyDrive structure
            mydrive_path = os.path.join(temp_dir, "MyDrive")
            os.makedirs(mydrive_path, exist_ok=True)
            
            # Test data
            state_data = {
                "current_stage": "sft",
                "current_step": 150,
                "model_path": "/path/to/model"
            }
            
            # Save state
            saved_path = session_manager.save_session_state(state_data, drive_manager)
            assert os.path.exists(saved_path)
            
            # Load state
            loaded_state = session_manager.load_session_state(drive_manager)
            
            assert loaded_state is not None
            assert loaded_state["state_data"] == state_data
            assert "session_info" in loaded_state
            assert "timestamp" in loaded_state


class TestColabDriveManager:
    """Test ColabDriveManager functionality."""
    
    def test_drive_manager_initialization(self):
        """Test drive manager initialization."""
        drive_manager = ColabDriveManager("/custom/mount/path")
        
        assert drive_manager.mount_path == "/custom/mount/path"
        assert drive_manager.mounted is False
        assert drive_manager.authenticated is False
    
    def test_is_mounted_check(self):
        """Test drive mount detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            drive_manager = ColabDriveManager(temp_dir)
            
            # Initially not mounted
            assert drive_manager.is_mounted() is False
            
            # Create MyDrive directory
            mydrive_path = os.path.join(temp_dir, "MyDrive")
            os.makedirs(mydrive_path)
            
            # Now should be detected as mounted
            assert drive_manager.is_mounted() is True
    
    def test_setup_project_directory(self):
        """Test project directory structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            drive_manager = ColabDriveManager(temp_dir)
            drive_manager.mounted = True
            
            # Create MyDrive
            mydrive_path = os.path.join(temp_dir, "MyDrive")
            os.makedirs(mydrive_path)
            
            # Setup project directory
            project_path = "test-project"
            full_path = drive_manager.setup_project_directory(project_path)
            
            # Verify directory structure
            expected_dirs = [
                full_path,
                os.path.join(full_path, "checkpoints"),
                os.path.join(full_path, "logs"),
                os.path.join(full_path, "models"),
                os.path.join(full_path, "configs"),
                os.path.join(full_path, "results")
            ]
            
            for directory in expected_dirs:
                assert os.path.exists(directory)
                assert os.path.isdir(directory)
    
    @patch('subprocess.run')
    def test_sync_file_to_drive(self, mock_subprocess):
        """Test file synchronization to Drive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            drive_manager = ColabDriveManager(temp_dir)
            drive_manager.mounted = True
            
            # Create test file
            local_file = os.path.join(temp_dir, "test_file.txt")
            with open(local_file, 'w') as f:
                f.write("test content")
            
            drive_file = os.path.join(temp_dir, "MyDrive", "test_file.txt")
            
            # Mock successful subprocess
            mock_subprocess.return_value = Mock(returncode=0)
            
            result = drive_manager.sync_file(local_file, drive_file, "to_drive")
            
            assert result is True
            mock_subprocess.assert_called_once_with(
                ["cp", "-r", local_file, drive_file], 
                check=True
            )
    
    @patch('subprocess.run')
    def test_sync_file_from_drive(self, mock_subprocess):
        """Test file synchronization from Drive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            drive_manager = ColabDriveManager(temp_dir)
            drive_manager.mounted = True
            
            # Create drive file
            drive_file = os.path.join(temp_dir, "MyDrive", "test_file.txt")
            os.makedirs(os.path.dirname(drive_file), exist_ok=True)
            with open(drive_file, 'w') as f:
                f.write("test content")
            
            local_file = os.path.join(temp_dir, "local_test_file.txt")
            
            # Mock successful subprocess
            mock_subprocess.return_value = Mock(returncode=0)
            
            result = drive_manager.sync_file(local_file, drive_file, "from_drive")
            
            assert result is True
            mock_subprocess.assert_called_once_with(
                ["cp", "-r", drive_file, local_file], 
                check=True
            )


class TestColabProgressTracker:
    """Test ColabProgressTracker functionality."""
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker initialization."""
        tracker = ColabProgressTracker()
        
        assert len(tracker.active_operations) == 0
        assert len(tracker.error_messages) == 0
        assert len(tracker.troubleshooting_guides) == 0
    
    def test_operation_tracking_lifecycle(self):
        """Test complete operation tracking lifecycle."""
        tracker = ColabProgressTracker()
        
        # Start operation
        operation_id = "test_op_123"
        tracker.start_operation(operation_id, "Test Operation", 100)
        
        # Verify operation started
        active_ops = tracker.get_active_operations()
        assert operation_id in active_ops
        assert active_ops[operation_id]["name"] == "Test Operation"
        assert active_ops[operation_id]["total_steps"] == 100
        assert active_ops[operation_id]["current_step"] == 0
        assert active_ops[operation_id]["status"] == "running"
        
        # Update progress
        tracker.update_progress(operation_id, 50, "Halfway done")
        
        updated_ops = tracker.get_active_operations()
        assert updated_ops[operation_id]["current_step"] == 50
        
        # Complete operation
        tracker.complete_operation(operation_id)
        
        final_ops = tracker.get_active_operations()
        assert operation_id not in final_ops
    
    def test_progress_bar_generation(self):
        """Test progress bar generation."""
        tracker = ColabProgressTracker()
        
        # Test with known total steps
        operation_id = "test_progress"
        tracker.start_operation(operation_id, "Progress Test", 100)
        tracker.update_progress(operation_id, 25)
        
        progress_bar = tracker.display_progress_bar(operation_id)
        
        assert "Progress Test" in progress_bar
        assert "25.0%" in progress_bar
        assert "25/100" in progress_bar
        assert "█" in progress_bar  # Filled portion
        assert "░" in progress_bar  # Empty portion
        
        # Test with unknown total steps
        operation_id_2 = "test_unknown"
        tracker.start_operation(operation_id_2, "Unknown Progress", None)
        tracker.update_progress(operation_id_2, 42)
        
        progress_bar_2 = tracker.display_progress_bar(operation_id_2)
        
        assert "Unknown Progress" in progress_bar_2
        assert "Step 42" in progress_bar_2
        assert "%" not in progress_bar_2  # No percentage for unknown total
    
    def test_error_message_management(self):
        """Test error message and troubleshooting guide management."""
        tracker = ColabProgressTracker()
        
        # Add error message
        error_type = "test_error"
        message = "This is a test error"
        troubleshooting = ["Step 1", "Step 2", "Step 3"]
        
        tracker.add_error_message(error_type, message, troubleshooting)
        
        # Retrieve error guidance
        guidance = tracker.get_error_guidance(error_type)
        
        assert guidance is not None
        assert guidance["message"] == message
        assert guidance["troubleshooting"] == troubleshooting
        assert "timestamp" in guidance
    
    def test_common_error_guides_setup(self):
        """Test setup of common error guides."""
        tracker = ColabProgressTracker()
        tracker.setup_common_error_guides()
        
        # Verify common error types are setup
        common_errors = [
            "gpu_memory_error",
            "session_timeout",
            "drive_mount_error",
            "dataset_loading_error",
            "model_loading_error",
            "training_divergence"
        ]
        
        for error_type in common_errors:
            guidance = tracker.get_error_guidance(error_type)
            assert guidance is not None
            assert len(guidance["message"]) > 0
            assert len(guidance["troubleshooting"]) > 0
    
    def test_context_manager_tracking(self):
        """Test context manager for operation tracking."""
        tracker = ColabProgressTracker()
        
        with tracker.track_operation("Context Test", 50) as operation_id:
            # Verify operation is active
            active_ops = tracker.get_active_operations()
            assert operation_id in active_ops
            assert active_ops[operation_id]["name"] == "Context Test"
            assert active_ops[operation_id]["total_steps"] == 50
        
        # Verify operation is completed after context exit
        final_ops = tracker.get_active_operations()
        assert operation_id not in final_ops
    
    def test_resume_instructions_generation(self):
        """Test generation of resume instructions."""
        tracker = ColabProgressTracker()
        
        checkpoint_path = "/content/drive/MyDrive/checkpoint.pt"
        stage = "sft"
        step = 150
        
        instructions = tracker.create_resume_instructions(checkpoint_path, stage, step)
        
        assert isinstance(instructions, str)
        assert len(instructions) > 0
        assert checkpoint_path in instructions
        assert stage in instructions
        assert str(step) in instructions
        assert "resume_from_stage" in instructions
        assert "TrainingOrchestrator" in instructions


class TestColabEnvironmentSetup:
    """Test ColabEnvironmentSetup functionality."""
    
    def test_environment_setup_initialization(self):
        """Test environment setup initialization."""
        env_setup = ColabEnvironmentSetup()
        
        assert env_setup.setup_complete is False
        assert len(env_setup.installed_packages) == 0
    
    @patch('subprocess.run')
    def test_install_dependencies_success(self, mock_subprocess):
        """Test successful dependency installation."""
        env_setup = ColabEnvironmentSetup()
        
        # Mock successful installation
        mock_subprocess.return_value = Mock(returncode=0, stderr="", stdout="")
        
        packages = ["torch", "transformers", "datasets"]
        result = env_setup.install_dependencies(packages=packages)
        
        assert result is True
        assert len(env_setup.installed_packages) == 3
        assert "torch" in env_setup.installed_packages
        assert "transformers" in env_setup.installed_packages
        assert "datasets" in env_setup.installed_packages
    
    @patch('subprocess.run')
    def test_install_dependencies_failure(self, mock_subprocess):
        """Test failed dependency installation."""
        env_setup = ColabEnvironmentSetup()
        
        # Mock failed installation
        from subprocess import CalledProcessError
        mock_subprocess.side_effect = CalledProcessError(1, "pip", stderr="Installation failed")
        
        packages = ["nonexistent-package"]
        result = env_setup.install_dependencies(packages=packages)
        
        assert result is False
        assert len(env_setup.installed_packages) == 0
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    def test_get_system_info(self, mock_device_name, mock_cuda_available):
        """Test system information gathering."""
        env_setup = ColabEnvironmentSetup()
        
        # Mock GPU available
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "Tesla T4"
        
        with patch('torch.cuda.device_count', return_value=1), \
             patch('torch.version.cuda', "11.8"):
            
            system_info = env_setup.get_system_info()
            
            assert "python_version" in system_info
            assert "torch_version" in system_info
            assert system_info["cuda_available"] is True
            assert system_info["gpu_count"] == 1
            assert system_info["gpu_names"] == ["Tesla T4"]
            assert system_info["cuda_version"] == "11.8"


class TestConvenienceFunctions:
    """Test convenience functions for easy notebook usage."""
    
    @patch('notebooks.colab_utils.ColabEnvironmentSetup.setup_colab_environment')
    def test_setup_colab_session(self, mock_env_setup):
        """Test complete Colab session setup."""
        # Mock successful environment setup
        mock_env_setup.return_value = {
            "success": True,
            "gpu_available": True,
            "drive_mounted": True,
            "dependencies_installed": True,
            "config_loaded": False
        }
        
        session_mgr, drive_mgr, progress_tracker = setup_colab_session()
        
        assert isinstance(session_mgr, ColabSessionManager)
        assert isinstance(drive_mgr, ColabDriveManager)
        assert isinstance(progress_tracker, ColabProgressTracker)
        
        # Verify environment setup was called
        mock_env_setup.assert_called_once()
    
    def test_handle_training_error_classification(self):
        """Test training error classification and handling."""
        # Test GPU memory error
        gpu_error = Exception("CUDA out of memory")
        error_info = handle_training_error(gpu_error, "During model training")
        
        assert error_info["error_type"] == "gpu_memory_error"
        assert error_info["original_error"] == "CUDA out of memory"
        assert error_info["context"] == "During model training"
        assert error_info["guidance"] is not None
        
        # Test session timeout error
        timeout_error = Exception("Session timeout occurred")
        error_info = handle_training_error(timeout_error)
        
        assert error_info["error_type"] == "session_timeout"
        
        # Test unknown error
        unknown_error = Exception("Some random error")
        error_info = handle_training_error(unknown_error)
        
        assert error_info["error_type"] == "unknown_error"
        assert error_info["guidance"] is None
    
    def test_create_session_resume_guide(self):
        """Test session resume guide creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory for file operations
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                checkpoint_path = "/content/drive/MyDrive/checkpoint.pt"
                stage = "reward"
                step = 250
                
                instructions = create_session_resume_guide(
                    checkpoint_path, stage, step, save_to_drive=False
                )
                
                # Verify instructions content
                assert isinstance(instructions, str)
                assert checkpoint_path in instructions
                assert stage in instructions
                assert str(step) in instructions
                
                # Verify file was created
                assert os.path.exists("resume_instructions.md")
                
                with open("resume_instructions.md", 'r') as f:
                    file_content = f.read()
                    assert file_content == instructions
                    
            finally:
                os.chdir(original_cwd)
    
    @patch('notebooks.colab_utils.ColabSessionManager')
    def test_monitor_session_health(self, mock_session_manager):
        """Test session health monitoring."""
        # Mock session manager
        mock_manager = Mock()
        mock_session_manager.return_value = mock_manager
        
        # Mock health info
        mock_manager.get_session_info.return_value = {"elapsed_hours": 2.0}
        mock_manager.check_memory_pressure.return_value = {"critical_level": False}
        mock_manager.estimate_time_remaining.return_value = {
            "remaining_hours": 10.0,
            "should_checkpoint": False
        }
        
        health_info = monitor_session_health(display_warnings=False)
        
        assert "session_info" in health_info
        assert "memory_pressure" in health_info
        assert "time_recommendations" in health_info
        
        # Verify manager methods were called
        mock_manager.get_session_info.assert_called_once()
        mock_manager.check_memory_pressure.assert_called_once()
        mock_manager.estimate_time_remaining.assert_called_once()
    
    def test_emergency_checkpoint_save(self):
        """Test emergency checkpoint save functionality."""
        # Mock checkpoint manager
        mock_checkpoint_manager = Mock()
        mock_checkpoint_manager.save_checkpoint.return_value = "/path/to/emergency/checkpoint.pt"
        
        # Mock model and optimizer
        mock_model = Mock()
        mock_optimizer = Mock()
        
        checkpoint_path = emergency_checkpoint_save(
            mock_checkpoint_manager, mock_model, mock_optimizer, 
            step=100, stage="sft"
        )
        
        assert checkpoint_path == "/path/to/emergency/checkpoint.pt"
        
        # Verify checkpoint manager was called with emergency flag
        mock_checkpoint_manager.save_checkpoint.assert_called_once_with(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=0,
            step=100,
            stage="sft",
            emergency=True
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])