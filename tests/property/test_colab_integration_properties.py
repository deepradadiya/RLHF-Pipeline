"""
Property-based tests for Google Colab integration components.

These tests validate universal properties that should hold across all valid
executions of the Colab integration utilities.
"""

import pytest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from notebooks.colab_utils import (
    ColabSessionManager,
    ColabDriveManager, 
    ColabProgressTracker,
    ColabEnvironmentSetup,
    monitor_session_health,
    handle_training_error
)


class TestProgressFeedbackConsistency:
    """
    **Property 31: Progress Feedback Consistency**
    **Validates: Requirement 13.4**
    
    For any long-running operation, the RLHF_Pipeline SHALL display 
    progress bars and status updates.
    """
    
    @given(
        operation_name=st.text(min_size=1, max_size=50),
        total_steps=st.one_of(st.none(), st.integers(min_value=1, max_value=1000)),
        current_steps=st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=20)
    )
    @settings(max_examples=50, deadline=5000)
    def test_progress_tracking_consistency(self, operation_name: str, total_steps: int, current_steps: List[int]):
        """Test that progress tracking provides consistent feedback for any operation."""
        # Filter current_steps to be valid for total_steps
        if total_steps is not None:
            current_steps = [step for step in current_steps if step <= total_steps]
            assume(len(current_steps) > 0)
        
        tracker = ColabProgressTracker()
        operation_id = f"test_op_{int(time.time())}"
        
        # Start operation
        tracker.start_operation(operation_id, operation_name, total_steps)
        
        # Verify operation is tracked
        active_ops = tracker.get_active_operations()
        assert operation_id in active_ops
        assert active_ops[operation_id]["name"] == operation_name
        assert active_ops[operation_id]["total_steps"] == total_steps
        assert active_ops[operation_id]["status"] == "running"
        
        # Test progress updates
        for step in current_steps:
            tracker.update_progress(operation_id, step, f"Step {step}")
            
            # Verify progress is updated
            updated_ops = tracker.get_active_operations()
            assert updated_ops[operation_id]["current_step"] == step
            
            # Verify progress bar generation
            progress_bar = tracker.display_progress_bar(operation_id)
            assert isinstance(progress_bar, str)
            assert len(progress_bar) > 0
            
            if total_steps is not None:
                # Progress bar should contain percentage for known total
                assert "%" in progress_bar
                assert f"{step}/{total_steps}" in progress_bar
            else:
                # Progress bar should show step count for unknown total
                assert f"Step {step}" in progress_bar
        
        # Complete operation
        tracker.complete_operation(operation_id)
        
        # Verify operation is removed from active operations
        final_ops = tracker.get_active_operations()
        assert operation_id not in final_ops
    
    @given(
        operations=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=30),  # operation_name
                st.one_of(st.none(), st.integers(min_value=1, max_value=100))  # total_steps
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_multiple_operations_tracking(self, operations: List[tuple]):
        """Test that multiple concurrent operations are tracked consistently."""
        tracker = ColabProgressTracker()
        operation_ids = []
        
        # Start all operations
        for i, (name, total_steps) in enumerate(operations):
            op_id = f"multi_op_{i}_{int(time.time())}"
            operation_ids.append(op_id)
            tracker.start_operation(op_id, name, total_steps)
        
        # Verify all operations are tracked
        active_ops = tracker.get_active_operations()
        assert len(active_ops) == len(operations)
        
        for i, op_id in enumerate(operation_ids):
            assert op_id in active_ops
            assert active_ops[op_id]["name"] == operations[i][0]
            assert active_ops[op_id]["total_steps"] == operations[i][1]
        
        # Update progress for each operation
        for i, op_id in enumerate(operation_ids):
            step = min(10, operations[i][1] if operations[i][1] else 10)
            tracker.update_progress(op_id, step)
            
            # Verify individual operation progress
            updated_ops = tracker.get_active_operations()
            assert updated_ops[op_id]["current_step"] == step
        
        # Complete operations one by one
        for op_id in operation_ids:
            tracker.complete_operation(op_id)
            remaining_ops = tracker.get_active_operations()
            assert op_id not in remaining_ops
        
        # Verify all operations are completed
        final_ops = tracker.get_active_operations()
        assert len(final_ops) == 0
    
    @given(
        error_types=st.lists(
            st.sampled_from([
                "gpu_memory_error",
                "session_timeout", 
                "drive_mount_error",
                "dataset_loading_error",
                "model_loading_error",
                "training_divergence"
            ]),
            min_size=1,
            max_size=6
        )
    )
    @settings(max_examples=20, deadline=3000)
    def test_error_guidance_consistency(self, error_types: List[str]):
        """Test that error guidance is consistently provided for all error types."""
        tracker = ColabProgressTracker()
        tracker.setup_common_error_guides()
        
        for error_type in error_types:
            guidance = tracker.get_error_guidance(error_type)
            
            # Verify guidance exists and has required structure
            assert guidance is not None
            assert "message" in guidance
            assert "troubleshooting" in guidance
            assert "timestamp" in guidance
            
            # Verify message is non-empty
            assert isinstance(guidance["message"], str)
            assert len(guidance["message"]) > 0
            
            # Verify troubleshooting steps exist
            assert isinstance(guidance["troubleshooting"], list)
            assert len(guidance["troubleshooting"]) > 0
            
            # Verify all troubleshooting steps are non-empty strings
            for step in guidance["troubleshooting"]:
                assert isinstance(step, str)
                assert len(step.strip()) > 0
    
    @given(
        session_timeout_hours=st.floats(min_value=0.1, max_value=24.0),
        elapsed_hours=st.floats(min_value=0.0, max_value=23.9)
    )
    @settings(max_examples=30, deadline=3000)
    def test_session_monitoring_consistency(self, session_timeout_hours: float, elapsed_hours: float):
        """Test that session monitoring provides consistent feedback."""
        assume(elapsed_hours < session_timeout_hours)
        
        # Mock session start time
        start_time = datetime.now() - timedelta(hours=elapsed_hours)
        
        with patch('notebooks.colab_utils.datetime') as mock_datetime:
            mock_datetime.now.return_value = start_time + timedelta(hours=elapsed_hours)
            
            session_manager = ColabSessionManager(session_timeout_hours)
            session_manager.session_start_time = start_time
            
            session_info = session_manager.get_session_info()
            
            # Verify session info consistency
            assert "elapsed_hours" in session_info
            assert "remaining_hours" in session_info
            assert abs(session_info["elapsed_hours"] - elapsed_hours) < 0.1
            
            remaining_expected = session_timeout_hours - elapsed_hours
            assert abs(session_info["remaining_hours"] - remaining_expected) < 0.1
            
            # Test time recommendations
            recommendations = session_manager.estimate_time_remaining()
            
            assert "remaining_hours" in recommendations
            assert "should_checkpoint" in recommendations
            assert "should_prepare_resume" in recommendations
            assert "emergency_save" in recommendations
            
            # Verify recommendation logic consistency
            if recommendations["remaining_hours"] < 0.1:
                assert recommendations["emergency_save"] is True
                assert recommendations["should_prepare_resume"] is True
                assert recommendations["should_checkpoint"] is True
            elif recommendations["remaining_hours"] < 0.5:
                assert recommendations["should_prepare_resume"] is True
                assert recommendations["should_checkpoint"] is True
            elif recommendations["remaining_hours"] < 1.0:
                assert recommendations["should_checkpoint"] is True


class TestMemoryOptimizationConsistency:
    """Test memory optimization and monitoring consistency."""
    
    @given(
        memory_threshold=st.floats(min_value=50.0, max_value=99.0)
    )
    @settings(max_examples=20, deadline=3000)
    def test_memory_pressure_detection_consistency(self, memory_threshold: float):
        """Test that memory pressure detection is consistent."""
        session_manager = ColabSessionManager()
        
        # Mock memory usage
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value = Mock(
                total=16 * 1024**3,  # 16GB
                available=4 * 1024**3,  # 4GB available
                used=12 * 1024**3,  # 12GB used
                percent=75.0
            )
            
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.device_count', return_value=1), \
                 patch('torch.cuda.get_device_properties') as mock_props, \
                 patch('torch.cuda.memory_allocated', return_value=int(memory_threshold/100 * 15 * 1024**3)), \
                 patch('torch.cuda.memory_reserved', return_value=int(memory_threshold/100 * 15 * 1024**3)):
                
                mock_props.return_value = Mock(
                    name="Tesla T4",
                    total_memory=15 * 1024**3  # 15GB T4
                )
                
                pressure = session_manager.check_memory_pressure(memory_threshold)
                
                # Verify pressure detection structure
                assert "system_pressure" in pressure
                assert "gpu_pressure" in pressure
                assert "critical_level" in pressure
                
                # Verify pressure logic consistency
                if memory_threshold > 75.0:  # System at 75%
                    assert pressure["system_pressure"] is False
                else:
                    assert pressure["system_pressure"] is True
                
                # GPU pressure should match threshold
                assert pressure["gpu_pressure"] == (memory_threshold <= memory_threshold)
                
                # Critical level should be OR of both
                expected_critical = pressure["system_pressure"] or pressure["gpu_pressure"]
                assert pressure["critical_level"] == expected_critical


class TestDriveIntegrationConsistency:
    """Test Google Drive integration consistency."""
    
    @given(
        project_paths=st.lists(
            st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_/')),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=20, deadline=3000)
    def test_project_directory_setup_consistency(self, project_paths: List[str]):
        """Test that project directory setup is consistent."""
        # Filter valid paths
        valid_paths = [path for path in project_paths if path and not path.startswith('/')]
        assume(len(valid_paths) > 0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            drive_manager = ColabDriveManager(temp_dir)
            drive_manager.mounted = True  # Mock mounted state
            
            for project_path in valid_paths:
                # Setup project directory
                full_path = drive_manager.setup_project_directory(project_path)
                
                # Verify directory structure exists
                assert os.path.exists(full_path)
                assert os.path.exists(os.path.join(full_path, "checkpoints"))
                assert os.path.exists(os.path.join(full_path, "logs"))
                assert os.path.exists(os.path.join(full_path, "models"))
                assert os.path.exists(os.path.join(full_path, "configs"))
                assert os.path.exists(os.path.join(full_path, "results"))
                
                # Verify path consistency
                expected_path = os.path.join(temp_dir, "MyDrive", project_path.lstrip("/"))
                assert full_path == expected_path


class TestSessionStateConsistency:
    """Test session state persistence consistency."""
    
    @given(
        state_data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.text(min_size=1, max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans()
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=20, deadline=3000)
    def test_session_state_round_trip_consistency(self, state_data: Dict[str, Any]):
        """Test that session state save/load is consistent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup managers
            session_manager = ColabSessionManager()
            drive_manager = ColabDriveManager(temp_dir)
            drive_manager.mounted = True
            
            # Create MyDrive directory structure
            mydrive_path = os.path.join(temp_dir, "MyDrive")
            os.makedirs(mydrive_path, exist_ok=True)
            
            # Save session state
            saved_path = session_manager.save_session_state(state_data, drive_manager)
            assert os.path.exists(saved_path)
            
            # Load session state
            loaded_state = session_manager.load_session_state(drive_manager)
            
            # Verify round-trip consistency
            assert loaded_state is not None
            assert "state_data" in loaded_state
            assert loaded_state["state_data"] == state_data
            
            # Verify metadata exists
            assert "session_info" in loaded_state
            assert "timestamp" in loaded_state
            
            # Verify timestamp is valid ISO format
            timestamp_str = loaded_state["timestamp"]
            datetime.fromisoformat(timestamp_str)  # Should not raise exception


class TestErrorHandlingConsistency:
    """Test error handling consistency across different error types."""
    
    @given(
        error_messages=st.lists(
            st.tuples(
                st.sampled_from([
                    "cuda out of memory",
                    "session timeout",
                    "drive mount failed", 
                    "dataset not found",
                    "model loading error",
                    "loss is nan"
                ]),
                st.text(min_size=10, max_size=100)
            ),
            min_size=1,
            max_size=6
        )
    )
    @settings(max_examples=20, deadline=3000)
    def test_error_classification_consistency(self, error_messages: List[tuple]):
        """Test that error classification is consistent."""
        for error_key, error_context in error_messages:
            # Create mock exception
            mock_error = Exception(error_key)
            
            # Handle error
            error_info = handle_training_error(mock_error, error_context)
            
            # Verify error info structure
            assert "error_type" in error_info
            assert "original_error" in error_info
            assert "context" in error_info
            assert "guidance" in error_info
            
            # Verify error classification consistency
            error_type = error_info["error_type"]
            
            if "cuda" in error_key or "memory" in error_key:
                assert error_type == "gpu_memory_error"
            elif "timeout" in error_key or "session" in error_key:
                assert error_type == "session_timeout"
            elif "drive" in error_key or "mount" in error_key:
                assert error_type == "drive_mount_error"
            elif "dataset" in error_key or "not found" in error_key:
                assert error_type == "dataset_loading_error"
            elif "model" in error_key and "loading" in error_key:
                assert error_type == "model_loading_error"
            elif "loss" in error_key or "nan" in error_key:
                assert error_type == "training_divergence"
            
            # Verify original error is preserved
            assert error_info["original_error"] == str(mock_error)
            assert error_info["context"] == error_context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])