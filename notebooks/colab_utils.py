"""
Google Colab Utilities for RLHF Phi-3 Pipeline

This module provides Colab-specific utilities for session management, 
Google Drive integration, memory optimization, and progress tracking.
Designed to work within Google Colab's T4 GPU constraints and 12-hour session limits.
"""

import os
import time
import psutil
import torch
import gc
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import subprocess
import sys
from contextlib import contextmanager

try:
    from google.colab import drive, auth
    from google.colab import output
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

try:
    import GPUtil
except ImportError:
    GPUtil = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColabSessionManager:
    """
    Manages Google Colab session lifecycle, including timeout handling,
    memory monitoring, and session state persistence.
    """
    
    def __init__(self, session_timeout_hours: float = 12.0):
        self.session_timeout_hours = session_timeout_hours
        self.session_start_time = datetime.now()
        self.drive_mounted = False
        self.drive_mount_path = "/content/drive"
        self.session_state_file = None
        
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information including runtime and memory usage."""
        current_time = datetime.now()
        elapsed_time = current_time - self.session_start_time
        remaining_time = timedelta(hours=self.session_timeout_hours) - elapsed_time
        
        # Get memory info
        memory_info = self.get_memory_usage()
        
        return {
            "session_start": self.session_start_time.isoformat(),
            "current_time": current_time.isoformat(),
            "elapsed_hours": elapsed_time.total_seconds() / 3600,
            "remaining_hours": max(0, remaining_time.total_seconds() / 3600),
            "memory_usage": memory_info,
            "drive_mounted": self.drive_mounted,
            "colab_available": COLAB_AVAILABLE
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        memory_info = {
            "system_memory": {},
            "gpu_memory": {}
        }
        
        # System memory
        vm = psutil.virtual_memory()
        memory_info["system_memory"] = {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent_used": vm.percent
        }
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = gpu_memory.total_memory / (1024**3)
                
                memory_info["gpu_memory"][f"gpu_{i}"] = {
                    "name": gpu_memory.name,
                    "total_gb": total,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "free_gb": total - reserved,
                    "percent_used": (reserved / total) * 100 if total > 0 else 0
                }
        
        return memory_info
    
    def check_memory_pressure(self, threshold_percent: float = 90.0) -> Dict[str, bool]:
        """Check if memory usage is approaching critical levels."""
        memory_info = self.get_memory_usage()
        
        pressure = {
            "system_pressure": False,
            "gpu_pressure": False,
            "critical_level": False
        }
        
        # Check system memory
        if memory_info["system_memory"]["percent_used"] > threshold_percent:
            pressure["system_pressure"] = True
            
        # Check GPU memory
        for gpu_id, gpu_info in memory_info["gpu_memory"].items():
            if gpu_info["percent_used"] > threshold_percent:
                pressure["gpu_pressure"] = True
                
        pressure["critical_level"] = pressure["system_pressure"] or pressure["gpu_pressure"]
        
        return pressure
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization operations."""
        logger.info("Starting memory optimization...")
        
        initial_memory = self.get_memory_usage()
        
        # Clear Python garbage collection
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection again
        gc.collect()
        
        final_memory = self.get_memory_usage()
        
        optimization_result = {
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "memory_freed": {}
        }
        
        # Calculate memory freed
        if "gpu_0" in initial_memory["gpu_memory"] and "gpu_0" in final_memory["gpu_memory"]:
            initial_gpu = initial_memory["gpu_memory"]["gpu_0"]["reserved_gb"]
            final_gpu = final_memory["gpu_memory"]["gpu_0"]["reserved_gb"]
            optimization_result["memory_freed"]["gpu_gb"] = initial_gpu - final_gpu
        
        initial_sys = initial_memory["system_memory"]["used_gb"]
        final_sys = final_memory["system_memory"]["used_gb"]
        optimization_result["memory_freed"]["system_gb"] = initial_sys - final_sys
        
        logger.info(f"Memory optimization completed. GPU freed: {optimization_result['memory_freed'].get('gpu_gb', 0):.2f}GB, "
                   f"System freed: {optimization_result['memory_freed'].get('system_gb', 0):.2f}GB")
        
        return optimization_result
    
    def estimate_time_remaining(self) -> Dict[str, float]:
        """Estimate remaining session time and provide recommendations."""
        session_info = self.get_session_info()
        remaining_hours = session_info["remaining_hours"]
        
        recommendations = {
            "remaining_hours": remaining_hours,
            "should_checkpoint": remaining_hours < 1.0,
            "should_prepare_resume": remaining_hours < 0.5,
            "emergency_save": remaining_hours < 0.1
        }
        
        return recommendations
    
    def save_session_state(self, state_data: Dict[str, Any], drive_manager: 'ColabDriveManager') -> str:
        """Save current session state for resumption after timeout."""
        if not drive_manager.mounted:
            logger.warning("Drive not mounted. Saving session state locally only.")
            
        session_state = {
            "session_info": self.get_session_info(),
            "timestamp": datetime.now().isoformat(),
            "state_data": state_data
        }
        
        # Save locally first
        local_path = "/content/session_state.json"
        with open(local_path, 'w') as f:
            json.dump(session_state, f, indent=2)
            
        # Try to save to Drive
        if drive_manager.mounted:
            drive_path = os.path.join(drive_manager.mount_path, "MyDrive", "rlhf-phi3", "session_state.json")
            try:
                drive_manager.sync_file(local_path, drive_path, "to_drive")
                logger.info(f"Session state saved to Drive: {drive_path}")
                return drive_path
            except Exception as e:
                logger.error(f"Failed to save session state to Drive: {str(e)}")
                
        logger.info(f"Session state saved locally: {local_path}")
        return local_path
    
    def load_session_state(self, drive_manager: 'ColabDriveManager') -> Optional[Dict[str, Any]]:
        """Load previous session state for resumption."""
        # Try Drive first
        if drive_manager.mounted:
            drive_path = os.path.join(drive_manager.mount_path, "MyDrive", "rlhf-phi3", "session_state.json")
            if os.path.exists(drive_path):
                try:
                    with open(drive_path, 'r') as f:
                        session_state = json.load(f)
                    logger.info("Session state loaded from Drive")
                    return session_state
                except Exception as e:
                    logger.error(f"Failed to load session state from Drive: {str(e)}")
        
        # Try local fallback
        local_path = "/content/session_state.json"
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r') as f:
                    session_state = json.load(f)
                logger.info("Session state loaded from local storage")
                return session_state
            except Exception as e:
                logger.error(f"Failed to load session state locally: {str(e)}")
                
        logger.info("No previous session state found")
        return None
    
    def setup_timeout_monitoring(self, progress_tracker: 'ColabProgressTracker', 
                                check_interval_minutes: int = 15):
        """Setup automatic session timeout monitoring."""
        import threading
        import time
        
        def monitor_timeout():
            while True:
                time.sleep(check_interval_minutes * 60)  # Convert to seconds
                
                recommendations = self.estimate_time_remaining()
                
                if recommendations["emergency_save"]:
                    progress_tracker.display_session_timeout_warning(recommendations["remaining_hours"])
                    logger.critical("EMERGENCY: Session timeout imminent! Save checkpoints immediately!")
                    
                elif recommendations["should_prepare_resume"]:
                    progress_tracker.display_session_timeout_warning(recommendations["remaining_hours"])
                    logger.warning("Session timeout approaching. Prepare for resumption.")
                    
                elif recommendations["should_checkpoint"]:
                    logger.info("Session timeout in <1 hour. Consider saving checkpoints.")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_timeout, daemon=True)
        monitor_thread.start()
        logger.info(f"Session timeout monitoring started (check every {check_interval_minutes} minutes)")


class ColabDriveManager:
    """
    Manages Google Drive mounting, authentication, and file operations
    for checkpoint persistence across Colab sessions.
    """
    
    def __init__(self, mount_path: str = "/content/drive"):
        self.mount_path = mount_path
        self.mounted = False
        self.authenticated = False
        
    def mount_drive(self, force_remount: bool = False) -> bool:
        """Mount Google Drive with authentication."""
        if not COLAB_AVAILABLE:
            logger.warning("Google Colab not available. Drive mounting skipped.")
            return False
            
        try:
            if force_remount or not self.is_mounted():
                logger.info("Mounting Google Drive...")
                drive.mount(self.mount_path, force_remount=force_remount)
                
            self.mounted = self.is_mounted()
            if self.mounted:
                logger.info(f"Google Drive successfully mounted at {self.mount_path}")
            else:
                logger.error("Failed to mount Google Drive")
                
            return self.mounted
            
        except Exception as e:
            logger.error(f"Error mounting Google Drive: {str(e)}")
            return False
    
    def is_mounted(self) -> bool:
        """Check if Google Drive is currently mounted."""
        return os.path.exists(os.path.join(self.mount_path, "MyDrive"))
    
    def authenticate(self) -> bool:
        """Authenticate with Google services."""
        if not COLAB_AVAILABLE:
            logger.warning("Google Colab not available. Authentication skipped.")
            return False
            
        try:
            auth.authenticate_user()
            self.authenticated = True
            logger.info("Google authentication successful")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def setup_project_directory(self, project_path: str) -> str:
        """Set up project directory structure in Google Drive."""
        if not self.mounted:
            if not self.mount_drive():
                raise RuntimeError("Cannot setup project directory: Drive not mounted")
        
        full_path = os.path.join(self.mount_path, "MyDrive", project_path.lstrip("/"))
        
        # Create directory structure
        directories = [
            full_path,
            os.path.join(full_path, "checkpoints"),
            os.path.join(full_path, "logs"),
            os.path.join(full_path, "models"),
            os.path.join(full_path, "configs"),
            os.path.join(full_path, "results")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info(f"Project directory structure created at {full_path}")
        return full_path
    
    def sync_file(self, local_path: str, drive_path: str, direction: str = "to_drive") -> bool:
        """Sync files between local storage and Google Drive."""
        if not self.mounted:
            logger.error("Cannot sync: Google Drive not mounted")
            return False
            
        try:
            if direction == "to_drive":
                if os.path.exists(local_path):
                    os.makedirs(os.path.dirname(drive_path), exist_ok=True)
                    subprocess.run(["cp", "-r", local_path, drive_path], check=True)
                    logger.info(f"Synced {local_path} to {drive_path}")
                    return True
                else:
                    logger.error(f"Local file not found: {local_path}")
                    return False
                    
            elif direction == "from_drive":
                if os.path.exists(drive_path):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    subprocess.run(["cp", "-r", drive_path, local_path], check=True)
                    logger.info(f"Synced {drive_path} to {local_path}")
                    return True
                else:
                    logger.error(f"Drive file not found: {drive_path}")
                    return False
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"Sync failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected sync error: {str(e)}")
            return False


class ColabProgressTracker:
    """
    Provides progress tracking and user interface elements optimized for Colab notebooks.
    Includes progress bars, status updates, and user-friendly error messages.
    """
    
    def __init__(self):
        self.active_operations = {}
        self.error_messages = {}
        self.troubleshooting_guides = {}
        
    @contextmanager
    def track_operation(self, operation_name: str, total_steps: Optional[int] = None):
        """Context manager for tracking long-running operations."""
        operation_id = f"{operation_name}_{int(time.time())}"
        
        try:
            self.start_operation(operation_id, operation_name, total_steps)
            yield operation_id
        finally:
            self.complete_operation(operation_id)
    
    def start_operation(self, operation_id: str, name: str, total_steps: Optional[int] = None):
        """Start tracking a new operation."""
        self.active_operations[operation_id] = {
            "name": name,
            "start_time": time.time(),
            "total_steps": total_steps,
            "current_step": 0,
            "status": "running"
        }
        
        logger.info(f"Started operation: {name}")
        
    def update_progress(self, operation_id: str, current_step: int, message: str = ""):
        """Update progress for an active operation."""
        if operation_id not in self.active_operations:
            return
            
        operation = self.active_operations[operation_id]
        operation["current_step"] = current_step
        
        elapsed_time = time.time() - operation["start_time"]
        
        if operation["total_steps"]:
            progress_percent = (current_step / operation["total_steps"]) * 100
            
            # Estimate remaining time
            if current_step > 0:
                time_per_step = elapsed_time / current_step
                remaining_steps = operation["total_steps"] - current_step
                eta_seconds = remaining_steps * time_per_step
                eta_str = f"ETA: {eta_seconds/60:.1f}min"
            else:
                eta_str = "ETA: calculating..."
                
            status_msg = f"{operation['name']}: {progress_percent:.1f}% ({current_step}/{operation['total_steps']}) - {eta_str}"
        else:
            status_msg = f"{operation['name']}: Step {current_step} - Elapsed: {elapsed_time/60:.1f}min"
            
        if message:
            status_msg += f" - {message}"
            
        logger.info(status_msg)
        
        # Update Colab output if available
        if COLAB_AVAILABLE:
            try:
                output.clear()
                print(status_msg)
            except:
                pass  # Fallback to regular logging
    
    def complete_operation(self, operation_id: str):
        """Mark an operation as completed."""
        if operation_id not in self.active_operations:
            return
            
        operation = self.active_operations[operation_id]
        elapsed_time = time.time() - operation["start_time"]
        operation["status"] = "completed"
        
        logger.info(f"Completed operation: {operation['name']} in {elapsed_time/60:.1f} minutes")
        
        # Clean up completed operation
        del self.active_operations[operation_id]
    
    def get_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active operations."""
        return self.active_operations.copy()
    
    def display_progress_bar(self, operation_id: str, width: int = 50) -> str:
        """Generate ASCII progress bar for display."""
        if operation_id not in self.active_operations:
            return ""
            
        operation = self.active_operations[operation_id]
        
        if not operation["total_steps"]:
            return f"[{operation['name']}] Running... (Step {operation['current_step']})"
            
        progress = operation["current_step"] / operation["total_steps"]
        filled_width = int(width * progress)
        bar = "█" * filled_width + "░" * (width - filled_width)
        
        percentage = progress * 100
        return f"[{operation['name']}] {bar} {percentage:.1f}% ({operation['current_step']}/{operation['total_steps']})"
    
    def add_error_message(self, error_type: str, message: str, troubleshooting_steps: List[str]):
        """Add user-friendly error message with troubleshooting guide."""
        self.error_messages[error_type] = {
            "message": message,
            "troubleshooting": troubleshooting_steps,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_error_guidance(self, error_type: str) -> Optional[Dict[str, Any]]:
        """Get user-friendly error guidance."""
        return self.error_messages.get(error_type)
    
    def setup_common_error_guides(self):
        """Setup common error messages and troubleshooting guides."""
        
        # GPU Memory Error
        self.add_error_message(
            "gpu_memory_error",
            "GPU memory exhausted during training. This is common with large models on T4 GPUs.",
            [
                "1. Reduce batch size in your configuration (try batch_size=1)",
                "2. Increase gradient_accumulation_steps to maintain effective batch size",
                "3. Enable gradient checkpointing (should be enabled by default)",
                "4. Use mixed precision training (fp16=True)",
                "5. Reduce model max_length if possible",
                "6. Clear GPU cache: torch.cuda.empty_cache()"
            ]
        )
        
        # Session Timeout
        self.add_error_message(
            "session_timeout",
            "Colab session is approaching the 12-hour limit. Training may be interrupted.",
            [
                "1. Save checkpoint immediately using emergency_checkpoint_save()",
                "2. Note the current step/epoch for resumption",
                "3. Download important files to local machine as backup",
                "4. Prepare to restart session and resume from checkpoint",
                "5. Consider breaking training into smaller chunks"
            ]
        )
        
        # Google Drive Issues
        self.add_error_message(
            "drive_mount_error", 
            "Failed to mount Google Drive. Checkpoints cannot be saved persistently.",
            [
                "1. Restart runtime and try mounting again",
                "2. Check Google account permissions",
                "3. Clear browser cache and cookies for colab.research.google.com",
                "4. Try mounting manually: drive.mount('/content/drive', force_remount=True)",
                "5. As fallback, save checkpoints locally and download manually"
            ]
        )
        
        # Dataset Loading Issues
        self.add_error_message(
            "dataset_loading_error",
            "Failed to load training dataset. This may be due to network issues or dataset availability.",
            [
                "1. Check internet connection",
                "2. Try loading a smaller subset of the dataset",
                "3. Clear HuggingFace cache: rm -rf ~/.cache/huggingface/",
                "4. Use alternative dataset or local files",
                "5. Check dataset name and split parameters"
            ]
        )
        
        # Model Loading Issues  
        self.add_error_message(
            "model_loading_error",
            "Failed to load the Phi-3 model. This may be due to network or authentication issues.",
            [
                "1. Check HuggingFace Hub status",
                "2. Verify model name: 'microsoft/Phi-3-mini-4k-instruct'",
                "3. Check HuggingFace authentication token if required",
                "4. Clear model cache and retry",
                "5. Try loading with use_auth_token=False"
            ]
        )
        
        # Training Divergence
        self.add_error_message(
            "training_divergence",
            "Training loss is diverging or not decreasing. Model may not be learning properly.",
            [
                "1. Reduce learning rate (try 1e-5 or lower)",
                "2. Check gradient clipping (max_grad_norm=0.5)",
                "3. Verify dataset preprocessing is correct",
                "4. Increase warmup steps",
                "5. Check for NaN values in loss",
                "6. Try different optimizer (AdamW with lower weight_decay)"
            ]
        )
    
    def display_session_timeout_warning(self, remaining_hours: float):
        """Display session timeout warning with resume instructions."""
        if COLAB_AVAILABLE:
            try:
                from IPython.display import display, HTML
                
                if remaining_hours < 0.5:
                    color = "red"
                    urgency = "CRITICAL"
                elif remaining_hours < 1.0:
                    color = "orange" 
                    urgency = "WARNING"
                else:
                    return  # No warning needed
                
                warning_html = f"""
                <div style="border: 2px solid {color}; padding: 15px; margin: 10px; border-radius: 5px; background-color: #f9f9f9;">
                    <h3 style="color: {color}; margin-top: 0;">⚠️ SESSION TIMEOUT {urgency}</h3>
                    <p><strong>Time remaining: {remaining_hours:.1f} hours</strong></p>
                    <p>Your Colab session will timeout soon. To avoid losing progress:</p>
                    <ol>
                        <li>Save checkpoint immediately</li>
                        <li>Download important files</li>
                        <li>Note current training step for resumption</li>
                        <li>Prepare to restart and resume training</li>
                    </ol>
                    <p><em>Use <code>emergency_checkpoint_save()</code> for quick saves.</em></p>
                </div>
                """
                
                display(HTML(warning_html))
                
            except ImportError:
                # Fallback to text warning
                logger.warning(f"SESSION TIMEOUT {urgency}: {remaining_hours:.1f} hours remaining!")
        else:
            logger.warning(f"Session timeout warning: {remaining_hours:.1f} hours remaining")
    
    def display_memory_warning(self, memory_info: Dict[str, Any]):
        """Display memory usage warning with optimization suggestions."""
        if not memory_info.get("critical_level", False):
            return
            
        if COLAB_AVAILABLE:
            try:
                from IPython.display import display, HTML
                
                # Get GPU memory info for display
                gpu_info = memory_info.get("gpu_memory", {}).get("gpu_0", {})
                gpu_usage = gpu_info.get("percent_used", 0)
                
                warning_html = f"""
                <div style="border: 2px solid red; padding: 15px; margin: 10px; border-radius: 5px; background-color: #fff5f5;">
                    <h3 style="color: red; margin-top: 0;">🚨 HIGH MEMORY USAGE</h3>
                    <p><strong>GPU Memory: {gpu_usage:.1f}% used</strong></p>
                    <p>Memory usage is critical. Consider these optimizations:</p>
                    <ul>
                        <li>Reduce batch size</li>
                        <li>Increase gradient accumulation steps</li>
                        <li>Clear GPU cache: <code>torch.cuda.empty_cache()</code></li>
                        <li>Enable gradient checkpointing</li>
                        <li>Use mixed precision (fp16)</li>
                    </ul>
                    <p><em>Use <code>session_manager.optimize_memory()</code> for automatic cleanup.</em></p>
                </div>
                """
                
                display(HTML(warning_html))
                
            except ImportError:
                logger.warning(f"HIGH MEMORY USAGE: GPU {gpu_usage:.1f}% used")
        else:
            logger.warning("Critical memory usage detected")
    
    def create_resume_instructions(self, checkpoint_path: str, stage: str, step: int) -> str:
        """Generate detailed resume instructions for session restart."""
        instructions = f"""
# Session Resume Instructions

Your training was interrupted. To resume:

## 1. Setup Environment
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install transformers peft trl datasets accelerate wandb

# Import required modules
from rlhf_phi3.training.training_orchestrator import TrainingOrchestrator
from rlhf_phi3.config.config_manager import Config
```

## 2. Load Configuration and Resume
```python
# Load your configuration
config = Config.load_from_file('path/to/your/config.yaml')

# Initialize orchestrator
orchestrator = TrainingOrchestrator(config)

# Resume from checkpoint
orchestrator.resume_from_stage(
    stage="{stage}",
    checkpoint_path="{checkpoint_path}"
)
```

## 3. Training Details
- **Stage**: {stage}
- **Last Step**: {step}
- **Checkpoint**: {checkpoint_path}
- **Resume Time**: {datetime.now().isoformat()}

## 4. Verification
After resuming, verify:
- Model loads correctly
- Training metrics continue from last step
- Optimizer state is restored
- Learning rate schedule is correct

Save these instructions to Google Drive for easy access!
        """
        
        return instructions


class ColabEnvironmentSetup:
    """
    Handles Colab environment setup, dependency installation, and configuration.
    """
    
    def __init__(self):
        self.setup_complete = False
        self.installed_packages = set()
        
    def install_dependencies(self, requirements_file: Optional[str] = None, 
                           packages: Optional[List[str]] = None) -> bool:
        """Install required dependencies for the RLHF pipeline."""
        try:
            if requirements_file and os.path.exists(requirements_file):
                logger.info(f"Installing dependencies from {requirements_file}")
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                             check=True, capture_output=True, text=True)
                
            if packages:
                for package in packages:
                    if package not in self.installed_packages:
                        logger.info(f"Installing {package}")
                        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                     check=True, capture_output=True, text=True)
                        self.installed_packages.add(package)
                        
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during installation: {str(e)}")
            return False
    
    def setup_colab_environment(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Complete Colab environment setup."""
        setup_result = {
            "success": False,
            "gpu_available": False,
            "drive_mounted": False,
            "dependencies_installed": False,
            "config_loaded": False
        }
        
        try:
            # Check GPU availability
            if torch.cuda.is_available():
                setup_result["gpu_available"] = True
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU available: {gpu_name}")
            else:
                logger.warning("No GPU available. Training will be very slow.")
            
            # Setup Google Drive
            drive_manager = ColabDriveManager()
            if drive_manager.mount_drive():
                setup_result["drive_mounted"] = True
                
            # Install dependencies
            essential_packages = [
                "transformers>=4.36.0",
                "torch>=2.0.0", 
                "peft>=0.7.0",
                "trl>=0.7.0",
                "datasets>=2.14.0",
                "accelerate>=0.24.0",
                "wandb>=0.16.0",
                "evaluate>=0.4.0"
            ]
            
            if self.install_dependencies(packages=essential_packages):
                setup_result["dependencies_installed"] = True
                
            # Load configuration if provided
            if config_path and os.path.exists(config_path):
                setup_result["config_loaded"] = True
                
            setup_result["success"] = all([
                setup_result["gpu_available"] or True,  # GPU not strictly required
                setup_result["drive_mounted"],
                setup_result["dependencies_installed"]
            ])
            
            self.setup_complete = setup_result["success"]
            
        except Exception as e:
            logger.error(f"Environment setup failed: {str(e)}")
            
        return setup_result
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information for debugging."""
        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__ if torch else "Not installed",
            "cuda_available": torch.cuda.is_available() if torch else False,
            "colab_available": COLAB_AVAILABLE,
            "system_memory_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        if torch and torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["cuda_version"] = torch.version.cuda
            
        return info


# Convenience functions for easy notebook usage
def setup_colab_session(config_path: Optional[str] = None, 
                       enable_timeout_monitoring: bool = True) -> Tuple[ColabSessionManager, ColabDriveManager, ColabProgressTracker]:
    """
    One-stop setup function for Colab sessions.
    Returns configured session manager, drive manager, and progress tracker.
    """
    logger.info("Setting up Colab session...")
    
    # Initialize managers
    session_manager = ColabSessionManager()
    drive_manager = ColabDriveManager()
    progress_tracker = ColabProgressTracker()
    
    # Setup common error guides
    progress_tracker.setup_common_error_guides()
    
    # Setup environment
    env_setup = ColabEnvironmentSetup()
    setup_result = env_setup.setup_colab_environment(config_path)
    
    if setup_result["success"]:
        logger.info("Colab session setup completed successfully!")
    else:
        logger.warning("Colab session setup completed with issues. Check logs for details.")
    
    # Display system info
    system_info = env_setup.get_system_info()
    logger.info(f"System Info: {system_info}")
    
    # Setup timeout monitoring if requested
    if enable_timeout_monitoring:
        session_manager.setup_timeout_monitoring(progress_tracker)
    
    return session_manager, drive_manager, progress_tracker


def monitor_session_health(display_warnings: bool = True) -> Dict[str, Any]:
    """Quick health check for the current Colab session."""
    session_manager = ColabSessionManager()
    progress_tracker = ColabProgressTracker()
    
    health_info = {
        "session_info": session_manager.get_session_info(),
        "memory_pressure": session_manager.check_memory_pressure(),
        "time_recommendations": session_manager.estimate_time_remaining()
    }
    
    # Display warnings if requested
    if display_warnings:
        if health_info["memory_pressure"]["critical_level"]:
            logger.warning("Critical memory pressure detected! Consider optimizing memory usage.")
            progress_tracker.display_memory_warning(health_info["memory_pressure"])
            
        if health_info["time_recommendations"]["should_checkpoint"]:
            logger.warning("Session time running low. Consider saving checkpoints soon.")
            progress_tracker.display_session_timeout_warning(
                health_info["time_recommendations"]["remaining_hours"]
            )
        
    return health_info


def handle_training_error(error: Exception, error_context: str = "") -> Dict[str, Any]:
    """
    Handle training errors with user-friendly messages and troubleshooting guidance.
    """
    progress_tracker = ColabProgressTracker()
    progress_tracker.setup_common_error_guides()
    
    error_str = str(error).lower()
    error_info = {
        "error_type": "unknown_error",
        "original_error": str(error),
        "context": error_context,
        "guidance": None
    }
    
    # Classify error type
    if "cuda out of memory" in error_str or "memory" in error_str:
        error_info["error_type"] = "gpu_memory_error"
    elif "timeout" in error_str or "session" in error_str:
        error_info["error_type"] = "session_timeout"
    elif "drive" in error_str or "mount" in error_str:
        error_info["error_type"] = "drive_mount_error"
    elif "dataset" in error_str or "load" in error_str:
        error_info["error_type"] = "dataset_loading_error"
    elif "model" in error_str:
        error_info["error_type"] = "model_loading_error"
    elif "loss" in error_str or "nan" in error_str or "diverge" in error_str:
        error_info["error_type"] = "training_divergence"
    
    # Get guidance
    error_info["guidance"] = progress_tracker.get_error_guidance(error_info["error_type"])
    
    # Log user-friendly error message
    if error_info["guidance"]:
        logger.error(f"Training Error: {error_info['guidance']['message']}")
        logger.info("Troubleshooting steps:")
        for i, step in enumerate(error_info["guidance"]["troubleshooting"], 1):
            logger.info(f"  {step}")
    else:
        logger.error(f"Unknown training error: {str(error)}")
        
    return error_info


def create_session_resume_guide(checkpoint_path: str, stage: str, step: int, 
                               save_to_drive: bool = True) -> str:
    """Create and save session resume instructions."""
    progress_tracker = ColabProgressTracker()
    instructions = progress_tracker.create_resume_instructions(checkpoint_path, stage, step)
    
    # Save instructions locally
    local_path = "/content/resume_instructions.md"
    with open(local_path, 'w') as f:
        f.write(instructions)
    
    # Save to Drive if requested
    if save_to_drive and COLAB_AVAILABLE:
        try:
            drive_manager = ColabDriveManager()
            if drive_manager.is_mounted():
                drive_path = os.path.join(drive_manager.mount_path, "MyDrive", "rlhf-phi3", "resume_instructions.md")
                drive_manager.sync_file(local_path, drive_path, "to_drive")
                logger.info(f"Resume instructions saved to Drive: {drive_path}")
            else:
                logger.warning("Drive not mounted. Instructions saved locally only.")
        except Exception as e:
            logger.error(f"Failed to save instructions to Drive: {str(e)}")
    
    logger.info(f"Resume instructions saved: {local_path}")
    return instructions


def emergency_checkpoint_save(checkpoint_manager, model, optimizer, step: int, stage: str) -> str:
    """Emergency checkpoint save for session timeout scenarios."""
    logger.warning("Performing emergency checkpoint save...")
    
    try:
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer, 
            epoch=0,  # Emergency save doesn't track epochs
            step=step,
            stage=stage,
            emergency=True
        )
        
        logger.info(f"Emergency checkpoint saved to: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Emergency checkpoint save failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    print("Colab Utils - Example Usage")
    
    # Setup session
    session_mgr, drive_mgr, progress_tracker = setup_colab_session()
    
    # Monitor health
    health = monitor_session_health()
    print(f"Session Health: {health}")
    
    # Example progress tracking
    with progress_tracker.track_operation("Example Training", 100) as op_id:
        for i in range(100):
            time.sleep(0.01)  # Simulate work
            progress_tracker.update_progress(op_id, i+1, f"Processing batch {i+1}")