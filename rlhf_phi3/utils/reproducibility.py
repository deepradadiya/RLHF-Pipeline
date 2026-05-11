"""
Reproducibility Utilities for RLHF Phi-3 Pipeline

This module provides utilities for ensuring reproducible training results through
fixed random seed management and comprehensive environment logging. It supports
deterministic training when specified and logs exact library versions and 
environment details for full reproducibility.

Requirements satisfied:
- 15.2: Fixed random seeds for deterministic training
- 15.3: Environment and library version logging
"""

import os
import sys
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import hashlib

# Core libraries for seed management
import random
import numpy as np

# ML libraries (imported conditionally to handle missing dependencies)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """
    Manager for reproducibility features including seed management and environment logging.
    
    This class provides utilities to ensure deterministic training results and comprehensive
    environment tracking for full reproducibility of training runs.
    
    Requirements satisfied:
    - 15.2: Fixed random seeds for deterministic training
    - 15.3: Environment and library version logging
    """
    
    def __init__(self, seed: Optional[int] = None, enable_deterministic: bool = True):
        """
        Initialize the reproducibility manager.
        
        Args:
            seed: Random seed for deterministic training. If None, uses current timestamp
            enable_deterministic: Whether to enable deterministic training mode
        """
        self.seed = seed if seed is not None else int(datetime.now().timestamp())
        self.enable_deterministic = enable_deterministic
        self.environment_info: Optional[Dict[str, Any]] = None
        
        logger.info(f"ReproducibilityManager initialized with seed: {self.seed}")
    
    def set_random_seeds(self, seed: Optional[int] = None) -> None:
        """
        Set random seeds for all relevant libraries to ensure deterministic training.
        
        Args:
            seed: Random seed to use. If None, uses the manager's seed
            
        Requirement 15.2: Fixed random seeds for deterministic training
        """
        if seed is not None:
            self.seed = seed
        
        logger.info(f"Setting random seeds to {self.seed}")
        
        # Set Python random seed
        random.seed(self.seed)
        
        # Set NumPy random seed
        np.random.seed(self.seed)
        
        # Set PyTorch seeds if available
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            
            if self.enable_deterministic:
                # Enable deterministic algorithms
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # Set deterministic algorithms (PyTorch 1.12+)
                if hasattr(torch, 'use_deterministic_algorithms'):
                    try:
                        torch.use_deterministic_algorithms(True)
                    except RuntimeError as e:
                        logger.warning(f"Could not enable deterministic algorithms: {e}")
                
                logger.info("PyTorch deterministic mode enabled")
        
        # Set transformers seed if available
        if TRANSFORMERS_AVAILABLE:
            transformers.set_seed(self.seed)
            logger.info("Transformers seed set")
        
        # Set environment variable for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        logger.info(f"All random seeds set to {self.seed}")
    
    def setup_deterministic_training(self) -> None:
        """
        Configure the environment for deterministic training.
        
        This method sets up all necessary configurations to ensure reproducible
        training results across different runs.
        
        Requirement 15.2: Fixed random seeds for deterministic training
        """
        logger.info("Setting up deterministic training environment")
        
        # Set random seeds
        self.set_random_seeds()
        
        # Configure additional environment variables for reproducibility
        env_vars = {
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',  # For deterministic CUDA operations
            'CUDA_LAUNCH_BLOCKING': '1',  # Synchronous CUDA operations
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            logger.debug(f"Set environment variable {var}={value}")
        
        # Configure PyTorch specific settings
        if TORCH_AVAILABLE:
            # Disable multithreading for deterministic results
            torch.set_num_threads(1)
            
            # Configure CUDA settings if available
            if torch.cuda.is_available():
                # Set CUDA device deterministic
                torch.cuda.set_device(0)  # Use first GPU
                logger.info("CUDA deterministic settings configured")
        
        logger.info("Deterministic training environment setup complete")
    
    def log_environment_info(self) -> Dict[str, Any]:
        """
        Log comprehensive environment and library version information.
        
        Returns:
            Dictionary containing complete environment information
            
        Requirement 15.3: Environment and library version logging
        """
        logger.info("Collecting environment information")
        
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'deterministic_mode': self.enable_deterministic,
            'system': self._get_system_info(),
            'python': self._get_python_info(),
            'libraries': self._get_library_versions(),
            'cuda': self._get_cuda_info(),
            'environment_variables': self._get_relevant_env_vars(),
            'git': self._get_git_info(),
            'hardware': self._get_hardware_info()
        }
        
        self.environment_info = env_info
        logger.info("Environment information collected")
        
        return env_info
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'hostname': platform.node()
        }
    
    def _get_python_info(self) -> Dict[str, Any]:
        """Get Python version and configuration information."""
        return {
            'version': sys.version,
            'version_info': {
                'major': sys.version_info.major,
                'minor': sys.version_info.minor,
                'micro': sys.version_info.micro,
                'releaselevel': sys.version_info.releaselevel,
                'serial': sys.version_info.serial
            },
            'executable': sys.executable,
            'path': sys.path[:5],  # First 5 paths to avoid too much detail
            'prefix': sys.prefix,
            'base_prefix': getattr(sys, 'base_prefix', sys.prefix)
        }
    
    def _get_library_versions(self) -> Dict[str, str]:
        """Get versions of all relevant ML libraries."""
        versions = {}
        
        # Core ML libraries
        libraries_to_check = [
            'torch', 'transformers', 'datasets', 'peft', 'trl', 'accelerate',
            'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
            'wandb', 'evaluate', 'bitsandbytes', 'huggingface_hub',
            'google-colab', 'google-auth', 'pytest', 'hypothesis'
        ]
        
        for lib_name in libraries_to_check:
            try:
                # Handle special cases
                if lib_name == 'google-colab':
                    import google.colab
                    versions[lib_name] = getattr(google.colab, '__version__', 'unknown')
                elif lib_name == 'google-auth':
                    import google.auth
                    versions[lib_name] = getattr(google.auth, '__version__', 'unknown')
                else:
                    # Standard import
                    module = __import__(lib_name.replace('-', '_'))
                    versions[lib_name] = getattr(module, '__version__', 'unknown')
            except ImportError:
                versions[lib_name] = 'not_installed'
            except Exception as e:
                versions[lib_name] = f'error: {str(e)}'
        
        return versions
    
    def _get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA and GPU information."""
        cuda_info = {
            'available': False,
            'version': None,
            'device_count': 0,
            'devices': []
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            cuda_info['available'] = True
            cuda_info['version'] = torch.version.cuda
            cuda_info['device_count'] = torch.cuda.device_count()
            
            # Get device information
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'index': i,
                    'name': device_props.name,
                    'total_memory': device_props.total_memory,
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'multi_processor_count': device_props.multi_processor_count
                }
                cuda_info['devices'].append(device_info)
        
        return cuda_info
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        relevant_vars = [
            'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG',
            'PYTHONHASHSEED', 'WANDB_PROJECT', 'WANDB_ENTITY', 'HF_HOME',
            'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE', 'TOKENIZERS_PARALLELISM'
        ]
        
        return {var: os.environ.get(var, 'not_set') for var in relevant_vars}
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information if available."""
        git_info = {
            'available': False,
            'commit_hash': None,
            'branch': None,
            'remote_url': None,
            'is_dirty': None
        }
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['available'] = True
                
                # Get commit hash
                result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    git_info['commit_hash'] = result.stdout.strip()
                
                # Get branch name
                result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    git_info['branch'] = result.stdout.strip()
                
                # Get remote URL
                result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    git_info['remote_url'] = result.stdout.strip()
                
                # Check if repository is dirty
                result = subprocess.run(['git', 'diff', '--quiet'], 
                                      capture_output=True, timeout=5)
                git_info['is_dirty'] = result.returncode != 0
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"Could not get git information: {e}")
        
        return git_info
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        hardware_info = {
            'cpu_count': os.cpu_count(),
            'memory': {}
        }
        
        # Get memory information (Linux/Unix)
        try:
            if platform.system() == 'Linux':
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            hardware_info['memory']['total'] = line.split()[1] + ' kB'
                        elif 'MemAvailable:' in line:
                            hardware_info['memory']['available'] = line.split()[1] + ' kB'
        except Exception as e:
            logger.debug(f"Could not get memory information: {e}")
        
        return hardware_info
    
    def save_environment_info(self, path: Union[str, Path], 
                            format: str = 'json') -> None:
        """
        Save environment information to file.
        
        Args:
            path: File path to save environment information
            format: File format ('json' or 'yaml')
            
        Requirement 15.3: Environment and library version logging
        """
        if self.environment_info is None:
            self.environment_info = self.log_environment_info()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(path, 'w') as f:
                json.dump(self.environment_info, f, indent=2, default=str)
        elif format.lower() == 'yaml':
            try:
                import yaml
                with open(path, 'w') as f:
                    yaml.dump(self.environment_info, f, default_flow_style=False, indent=2)
            except ImportError:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'yaml'")
        
        logger.info(f"Environment information saved to {path}")
    
    def create_reproducibility_hash(self) -> str:
        """
        Create a hash representing the current reproducibility state.
        
        This hash includes the seed, library versions, and key environment settings
        to uniquely identify the reproducibility context.
        
        Returns:
            SHA-256 hash string representing the reproducibility state
        """
        if self.environment_info is None:
            self.environment_info = self.log_environment_info()
        
        # Create a reproducibility fingerprint
        fingerprint_data = {
            'seed': self.seed,
            'deterministic_mode': self.enable_deterministic,
            'python_version': self.environment_info['python']['version'],
            'key_libraries': {
                lib: version for lib, version in self.environment_info['libraries'].items()
                if lib in ['torch', 'transformers', 'datasets', 'peft', 'trl', 'numpy']
            },
            'cuda_version': self.environment_info['cuda']['version'],
            'system': self.environment_info['system']['platform']
        }
        
        # Create hash
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        hash_obj = hashlib.sha256(fingerprint_str.encode())
        return hash_obj.hexdigest()
    
    def validate_reproducibility(self, reference_hash: str) -> bool:
        """
        Validate that the current environment matches a reference reproducibility hash.
        
        Args:
            reference_hash: Reference hash to compare against
            
        Returns:
            True if environments match, False otherwise
        """
        current_hash = self.create_reproducibility_hash()
        matches = current_hash == reference_hash
        
        if matches:
            logger.info("Reproducibility validation passed")
        else:
            logger.warning(f"Reproducibility validation failed. "
                         f"Current: {current_hash}, Reference: {reference_hash}")
        
        return matches
    
    def get_reproducibility_summary(self) -> Dict[str, Any]:
        """
        Get a summary of reproducibility settings and environment.
        
        Returns:
            Dictionary containing reproducibility summary
        """
        if self.environment_info is None:
            self.environment_info = self.log_environment_info()
        
        return {
            'seed': self.seed,
            'deterministic_mode': self.enable_deterministic,
            'reproducibility_hash': self.create_reproducibility_hash(),
            'key_versions': {
                'python': self.environment_info['python']['version_info'],
                'torch': self.environment_info['libraries'].get('torch', 'not_installed'),
                'transformers': self.environment_info['libraries'].get('transformers', 'not_installed'),
                'cuda': self.environment_info['cuda']['version']
            },
            'timestamp': self.environment_info['timestamp']
        }


# Convenience functions for common reproducibility tasks

def setup_reproducible_training(seed: Optional[int] = None, 
                               enable_deterministic: bool = True) -> ReproducibilityManager:
    """
    Set up reproducible training environment with a single function call.
    
    Args:
        seed: Random seed for deterministic training
        enable_deterministic: Whether to enable deterministic training mode
        
    Returns:
        ReproducibilityManager instance
        
    Requirement 15.2: Fixed random seeds for deterministic training
    """
    manager = ReproducibilityManager(seed=seed, enable_deterministic=enable_deterministic)
    manager.setup_deterministic_training()
    return manager


def log_training_environment(output_path: Union[str, Path], 
                           seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Log complete training environment information to file.
    
    Args:
        output_path: Path to save environment information
        seed: Random seed used for training
        
    Returns:
        Environment information dictionary
        
    Requirement 15.3: Environment and library version logging
    """
    manager = ReproducibilityManager(seed=seed)
    env_info = manager.log_environment_info()
    manager.save_environment_info(output_path)
    return env_info


def create_training_fingerprint(seed: int, config_dict: Dict[str, Any]) -> str:
    """
    Create a unique fingerprint for a training run combining seed and configuration.
    
    Args:
        seed: Random seed used for training
        config_dict: Training configuration dictionary
        
    Returns:
        SHA-256 hash string representing the training fingerprint
    """
    manager = ReproducibilityManager(seed=seed)
    env_info = manager.log_environment_info()
    
    # Combine configuration and environment for fingerprint
    fingerprint_data = {
        'seed': seed,
        'config': config_dict,
        'environment': {
            'python_version': env_info['python']['version'],
            'key_libraries': {
                lib: version for lib, version in env_info['libraries'].items()
                if lib in ['torch', 'transformers', 'datasets', 'peft', 'trl']
            },
            'cuda_version': env_info['cuda']['version']
        }
    }
    
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    hash_obj = hashlib.sha256(fingerprint_str.encode())
    return hash_obj.hexdigest()


def ensure_deterministic_environment() -> None:
    """
    Ensure the current environment is configured for deterministic training.
    
    This is a convenience function that sets up deterministic training without
    requiring explicit ReproducibilityManager instantiation.
    
    Requirement 15.2: Fixed random seeds for deterministic training
    """
    manager = ReproducibilityManager()
    manager.setup_deterministic_training()
    logger.info("Deterministic environment configured")