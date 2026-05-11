#!/usr/bin/env python3
"""
Environment Recreation Script for RLHF Phi-3 Pipeline

This script recreates the exact training environment from saved reproducibility metadata.
It can be used to reproduce training results or validate environment consistency.

Requirements satisfied:
- 15.5: Reproducibility scripts and environment recreation

Usage:
    python scripts/recreate_environment.py --metadata path/to/environment_info.json
    python scripts/recreate_environment.py --checkpoint path/to/checkpoint/dir
    python scripts/recreate_environment.py --model path/to/model/dir
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rlhf_phi3.utils.reproducibility import (
    ReproducibilityManager,
    setup_reproducible_training,
    ensure_deterministic_environment
)

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentRecreationResult:
    """Result of environment recreation attempt."""
    success: bool
    matched_components: List[str]
    mismatched_components: List[str]
    missing_components: List[str]
    warnings: List[str]
    errors: List[str]


class EnvironmentRecreator:
    """
    Recreates training environments from saved metadata.
    
    This class can recreate environments from:
    - Environment metadata files
    - Checkpoint directories with embedded metadata
    - Model directories with training provenance
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize environment recreator."""
        self.verbose = verbose
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def recreate_from_metadata_file(self, metadata_path: Path) -> EnvironmentRecreationResult:
        """
        Recreate environment from a metadata JSON file.
        
        Args:
            metadata_path: Path to environment metadata JSON file
            
        Returns:
            Environment recreation result
        """
        logger.info(f"Recreating environment from metadata file: {metadata_path}")
        
        if not metadata_path.exists():
            return EnvironmentRecreationResult(
                success=False,
                matched_components=[],
                mismatched_components=[],
                missing_components=[],
                warnings=[],
                errors=[f"Metadata file not found: {metadata_path}"]
            )
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return self._recreate_from_metadata(metadata)
            
        except Exception as e:
            return EnvironmentRecreationResult(
                success=False,
                matched_components=[],
                mismatched_components=[],
                missing_components=[],
                warnings=[],
                errors=[f"Failed to load metadata: {e}"]
            )
    
    def recreate_from_checkpoint(self, checkpoint_path: Path) -> EnvironmentRecreationResult:
        """
        Recreate environment from checkpoint directory.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Environment recreation result
        """
        logger.info(f"Recreating environment from checkpoint: {checkpoint_path}")
        
        # Look for environment metadata in checkpoint
        metadata_files = [
            checkpoint_path / "environment_info.json",
            checkpoint_path / "config_snapshot.json",
            checkpoint_path / "metadata.json"
        ]
        
        metadata = None
        for metadata_file in metadata_files:
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract environment info from different file types
                    if "environment_info" in data:
                        metadata = data["environment_info"]
                        break
                    elif "reproducibility" in data:
                        metadata = data["reproducibility"]
                        break
                    elif "libraries" in data:  # Direct environment file
                        metadata = data
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load {metadata_file}: {e}")
        
        if metadata is None:
            return EnvironmentRecreationResult(
                success=False,
                matched_components=[],
                mismatched_components=[],
                missing_components=[],
                warnings=[],
                errors=["No environment metadata found in checkpoint"]
            )
        
        return self._recreate_from_metadata(metadata)
    
    def recreate_from_model(self, model_path: Path) -> EnvironmentRecreationResult:
        """
        Recreate environment from model directory with training provenance.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Environment recreation result
        """
        logger.info(f"Recreating environment from model: {model_path}")
        
        # Look for training provenance files
        provenance_files = [
            model_path / "training_provenance.json",
            model_path / "config_snapshot.json",
            model_path / "environment_info.json"
        ]
        
        metadata = None
        for provenance_file in provenance_files:
            if provenance_file.exists():
                try:
                    with open(provenance_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract environment info from provenance
                    if "environment" in data:
                        metadata = data["environment"]
                        break
                    elif "environment_info" in data:
                        metadata = data["environment_info"]
                        break
                    elif "libraries" in data:  # Direct environment file
                        metadata = data
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load {provenance_file}: {e}")
        
        if metadata is None:
            return EnvironmentRecreationResult(
                success=False,
                matched_components=[],
                mismatched_components=[],
                missing_components=[],
                warnings=[],
                errors=["No training provenance found in model directory"]
            )
        
        return self._recreate_from_metadata(metadata)
    
    def _recreate_from_metadata(self, metadata: Dict[str, Any]) -> EnvironmentRecreationResult:
        """
        Recreate environment from metadata dictionary.
        
        Args:
            metadata: Environment metadata dictionary
            
        Returns:
            Environment recreation result
        """
        result = EnvironmentRecreationResult(
            success=True,
            matched_components=[],
            mismatched_components=[],
            missing_components=[],
            warnings=[],
            errors=[]
        )
        
        logger.info("Analyzing current environment against target metadata")
        
        # Get current environment info
        current_manager = ReproducibilityManager()
        current_env = current_manager.log_environment_info()
        
        # Compare Python version
        self._compare_python_version(metadata, current_env, result)
        
        # Compare library versions
        self._compare_library_versions(metadata, current_env, result)
        
        # Compare CUDA version
        self._compare_cuda_version(metadata, current_env, result)
        
        # Compare system information
        self._compare_system_info(metadata, current_env, result)
        
        # Check reproducibility seed
        self._check_reproducibility_seed(metadata, result)
        
        # Generate recommendations
        self._generate_recommendations(metadata, result)
        
        # Determine overall success
        result.success = len(result.errors) == 0 and len(result.mismatched_components) == 0
        
        return result
    
    def _compare_python_version(self, target: Dict[str, Any], current: Dict[str, Any], 
                               result: EnvironmentRecreationResult) -> None:
        """Compare Python versions."""
        target_python = target.get("python", {})
        current_python = current.get("python", {})
        
        target_version = target_python.get("version_info", {})
        current_version = current_python.get("version_info", {})
        
        if target_version and current_version:
            target_major_minor = f"{target_version.get('major', 0)}.{target_version.get('minor', 0)}"
            current_major_minor = f"{current_version.get('major', 0)}.{current_version.get('minor', 0)}"
            
            if target_major_minor == current_major_minor:
                result.matched_components.append(f"Python {current_major_minor}")
            else:
                result.mismatched_components.append(
                    f"Python version (target: {target_major_minor}, current: {current_major_minor})"
                )
        else:
            result.warnings.append("Could not compare Python versions")
    
    def _compare_library_versions(self, target: Dict[str, Any], current: Dict[str, Any],
                                 result: EnvironmentRecreationResult) -> None:
        """Compare library versions."""
        target_libs = target.get("libraries", {})
        current_libs = current.get("libraries", {})
        
        # Key libraries to check
        key_libraries = [
            "torch", "transformers", "datasets", "peft", "trl", 
            "accelerate", "numpy", "pandas", "wandb"
        ]
        
        for lib in key_libraries:
            target_version = target_libs.get(lib, "not_installed")
            current_version = current_libs.get(lib, "not_installed")
            
            if target_version == "not_installed" and current_version == "not_installed":
                continue  # Both not installed, skip
            elif target_version == "not_installed":
                result.warnings.append(f"{lib} not required in target but installed in current")
            elif current_version == "not_installed":
                result.missing_components.append(f"{lib} {target_version}")
            elif target_version == current_version:
                result.matched_components.append(f"{lib} {current_version}")
            else:
                result.mismatched_components.append(
                    f"{lib} (target: {target_version}, current: {current_version})"
                )
    
    def _compare_cuda_version(self, target: Dict[str, Any], current: Dict[str, Any],
                             result: EnvironmentRecreationResult) -> None:
        """Compare CUDA versions."""
        target_cuda = target.get("cuda", {})
        current_cuda = current.get("cuda", {})
        
        target_version = target_cuda.get("version")
        current_version = current_cuda.get("version")
        
        if target_version and current_version:
            if target_version == current_version:
                result.matched_components.append(f"CUDA {current_version}")
            else:
                result.mismatched_components.append(
                    f"CUDA version (target: {target_version}, current: {current_version})"
                )
        elif target_version:
            result.missing_components.append(f"CUDA {target_version}")
        elif current_version:
            result.warnings.append(f"CUDA {current_version} available but not required in target")
    
    def _compare_system_info(self, target: Dict[str, Any], current: Dict[str, Any],
                            result: EnvironmentRecreationResult) -> None:
        """Compare system information."""
        target_system = target.get("system", {})
        current_system = current.get("system", {})
        
        # Compare platform
        target_platform = target_system.get("platform", "")
        current_platform = current_system.get("platform", "")
        
        if target_platform and current_platform:
            # Extract OS name for comparison
            target_os = target_platform.split('-')[0] if '-' in target_platform else target_platform
            current_os = current_platform.split('-')[0] if '-' in current_platform else current_platform
            
            if target_os.lower() == current_os.lower():
                result.matched_components.append(f"OS {current_os}")
            else:
                result.warnings.append(
                    f"Different OS (target: {target_os}, current: {current_os})"
                )
    
    def _check_reproducibility_seed(self, metadata: Dict[str, Any], 
                                   result: EnvironmentRecreationResult) -> None:
        """Check if reproducibility seed is available."""
        seed = metadata.get("seed")
        if seed is not None:
            result.matched_components.append(f"Reproducibility seed: {seed}")
            
            # Setup reproducible environment with the same seed
            try:
                setup_reproducible_training(seed=seed)
                result.matched_components.append("Deterministic environment configured")
            except Exception as e:
                result.warnings.append(f"Failed to setup deterministic environment: {e}")
        else:
            result.warnings.append("No reproducibility seed found in metadata")
    
    def _generate_recommendations(self, metadata: Dict[str, Any], 
                                 result: EnvironmentRecreationResult) -> None:
        """Generate recommendations for environment recreation."""
        if result.missing_components or result.mismatched_components:
            result.warnings.append("Environment recreation recommendations:")
            
            # Generate pip install commands for missing libraries
            if result.missing_components:
                missing_libs = []
                for component in result.missing_components:
                    if "CUDA" not in component and "Python" not in component:
                        lib_info = component.split()
                        if len(lib_info) >= 2:
                            lib_name = lib_info[0]
                            lib_version = lib_info[1]
                            missing_libs.append(f"{lib_name}=={lib_version}")
                
                if missing_libs:
                    install_cmd = f"pip install {' '.join(missing_libs)}"
                    result.warnings.append(f"Install missing libraries: {install_cmd}")
            
            # Generate upgrade commands for mismatched libraries
            if result.mismatched_components:
                target_libs = metadata.get("libraries", {})
                upgrade_libs = []
                
                for component in result.mismatched_components:
                    if "(" in component and "target:" in component:
                        lib_name = component.split()[0]
                        if lib_name in target_libs:
                            target_version = target_libs[lib_name]
                            if target_version != "not_installed":
                                upgrade_libs.append(f"{lib_name}=={target_version}")
                
                if upgrade_libs:
                    upgrade_cmd = f"pip install {' '.join(upgrade_libs)}"
                    result.warnings.append(f"Upgrade libraries: {upgrade_cmd}")
    
    def generate_requirements_file(self, metadata: Dict[str, Any], 
                                  output_path: Path) -> None:
        """
        Generate requirements.txt file from environment metadata.
        
        Args:
            metadata: Environment metadata
            output_path: Path to save requirements.txt
        """
        logger.info(f"Generating requirements.txt from metadata: {output_path}")
        
        libraries = metadata.get("libraries", {})
        
        # Filter and format library versions
        requirements = []
        for lib_name, version in libraries.items():
            if version != "not_installed" and not version.startswith("error:"):
                # Skip system libraries and special cases
                if lib_name not in ["google-colab"]:  # Skip Colab-specific libraries
                    requirements.append(f"{lib_name}=={version}")
        
        # Sort requirements
        requirements.sort()
        
        # Add header
        header = [
            "# Requirements file generated from RLHF Phi-3 Pipeline environment metadata",
            f"# Generated on: {metadata.get('timestamp', 'unknown')}",
            f"# Python version: {metadata.get('python', {}).get('version', 'unknown')}",
            f"# CUDA version: {metadata.get('cuda', {}).get('version', 'unknown')}",
            "",
        ]
        
        # Write requirements file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(header + requirements))
        
        logger.info(f"Requirements file saved with {len(requirements)} packages")
    
    def generate_environment_script(self, metadata: Dict[str, Any], 
                                   output_path: Path) -> None:
        """
        Generate shell script to recreate environment.
        
        Args:
            metadata: Environment metadata
            output_path: Path to save the script
        """
        logger.info(f"Generating environment recreation script: {output_path}")
        
        python_version = metadata.get("python", {}).get("version_info", {})
        python_ver = f"{python_version.get('major', 3)}.{python_version.get('minor', 8)}"
        
        libraries = metadata.get("libraries", {})
        seed = metadata.get("seed", 42)
        
        script_content = f"""#!/bin/bash
# Environment Recreation Script for RLHF Phi-3 Pipeline
# Generated from training metadata

set -e  # Exit on any error

echo "=== RLHF Phi-3 Pipeline Environment Recreation ==="
echo "Target Python version: {python_ver}"
echo "Reproducibility seed: {seed}"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$PYTHON_VERSION" != "{python_ver}" ]; then
    echo "WARNING: Python version mismatch. Expected {python_ver}, found $PYTHON_VERSION"
    echo "Consider using pyenv or conda to install Python {python_ver}"
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv rlhf_phi3_env
source rlhf_phi3_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "Installing core ML libraries..."
"""
        
        # Add library installations
        core_libs = ["torch", "transformers", "datasets", "peft", "trl", "accelerate"]
        other_libs = []
        
        for lib_name, version in libraries.items():
            if version != "not_installed" and not version.startswith("error:"):
                if lib_name in core_libs:
                    script_content += f'pip install "{lib_name}=={version}"\n'
                elif lib_name not in ["google-colab"]:  # Skip Colab-specific
                    other_libs.append(f"{lib_name}=={version}")
        
        if other_libs:
            script_content += f'\necho "Installing additional libraries..."\n'
            # Install in batches to avoid command line length issues
            batch_size = 10
            for i in range(0, len(other_libs), batch_size):
                batch = other_libs[i:i+batch_size]
                script_content += f'pip install {" ".join(f\'"{lib}"\' for lib in batch)}\n'
        
        script_content += f"""
# Set environment variables for reproducibility
export PYTHONHASHSEED={seed}
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Verify installation
echo ""
echo "=== Environment Verification ==="
python3 -c "
import sys
print(f'Python version: {{sys.version}}')

try:
    import torch
    print(f'PyTorch version: {{torch.__version__}}')
    print(f'CUDA available: {{torch.cuda.is_available()}}')
    if torch.cuda.is_available():
        print(f'CUDA version: {{torch.version.cuda}}')
        print(f'GPU count: {{torch.cuda.device_count()}}')
except ImportError:
    print('PyTorch not available')

try:
    import transformers
    print(f'Transformers version: {{transformers.__version__}}')
except ImportError:
    print('Transformers not available')

try:
    from rlhf_phi3.utils.reproducibility import setup_reproducible_training
    manager = setup_reproducible_training(seed={seed})
    print('Reproducibility utilities available')
except ImportError:
    print('RLHF Phi-3 pipeline not available')
"

echo ""
echo "=== Environment Recreation Complete ==="
echo "To activate this environment in the future, run:"
echo "source rlhf_phi3_env/bin/activate"
echo ""
echo "To reproduce training with the same seed, set:"
echo "export PYTHONHASHSEED={seed}"
"""
        
        # Write script
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        output_path.chmod(0o755)
        
        logger.info("Environment recreation script generated")


def main():
    """Main function for environment recreation script."""
    parser = argparse.ArgumentParser(
        description="Recreate training environment from saved metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recreate from environment metadata file
  python scripts/recreate_environment.py --metadata outputs/environment_info.json
  
  # Recreate from checkpoint directory
  python scripts/recreate_environment.py --checkpoint checkpoints/sft_epoch_1_step_500/
  
  # Recreate from model directory
  python scripts/recreate_environment.py --model models/final_rlhf_model/
  
  # Generate requirements.txt and setup script
  python scripts/recreate_environment.py --metadata outputs/environment_info.json --generate-files
        """
    )
    
    # Input source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--metadata", 
        type=Path,
        help="Path to environment metadata JSON file"
    )
    source_group.add_argument(
        "--checkpoint", 
        type=Path,
        help="Path to checkpoint directory with embedded metadata"
    )
    source_group.add_argument(
        "--model", 
        type=Path,
        help="Path to model directory with training provenance"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/environment_recreation"),
        help="Directory to save generated files (default: outputs/environment_recreation)"
    )
    parser.add_argument(
        "--generate-files",
        action="store_true",
        help="Generate requirements.txt and setup script"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle verbosity
    verbose = args.verbose and not args.quiet
    
    # Create recreator
    recreator = EnvironmentRecreator(verbose=verbose)
    
    # Determine source and recreate environment
    if args.metadata:
        result = recreator.recreate_from_metadata_file(args.metadata)
        source_name = args.metadata.name
    elif args.checkpoint:
        result = recreator.recreate_from_checkpoint(args.checkpoint)
        source_name = args.checkpoint.name
    else:  # args.model
        result = recreator.recreate_from_model(args.model)
        source_name = args.model.name
    
    # Print results
    print("\n" + "=" * 60)
    print("ENVIRONMENT RECREATION RESULTS")
    print("=" * 60)
    print(f"Source: {source_name}")
    print(f"Success: {'✅ YES' if result.success else '❌ NO'}")
    print()
    
    if result.matched_components:
        print("✅ Matched Components:")
        for component in result.matched_components:
            print(f"  - {component}")
        print()
    
    if result.mismatched_components:
        print("⚠️  Mismatched Components:")
        for component in result.mismatched_components:
            print(f"  - {component}")
        print()
    
    if result.missing_components:
        print("❌ Missing Components:")
        for component in result.missing_components:
            print(f"  - {component}")
        print()
    
    if result.warnings:
        print("⚠️  Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
        print()
    
    if result.errors:
        print("❌ Errors:")
        for error in result.errors:
            print(f"  - {error}")
        print()
    
    # Generate files if requested
    if args.generate_files and not result.errors:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata for file generation
        metadata = None
        if args.metadata:
            with open(args.metadata, 'r') as f:
                metadata = json.load(f)
        elif args.checkpoint:
            # Try to load from checkpoint
            metadata_files = [
                args.checkpoint / "environment_info.json",
                args.checkpoint / "config_snapshot.json"
            ]
            for metadata_file in metadata_files:
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        metadata = data.get("environment_info") or data.get("reproducibility") or data
                        break
        elif args.model:
            # Try to load from model
            provenance_files = [
                args.model / "training_provenance.json",
                args.model / "environment_info.json"
            ]
            for provenance_file in provenance_files:
                if provenance_file.exists():
                    with open(provenance_file, 'r') as f:
                        data = json.load(f)
                        metadata = data.get("environment") or data
                        break
        
        if metadata:
            # Generate requirements.txt
            requirements_path = args.output_dir / "requirements.txt"
            recreator.generate_requirements_file(metadata, requirements_path)
            print(f"📄 Generated requirements.txt: {requirements_path}")
            
            # Generate setup script
            script_path = args.output_dir / "setup_environment.sh"
            recreator.generate_environment_script(metadata, script_path)
            print(f"📜 Generated setup script: {script_path}")
            
            print(f"\n📁 Files saved to: {args.output_dir}")
        else:
            print("❌ Could not load metadata for file generation")
    
    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()