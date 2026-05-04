#!/usr/bin/env python3
"""
Setup Validation Script

This script validates that the RLHF Phi-3 pipeline project structure
has been created correctly and all components can be imported.
"""

import sys
import os
from pathlib import Path

def validate_directory_structure():
    """Validate that all required directories exist."""
    required_dirs = [
        "rlhf_phi3",
        "rlhf_phi3/config",
        "rlhf_phi3/data", 
        "rlhf_phi3/models",
        "rlhf_phi3/training",
        "rlhf_phi3/checkpoints",
        "rlhf_phi3/tracking",
        "rlhf_phi3/evaluation",
        "rlhf_phi3/publishing",
        "rlhf_phi3/utils",
        "tests",
        "tests/unit",
        "tests/property",
        "tests/integration",
        "notebooks",
        "configs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True

def validate_files():
    """Validate that all required files exist."""
    required_files = [
        "requirements.txt",
        "setup.py", 
        "pyproject.toml",
        "README.md",
        ".gitignore",
        "rlhf_phi3/__init__.py",
        "configs/default_config.yaml",
        "configs/colab_config.yaml",
        "tests/conftest.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def validate_imports():
    """Validate that the package can be imported."""
    try:
        sys.path.insert(0, '.')
        import rlhf_phi3
        
        # Check that all expected components are available
        expected_components = [
            'Config',
            'DatasetManager', 
            'ModelManager',
            'TrainingOrchestrator',
            'CheckpointManager',
            'ExperimentTracker',
            'EvaluationEngine'
        ]
        
        missing_components = []
        for component in expected_components:
            if not hasattr(rlhf_phi3, component):
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False
        else:
            print("✅ All components can be imported")
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🔍 Validating RLHF Phi-3 Pipeline Setup...")
    print("=" * 50)
    
    checks = [
        ("Directory Structure", validate_directory_structure),
        ("Required Files", validate_files),
        ("Package Imports", validate_imports)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Setup validation passed! Project structure is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run tests: pytest tests/")
        print("3. Start implementing Task 1.2: Configuration Manager")
    else:
        print("❌ Setup validation failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()