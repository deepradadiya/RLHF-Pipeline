"""
Reproducibility Features Demo

This script demonstrates how to use the reproducibility utilities in the RLHF Phi-3 pipeline
to ensure deterministic training results and comprehensive environment logging.

Requirements satisfied:
- 15.2: Fixed random seeds for deterministic training
- 15.3: Environment and library version logging
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import configuration and reproducibility utilities
from rlhf_phi3.config.config_manager import Config
from rlhf_phi3.utils.reproducibility import (
    ReproducibilityManager,
    setup_reproducible_training,
    log_training_environment,
    create_training_fingerprint
)


def demo_basic_reproducibility():
    """Demonstrate basic reproducibility features."""
    print("=== Basic Reproducibility Demo ===")
    
    # Create reproducibility manager
    seed = 42
    manager = ReproducibilityManager(seed=seed, enable_deterministic=True)
    
    print(f"Created ReproducibilityManager with seed: {seed}")
    print(f"Deterministic mode: {manager.enable_deterministic}")
    
    # Set random seeds for all libraries
    manager.set_random_seeds()
    print("✓ Random seeds set for all libraries")
    
    # Setup deterministic training environment
    manager.setup_deterministic_training()
    print("✓ Deterministic training environment configured")
    
    # Log environment information
    env_info = manager.log_environment_info()
    print(f"✓ Environment information collected ({len(env_info)} categories)")
    
    # Create reproducibility hash
    repro_hash = manager.create_reproducibility_hash()
    print(f"✓ Reproducibility hash: {repro_hash[:16]}...")
    
    # Get summary
    summary = manager.get_reproducibility_summary()
    print(f"✓ Reproducibility summary generated")
    print(f"  - Seed: {summary['seed']}")
    print(f"  - Python: {summary['key_versions']['python']['major']}.{summary['key_versions']['python']['minor']}")
    print(f"  - PyTorch: {summary['key_versions']['torch']}")
    print()


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("=== Convenience Functions Demo ===")
    
    # Setup reproducible training with one function call
    seed = 123
    manager = setup_reproducible_training(seed=seed)
    print(f"✓ Reproducible training setup complete (seed: {seed})")
    
    # Log training environment to file
    output_dir = project_root / "outputs" / "reproducibility_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env_path = output_dir / "training_environment.json"
    env_info = log_training_environment(env_path, seed=seed)
    print(f"✓ Environment logged to: {env_path}")
    
    # Create training fingerprint with configuration
    config = Config()
    config_dict = {
        'model_name': config.model.name,
        'max_length': config.model.max_length,
        'lora_r': config.lora.r,
        'sft_learning_rate': config.training.sft.learning_rate,
        'sft_batch_size': config.training.sft.batch_size
    }
    
    fingerprint = create_training_fingerprint(seed, config_dict)
    print(f"✓ Training fingerprint: {fingerprint[:16]}...")
    print()


def demo_config_integration():
    """Demonstrate integration with configuration system."""
    print("=== Configuration Integration Demo ===")
    
    # Create configuration
    config = Config()
    print(f"✓ Configuration created: {config.model.name}")
    
    # Setup reproducibility
    seed = 456
    manager = ReproducibilityManager(seed=seed)
    
    # Get stage-specific configurations with reproducibility
    stages = ['sft', 'reward', 'ppo']
    stage_fingerprints = {}
    
    for stage in stages:
        stage_config = config.get_stage_config(stage)
        fingerprint = create_training_fingerprint(seed, stage_config)
        stage_fingerprints[stage] = fingerprint
        print(f"✓ {stage.upper()} stage fingerprint: {fingerprint[:16]}...")
    
    # Verify fingerprints are unique per stage
    unique_fingerprints = set(stage_fingerprints.values())
    print(f"✓ Generated {len(unique_fingerprints)} unique fingerprints for {len(stages)} stages")
    
    # Save complete reproducibility record
    output_dir = project_root / "outputs" / "reproducibility_demo"
    record_path = output_dir / "reproducibility_record.json"
    
    import json
    record = {
        'seed': seed,
        'timestamp': manager.log_environment_info()['timestamp'],
        'reproducibility_hash': manager.create_reproducibility_hash(),
        'stage_fingerprints': stage_fingerprints,
        'config_summary': {
            'model_name': config.model.name,
            'max_length': config.model.max_length,
            'lora_r': config.lora.r,
            'sft_epochs': config.training.sft.epochs
        }
    }
    
    with open(record_path, 'w') as f:
        json.dump(record, f, indent=2)
    
    print(f"✓ Complete reproducibility record saved to: {record_path}")
    print()


def demo_reproducibility_validation():
    """Demonstrate reproducibility validation."""
    print("=== Reproducibility Validation Demo ===")
    
    seed = 789
    
    # First training run
    manager1 = ReproducibilityManager(seed=seed)
    hash1 = manager1.create_reproducibility_hash()
    print(f"✓ First run hash: {hash1[:16]}...")
    
    # Second training run with same seed and environment
    manager2 = ReproducibilityManager(seed=seed)
    hash2 = manager2.create_reproducibility_hash()
    print(f"✓ Second run hash: {hash2[:16]}...")
    
    # Validate reproducibility
    is_reproducible = manager2.validate_reproducibility(hash1)
    print(f"✓ Reproducibility validation: {'PASSED' if is_reproducible else 'FAILED'}")
    
    if is_reproducible:
        print("  → Same environment and seed produce identical results")
    else:
        print("  → Different environments detected (expected in different runs)")
    
    # Different seed should produce different hash
    manager3 = ReproducibilityManager(seed=seed + 1)
    hash3 = manager3.create_reproducibility_hash()
    is_different = not manager3.validate_reproducibility(hash1)
    print(f"✓ Different seed validation: {'PASSED' if is_different else 'FAILED'}")
    print()


def demo_training_workflow():
    """Demonstrate complete training workflow with reproducibility."""
    print("=== Complete Training Workflow Demo ===")
    
    # Step 1: Create and validate configuration
    config = Config()
    config.model.max_length = 1024  # Smaller for demo
    config.training.sft.batch_size = 2
    config.training.sft.max_steps = 100
    
    is_valid = config.validate_config()
    print(f"✓ Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Step 2: Setup reproducible training environment
    seed = 999
    manager = setup_reproducible_training(seed=seed, enable_deterministic=True)
    print(f"✓ Reproducible training environment setup (seed: {seed})")
    
    # Step 3: Create training metadata for each stage
    output_dir = project_root / "outputs" / "reproducibility_demo"
    stages = ['sft', 'reward', 'ppo']
    
    for stage in stages:
        stage_config = config.get_stage_config(stage)
        fingerprint = create_training_fingerprint(seed, stage_config)
        
        # Create stage-specific metadata
        stage_metadata = {
            'stage': stage,
            'seed': seed,
            'fingerprint': fingerprint,
            'timestamp': manager.log_environment_info()['timestamp'],
            'config': {
                'learning_rate': stage_config['training'].get('learning_rate', 'N/A'),
                'batch_size': stage_config['training'].get('batch_size', 'N/A'),
                'max_steps': stage_config['training'].get('max_steps', 'N/A')
            }
        }
        
        # Save stage metadata
        stage_path = output_dir / f"{stage}_training_metadata.json"
        with open(stage_path, 'w') as f:
            json.dump(stage_metadata, f, indent=2)
        
        print(f"✓ {stage.upper()} stage metadata saved: {stage_path}")
    
    print(f"✓ Complete training workflow metadata generated")
    print()


def main():
    """Run all reproducibility demos."""
    print("RLHF Phi-3 Pipeline - Reproducibility Features Demo")
    print("=" * 60)
    print()
    
    try:
        demo_basic_reproducibility()
        demo_convenience_functions()
        demo_config_integration()
        demo_reproducibility_validation()
        demo_training_workflow()
        
        print("🎉 All reproducibility demos completed successfully!")
        print()
        print("Key Features Demonstrated:")
        print("- ✅ Fixed random seed management for deterministic training")
        print("- ✅ Comprehensive environment and library version logging")
        print("- ✅ Integration with configuration management system")
        print("- ✅ Training fingerprint generation for unique identification")
        print("- ✅ Reproducibility validation across different runs")
        print("- ✅ Complete training workflow with reproducibility tracking")
        print()
        print("Output files saved to: outputs/reproducibility_demo/")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()