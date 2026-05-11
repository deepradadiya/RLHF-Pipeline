"""
Simple Reproducibility Features Demo

This script demonstrates the core reproducibility utilities without requiring
the full RLHF pipeline dependencies.

Requirements satisfied:
- 15.2: Fixed random seeds for deterministic training
- 15.3: Environment and library version logging
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load reproducibility utilities directly
exec(open('rlhf_phi3/utils/reproducibility.py').read())


def demo_basic_features():
    """Demonstrate basic reproducibility features."""
    print("=== Basic Reproducibility Features ===")
    
    # Create reproducibility manager
    seed = 42
    manager = ReproducibilityManager(seed=seed, enable_deterministic=True)
    
    print(f"✓ Created ReproducibilityManager with seed: {seed}")
    print(f"✓ Deterministic mode: {manager.enable_deterministic}")
    
    # Set random seeds
    manager.set_random_seeds()
    print("✓ Random seeds set for all available libraries")
    
    # Setup deterministic training
    manager.setup_deterministic_training()
    print("✓ Deterministic training environment configured")
    
    # Log environment information
    env_info = manager.log_environment_info()
    print(f"✓ Environment information collected:")
    print(f"  - Python version: {env_info['python']['version_info']['major']}.{env_info['python']['version_info']['minor']}")
    print(f"  - System: {env_info['system']['system']}")
    print(f"  - Libraries tracked: {len(env_info['libraries'])}")
    
    # Create reproducibility hash
    repro_hash = manager.create_reproducibility_hash()
    print(f"✓ Reproducibility hash: {repro_hash[:16]}...")
    
    return manager, env_info


def demo_environment_logging():
    """Demonstrate environment logging capabilities."""
    print("\n=== Environment Logging Demo ===")
    
    # Create output directory
    output_dir = project_root / "outputs" / "reproducibility_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log environment with convenience function
    seed = 123
    env_path = output_dir / "environment_info.json"
    env_info = log_training_environment(env_path, seed=seed)
    
    print(f"✓ Environment logged to: {env_path}")
    print(f"✓ Environment contains {len(env_info)} information categories")
    
    # Show some key information
    print("✓ Key environment details:")
    print(f"  - Timestamp: {env_info['timestamp']}")
    print(f"  - Seed: {env_info['seed']}")
    print(f"  - Platform: {env_info['system']['platform']}")
    
    # Show library versions
    key_libs = ['torch', 'transformers', 'numpy', 'pandas']
    print("✓ Key library versions:")
    for lib in key_libs:
        version = env_info['libraries'].get(lib, 'not_installed')
        print(f"  - {lib}: {version}")
    
    return env_info


def demo_training_fingerprints():
    """Demonstrate training fingerprint generation."""
    print("\n=== Training Fingerprint Demo ===")
    
    # Create sample configurations
    configs = {
        'sft_config': {
            'model_name': 'microsoft/Phi-3-mini-4k-instruct',
            'learning_rate': 2e-4,
            'batch_size': 4,
            'max_steps': 1000,
            'lora_r': 16
        },
        'reward_config': {
            'model_name': 'microsoft/Phi-3-mini-4k-instruct',
            'learning_rate': 1e-4,
            'batch_size': 2,
            'max_steps': 500,
            'lora_r': 16
        },
        'ppo_config': {
            'model_name': 'microsoft/Phi-3-mini-4k-instruct',
            'learning_rate': 1e-5,
            'batch_size': 1,
            'max_steps': 200,
            'lora_r': 16
        }
    }
    
    seed = 456
    fingerprints = {}
    
    for config_name, config in configs.items():
        fingerprint = create_training_fingerprint(seed, config)
        fingerprints[config_name] = fingerprint
        print(f"✓ {config_name}: {fingerprint[:16]}...")
    
    # Verify uniqueness
    unique_fingerprints = set(fingerprints.values())
    print(f"✓ Generated {len(unique_fingerprints)} unique fingerprints for {len(configs)} configurations")
    
    # Test reproducibility
    same_fingerprint = create_training_fingerprint(seed, configs['sft_config'])
    is_same = same_fingerprint == fingerprints['sft_config']
    print(f"✓ Reproducibility test: {'PASSED' if is_same else 'FAILED'}")
    
    return fingerprints


def demo_reproducibility_validation():
    """Demonstrate reproducibility validation."""
    print("\n=== Reproducibility Validation Demo ===")
    
    seed = 789
    
    # Create two managers with same seed
    manager1 = ReproducibilityManager(seed=seed)
    manager2 = ReproducibilityManager(seed=seed)
    
    # Generate hashes
    hash1 = manager1.create_reproducibility_hash()
    hash2 = manager2.create_reproducibility_hash()
    
    print(f"✓ Manager 1 hash: {hash1[:16]}...")
    print(f"✓ Manager 2 hash: {hash2[:16]}...")
    
    # Validate reproducibility
    is_reproducible = manager2.validate_reproducibility(hash1)
    print(f"✓ Same seed validation: {'PASSED' if is_reproducible else 'FAILED'}")
    
    # Test with different seed
    manager3 = ReproducibilityManager(seed=seed + 1)
    hash3 = manager3.create_reproducibility_hash()
    is_different = not manager3.validate_reproducibility(hash1)
    
    print(f"✓ Different seed hash: {hash3[:16]}...")
    print(f"✓ Different seed validation: {'PASSED' if is_different else 'FAILED'}")


def demo_complete_workflow():
    """Demonstrate complete reproducibility workflow."""
    print("\n=== Complete Workflow Demo ===")
    
    # Step 1: Setup reproducible training
    seed = 999
    manager = setup_reproducible_training(seed=seed, enable_deterministic=True)
    print(f"✓ Step 1: Reproducible training setup (seed: {seed})")
    
    # Step 2: Create training configurations
    training_stages = {
        'sft': {
            'learning_rate': 2e-4,
            'batch_size': 4,
            'epochs': 3,
            'max_steps': 1000
        },
        'reward': {
            'learning_rate': 1e-4,
            'batch_size': 2,
            'epochs': 1,
            'max_steps': 500
        },
        'ppo': {
            'learning_rate': 1e-5,
            'batch_size': 1,
            'ppo_epochs': 4,
            'max_steps': 200
        }
    }
    
    print("✓ Step 2: Training configurations created")
    
    # Step 3: Generate reproducibility records for each stage
    output_dir = project_root / "outputs" / "reproducibility_demo"
    
    for stage_name, stage_config in training_stages.items():
        # Create fingerprint
        fingerprint = create_training_fingerprint(seed, stage_config)
        
        # Create complete record
        record = {
            'stage': stage_name,
            'seed': seed,
            'fingerprint': fingerprint,
            'timestamp': manager.log_environment_info()['timestamp'],
            'reproducibility_hash': manager.create_reproducibility_hash(),
            'configuration': stage_config,
            'environment_summary': manager.get_reproducibility_summary()
        }
        
        # Save record
        record_path = output_dir / f"{stage_name}_reproducibility_record.json"
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2, default=str)
        
        print(f"✓ Step 3.{list(training_stages.keys()).index(stage_name) + 1}: {stage_name.upper()} record saved")
    
    print("✓ Step 4: Complete workflow records generated")
    
    # Step 4: Create master reproducibility manifest
    manifest = {
        'project': 'RLHF Phi-3 Pipeline',
        'seed': seed,
        'timestamp': manager.log_environment_info()['timestamp'],
        'stages': list(training_stages.keys()),
        'reproducibility_hash': manager.create_reproducibility_hash(),
        'environment_summary': manager.get_reproducibility_summary(),
        'files': [f"{stage}_reproducibility_record.json" for stage in training_stages.keys()]
    }
    
    manifest_path = output_dir / "reproducibility_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"✓ Step 5: Master manifest saved: {manifest_path}")


def main():
    """Run all reproducibility demos."""
    print("RLHF Phi-3 Pipeline - Reproducibility Features Demo")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_basic_features()
        demo_environment_logging()
        demo_training_fingerprints()
        demo_reproducibility_validation()
        demo_complete_workflow()
        
        print("\n" + "=" * 60)
        print("🎉 All reproducibility demos completed successfully!")
        print()
        print("Features Demonstrated:")
        print("✅ Fixed random seed management for deterministic training")
        print("✅ Comprehensive environment and library version logging")
        print("✅ Training fingerprint generation for unique identification")
        print("✅ Reproducibility validation across different runs")
        print("✅ Complete training workflow with reproducibility tracking")
        print()
        
        # Show output files
        output_dir = Path(__file__).parent.parent / "outputs" / "reproducibility_demo"
        if output_dir.exists():
            files = list(output_dir.glob("*.json"))
            print(f"📁 Output files generated ({len(files)} files):")
            for file in sorted(files):
                print(f"   - {file.name}")
            print(f"   Location: {output_dir}")
        
        print("\n✨ Reproducibility features are ready for use in the RLHF pipeline!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()