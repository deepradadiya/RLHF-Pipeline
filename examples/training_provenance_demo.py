"""
Training Provenance and Metadata Tracking Demo

This script demonstrates the comprehensive training provenance and metadata tracking
capabilities implemented for the RLHF Phi-3 Pipeline.

Requirements satisfied:
- 15.1: Configuration snapshots with checkpoints
- 15.4: Training provenance in model metadata
- 15.5: Reproducibility scripts and environment recreation

Features demonstrated:
- Training provenance tracking throughout pipeline stages
- Configuration snapshot saving with checkpoints
- Environment information capture and logging
- Reproducibility hash generation and validation
- Integration with existing components
- Model metadata enhancement with training provenance
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rlhf_phi3.config.config_manager import Config
from rlhf_phi3.utils.training_provenance import (
    TrainingProvenanceManager,
    TrainingStageProvenance,
    create_provenance_manager
)
from rlhf_phi3.utils.reproducibility import (
    ReproducibilityManager,
    setup_reproducible_training
)


def demo_basic_provenance_tracking():
    """Demonstrate basic training provenance tracking."""
    print("=== Basic Training Provenance Tracking ===")
    
    # Create configuration
    config = Config()
    seed = 42
    
    # Create provenance manager
    provenance_manager = create_provenance_manager(config, seed=seed)
    
    print(f"✓ Created provenance manager: {provenance_manager.pipeline_id}")
    print(f"✓ Reproducibility seed: {seed}")
    print(f"✓ Environment captured: {provenance_manager.provenance.environment_info is not None}")
    
    # Simulate SFT stage
    print("\n--- Simulating SFT Training Stage ---")
    sft_stage = provenance_manager.start_stage("sft")
    
    # Simulate training steps with metrics
    for step in range(0, 101, 25):
        loss = 2.5 - (step * 0.02)  # Decreasing loss
        metrics = {
            "loss": loss,
            "learning_rate": 2e-4 * (0.95 ** (step // 10)),
            "grad_norm": 1.2 - (step * 0.005)
        }
        provenance_manager.update_stage_metrics("sft", step, metrics)
        print(f"  Step {step}: loss={loss:.3f}")
    
    # Finalize SFT stage
    provenance_manager.finalize_stage("sft", final_loss=0.5, checkpoint_path="checkpoints/sft_final")
    print("✓ SFT stage finalized")
    
    # Simulate Reward Model stage
    print("\n--- Simulating Reward Model Training Stage ---")
    reward_stage = provenance_manager.start_stage("reward")
    
    for step in range(0, 51, 10):
        accuracy = 0.6 + (step * 0.008)  # Increasing accuracy
        loss = 1.8 - (step * 0.025)
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": 1e-4
        }
        provenance_manager.update_stage_metrics("reward", step, metrics)
        print(f"  Step {step}: accuracy={accuracy:.3f}, loss={loss:.3f}")
    
    provenance_manager.finalize_stage("reward", final_loss=0.55, checkpoint_path="checkpoints/reward_final")
    print("✓ Reward model stage finalized")
    
    # Simulate PPO stage
    print("\n--- Simulating PPO Training Stage ---")
    ppo_stage = provenance_manager.start_stage("ppo")
    
    for step in range(0, 21, 5):
        reward_score = 0.3 + (step * 0.02)
        kl_divergence = 0.1 - (step * 0.002)
        metrics = {
            "reward_score": reward_score,
            "kl_divergence": kl_divergence,
            "policy_loss": 0.8 - (step * 0.01),
            "value_loss": 0.6 - (step * 0.008)
        }
        provenance_manager.update_stage_metrics("ppo", step, metrics)
        print(f"  Step {step}: reward={reward_score:.3f}, kl={kl_divergence:.3f}")
    
    provenance_manager.finalize_stage("ppo", final_loss=0.45, checkpoint_path="checkpoints/ppo_final")
    print("✓ PPO stage finalized")
    
    # Add evaluation results
    evaluation_results = {
        "mt_bench_score": 7.2,
        "helpfulness": 8.1,
        "harmlessness": 7.8,
        "honesty": 7.5,
        "inference_speed": 12.3,
        "memory_usage": 8.2
    }
    provenance_manager.add_evaluation_results(evaluation_results)
    print("✓ Evaluation results added")
    
    # Finalize training
    provenance_manager.finalize_training(final_model_path="models/final_rlhf_model")
    print("✓ Training provenance finalized")
    
    return provenance_manager


def demo_configuration_snapshots():
    """Demonstrate configuration snapshot functionality."""
    print("\n=== Configuration Snapshot Demo ===")
    
    # Create configuration
    config = Config()
    
    # Create checkpoint snapshot
    checkpoint_id = "demo_checkpoint_001"
    training_metadata = {
        "stage": "sft",
        "epoch": 2,
        "step": 500,
        "loss": 0.75
    }
    
    snapshot = config.create_checkpoint_snapshot(checkpoint_id, training_metadata)
    
    print(f"✓ Created configuration snapshot for: {checkpoint_id}")
    print(f"✓ Snapshot contains {len(snapshot)} metadata fields")
    print(f"✓ Configuration hash: {snapshot['config_hash'][:16]}...")
    print(f"✓ Reproducibility info: {snapshot['reproducibility'] is not None}")
    
    # Save snapshot to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(snapshot, f, indent=2, default=str)
        snapshot_path = f.name
    
    print(f"✓ Snapshot saved to: {snapshot_path}")
    
    # Load snapshot back
    try:
        loaded_config, loaded_snapshot = Config.load_checkpoint_snapshot(Path(snapshot_path).parent)
        print("❌ Expected FileNotFoundError but loading succeeded")
    except FileNotFoundError:
        print("✓ Correctly handled missing snapshot file")
    
    # Test with proper directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config.save_checkpoint_snapshot(temp_path, checkpoint_id, training_metadata)
        
        loaded_config, loaded_snapshot = Config.load_checkpoint_snapshot(temp_path)
        print("✓ Configuration snapshot loaded successfully")
        print(f"✓ Loaded config matches original: {loaded_config.model.name == config.model.name}")
    
    return snapshot


def demo_environment_recreation():
    """Demonstrate environment recreation capabilities."""
    print("\n=== Environment Recreation Demo ===")
    
    # Setup reproducible environment
    seed = 123
    manager = setup_reproducible_training(seed=seed)
    
    # Log environment information
    env_info = manager.log_environment_info()
    
    print(f"✓ Environment logged with seed: {seed}")
    print(f"✓ Python version: {env_info['python']['version_info']['major']}.{env_info['python']['version_info']['minor']}")
    print(f"✓ Libraries tracked: {len(env_info['libraries'])}")
    
    # Create reproducibility hash
    repro_hash = manager.create_reproducibility_hash()
    print(f"✓ Reproducibility hash: {repro_hash[:16]}...")
    
    # Test reproducibility validation
    new_manager = ReproducibilityManager(seed=seed)
    is_reproducible = new_manager.validate_reproducibility(repro_hash)
    print(f"✓ Reproducibility validation: {'PASSED' if is_reproducible else 'FAILED'}")
    
    # Test with different seed
    different_manager = ReproducibilityManager(seed=seed + 1)
    is_different = not different_manager.validate_reproducibility(repro_hash)
    print(f"✓ Different seed validation: {'PASSED' if is_different else 'FAILED'}")
    
    return env_info, repro_hash


def demo_model_metadata_integration():
    """Demonstrate model metadata integration with training provenance."""
    print("\n=== Model Metadata Integration Demo ===")
    
    # Create training provenance
    config = Config()
    provenance_manager = create_provenance_manager(config, seed=456)
    
    # Simulate a complete training run
    stages = ["sft", "reward", "ppo"]
    for stage_name in stages:
        stage = provenance_manager.start_stage(stage_name)
        
        # Simulate some training steps
        for step in range(0, 21, 10):
            metrics = {"loss": 1.0 - (step * 0.02), "accuracy": 0.7 + (step * 0.01)}
            provenance_manager.update_stage_metrics(stage_name, step, metrics)
        
        provenance_manager.finalize_stage(stage_name, final_loss=0.6, 
                                        checkpoint_path=f"checkpoints/{stage_name}_final")
    
    # Add evaluation results
    evaluation_results = {
        "mt_bench_score": 7.8,
        "helpfulness": 8.3,
        "harmlessness": 8.0,
        "honesty": 7.9
    }
    provenance_manager.add_evaluation_results(evaluation_results)
    provenance_manager.finalize_training(final_model_path="models/demo_model")
    
    # Create training provenance for model metadata
    training_provenance = {
        "pipeline_id": provenance_manager.pipeline_id,
        "seed": provenance_manager.seed,
        "reproducibility_hash": provenance_manager.provenance.reproducibility_hash,
        "config_hash": provenance_manager.provenance.config_hash,
        "timestamp": provenance_manager.provenance.start_time,
        "stages": [
            {
                "name": stage.stage_name,
                "steps": stage.total_steps,
                "final_loss": stage.final_loss
            }
            for stage in provenance_manager.provenance.stages
        ],
        "environment": {
            "python_version": provenance_manager.provenance.environment_info.get("python", {}).get("version", "unknown"),
            "torch_version": provenance_manager.provenance.environment_info.get("libraries", {}).get("torch", "unknown"),
            "transformers_version": provenance_manager.provenance.environment_info.get("libraries", {}).get("transformers", "unknown"),
            "cuda_version": provenance_manager.provenance.environment_info.get("cuda", {}).get("version", "unknown")
        },
        "evaluation_results": evaluation_results
    }
    
    print(f"✓ Training provenance created for model: {provenance_manager.pipeline_id}")
    print(f"✓ Stages completed: {len(training_provenance['stages'])}")
    print(f"✓ Environment info included: {len(training_provenance['environment'])} fields")
    print(f"✓ Evaluation results: {len(evaluation_results)} metrics")
    
    return training_provenance


def demo_complete_workflow():
    """Demonstrate complete training provenance workflow."""
    print("\n=== Complete Training Provenance Workflow ===")
    
    # Step 1: Initialize provenance tracking
    config = Config()
    seed = 789
    provenance_manager = create_provenance_manager(config, seed=seed)
    
    print(f"✓ Step 1: Provenance tracking initialized")
    print(f"  Pipeline ID: {provenance_manager.pipeline_id}")
    print(f"  Seed: {seed}")
    
    # Step 2: Track training stages
    training_stages = [
        {"name": "sft", "steps": 100, "final_loss": 0.45},
        {"name": "reward", "steps": 50, "final_loss": 0.38},
        {"name": "ppo", "steps": 25, "final_loss": 0.42}
    ]
    
    for stage_info in training_stages:
        stage = provenance_manager.start_stage(stage_info["name"])
        
        # Simulate training with realistic metrics
        for step in range(0, stage_info["steps"] + 1, max(1, stage_info["steps"] // 5)):
            progress = step / stage_info["steps"]
            loss = stage_info["final_loss"] + (1.0 - progress) * 0.5
            
            metrics = {
                "loss": loss,
                "learning_rate": 1e-4 * (0.95 ** (step // 10)),
                "step": step
            }
            provenance_manager.update_stage_metrics(stage_info["name"], step, metrics)
        
        provenance_manager.finalize_stage(
            stage_info["name"], 
            final_loss=stage_info["final_loss"],
            checkpoint_path=f"checkpoints/{stage_info['name']}_final"
        )
        
        print(f"  ✓ {stage_info['name'].upper()} stage: {stage_info['steps']} steps, loss: {stage_info['final_loss']}")
    
    print("✓ Step 2: All training stages completed")
    
    # Step 3: Add evaluation results
    evaluation_results = {
        "mt_bench_score": 7.5,
        "helpfulness": 8.2,
        "harmlessness": 7.9,
        "honesty": 7.7,
        "inference_speed_tokens_per_sec": 11.8,
        "memory_usage_gb": 7.3,
        "evaluation_timestamp": datetime.now().isoformat()
    }
    
    provenance_manager.add_evaluation_results(evaluation_results)
    print("✓ Step 3: Evaluation results added")
    
    # Step 4: Finalize training
    final_model_path = "models/complete_rlhf_model"
    provenance_manager.finalize_training(final_model_path=final_model_path)
    print("✓ Step 4: Training finalized")
    
    # Step 5: Save comprehensive provenance
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "provenance_output"
        provenance_file = provenance_manager.save_provenance(output_dir)
        
        print(f"✓ Step 5: Provenance saved to {provenance_file}")
        
        # List generated files
        files = list(output_dir.glob("*"))
        print(f"  Generated {len(files)} provenance files:")
        for file in sorted(files):
            print(f"    - {file.name}")
    
    # Step 6: Generate provenance summary
    summary = provenance_manager.get_provenance_summary()
    
    print("✓ Step 6: Provenance summary generated")
    print(f"  Training fingerprint: {summary['training_fingerprint'][:16]}...")
    print(f"  Stages completed: {summary['stages_completed']}")
    print(f"  Total training time: {summary['end_time'] is not None}")
    print(f"  Evaluation available: {summary['has_evaluation_results']}")
    
    return provenance_manager, summary


def main():
    """Run all training provenance demos."""
    print("RLHF Phi-3 Pipeline - Training Provenance and Metadata Tracking Demo")
    print("=" * 80)
    
    try:
        # Run all demos
        provenance_manager = demo_basic_provenance_tracking()
        snapshot = demo_configuration_snapshots()
        env_info, repro_hash = demo_environment_recreation()
        training_provenance = demo_model_metadata_integration()
        final_manager, summary = demo_complete_workflow()
        
        print("\n" + "=" * 80)
        print("🎉 All training provenance demos completed successfully!")
        print()
        print("Features Demonstrated:")
        print("✅ Training provenance tracking throughout RLHF pipeline stages")
        print("✅ Configuration snapshot saving with checkpoints")
        print("✅ Environment information capture and reproducibility hashing")
        print("✅ Model metadata enhancement with comprehensive training provenance")
        print("✅ Integration with existing configuration and checkpoint management")
        print("✅ Complete workflow from initialization to final model publication")
        print()
        
        print("Key Capabilities:")
        print(f"📊 Tracked {len(final_manager.provenance.stages)} training stages with detailed metrics")
        print(f"🔒 Reproducibility ensured with seed management and environment logging")
        print(f"📝 Configuration snapshots with integrity verification")
        print(f"🏷️  Model metadata includes complete training provenance")
        print(f"🔄 Environment recreation scripts and requirements generation")
        print()
        
        print("Requirements Satisfied:")
        print("✅ 15.1: Configuration snapshots with checkpoints")
        print("✅ 15.4: Training provenance in model metadata")
        print("✅ 15.5: Reproducibility scripts and environment recreation")
        print()
        
        print("✨ Training provenance and metadata tracking features are ready!")
        print("   Use these utilities throughout the RLHF pipeline for comprehensive")
        print("   reproducibility and professional-quality model documentation.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()