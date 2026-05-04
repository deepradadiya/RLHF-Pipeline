#!/usr/bin/env python3
"""
Configuration Manager Demo

This script demonstrates the key features of the RLHF Phi-3 Configuration Manager:
- Creating and validating configurations
- Stage-specific configuration subsets
- Serialization and loading
- Parameter bounds enforcement
- Colab-optimized configurations
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rlhf_phi3.config.config_manager import Config, load_default_config, load_colab_config


def main():
    print("🔧 RLHF Phi-3 Configuration Manager Demo")
    print("=" * 50)
    
    # 1. Create and validate default configuration
    print("\n1. Creating Default Configuration")
    config = Config()
    print(f"   Model: {config.model.name}")
    print(f"   Max Length: {config.model.max_length}")
    print(f"   LoRA Rank: {config.lora.r}")
    print(f"   SFT Epochs: {config.training.sft.epochs}")
    print(f"   Valid: {config.validate_config()}")
    
    # 2. Demonstrate stage-specific configurations
    print("\n2. Stage-Specific Configuration Subsets")
    
    sft_config = config.get_stage_config("sft")
    print(f"   SFT Config Sections: {list(sft_config.keys())}")
    print(f"   SFT Dataset: {sft_config['dataset']['name']}")
    
    reward_config = config.get_stage_config("reward")
    print(f"   Reward Dataset: {reward_config['dataset']['name']}")
    
    ppo_config = config.get_stage_config("ppo")
    print(f"   PPO Has Both Datasets: {'datasets' in ppo_config}")
    
    # 3. Demonstrate parameter bounds enforcement
    print("\n3. Parameter Bounds Enforcement")
    
    # Create invalid configuration
    invalid_config = Config()
    invalid_config.training.sft.learning_rate = 1e-7  # Too low
    invalid_config.model.max_length = -1  # Invalid
    invalid_config.lora.r = 0  # Invalid
    
    errors = invalid_config.get_validation_errors()
    print(f"   Validation Errors Found: {len(errors)}")
    for i, error in enumerate(errors[:3], 1):  # Show first 3 errors
        print(f"   {i}. {error}")
    
    # 4. Demonstrate serialization and loading
    print("\n4. Configuration Serialization")
    
    # Modify configuration
    config.training.sft.epochs = 5
    config.model.max_length = 1024
    
    # Save to file
    config_file = Path("temp_config.json")
    config.save_config(config_file)
    print(f"   Saved to: {config_file}")
    
    # Load from file
    loaded_config = Config.load_config(config_file)
    print(f"   Loaded epochs: {loaded_config.training.sft.epochs}")
    print(f"   Loaded max_length: {loaded_config.model.max_length}")
    print(f"   Values preserved: {loaded_config.lora.r == config.lora.r}")
    
    # Clean up
    config_file.unlink()
    
    # 5. Demonstrate Colab-optimized configuration
    print("\n5. Colab-Optimized Configuration")
    
    default_config = load_default_config()
    colab_config = load_colab_config()
    
    print(f"   Default Max Length: {default_config.model.max_length}")
    print(f"   Colab Max Length: {colab_config.model.max_length}")
    print(f"   Default LoRA Rank: {default_config.lora.r}")
    print(f"   Colab LoRA Rank: {colab_config.lora.r}")
    print(f"   Default SFT Batch Size: {default_config.training.sft.batch_size}")
    print(f"   Colab SFT Batch Size: {colab_config.training.sft.batch_size}")
    
    # 6. Demonstrate configuration updates
    print("\n6. Configuration Updates")
    
    original_config = Config()
    updates = {
        "model": {"name": "custom-phi3-model"},
        "training": {"sft": {"epochs": 10}},
        "lora": {"r": 32}
    }
    
    updated_config = original_config.update_from_dict(updates)
    
    print(f"   Original Model: {original_config.model.name}")
    print(f"   Updated Model: {updated_config.model.name}")
    print(f"   Original Epochs: {original_config.training.sft.epochs}")
    print(f"   Updated Epochs: {updated_config.training.sft.epochs}")
    print(f"   Original Unchanged: {original_config.model.name != updated_config.model.name}")
    
    print("\n✅ Configuration Manager Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Comprehensive validation with parameter bounds")
    print("  ✓ Stage-specific configuration subsets")
    print("  ✓ Serialization and reproducibility")
    print("  ✓ Colab-optimized configurations")
    print("  ✓ Immutable updates and deep copying")


if __name__ == "__main__":
    main()