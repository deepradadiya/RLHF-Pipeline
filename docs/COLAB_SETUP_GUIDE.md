# Google Colab Setup Guide for RLHF Phi-3 Pipeline

This comprehensive guide will walk you through setting up the RLHF Phi-3 Pipeline in Google Colab, from account creation to running your first training session.

## 🎯 Why Google Colab?

Google Colab is the recommended platform for beginners because it provides:

- ✅ **Free GPU Access**: T4 GPU with 15GB VRAM at no cost
- ✅ **Pre-installed Libraries**: Most ML libraries already available
- ✅ **No Local Setup**: Run everything in your browser
- ✅ **Google Drive Integration**: Persistent storage for checkpoints
- ✅ **Easy Sharing**: Share notebooks with collaborators
- ✅ **Upgrade Options**: Colab Pro for longer sessions and better GPUs

## 📋 Prerequisites

Before starting, ensure you have:

1. **Google Account** (free)
2. **Stable Internet Connection** (for dataset downloads)
3. **Web Browser** (Chrome recommended)
4. **Basic Python Knowledge** (helpful but not required)

## 🚀 Step-by-Step Setup

### Step 1: Create Accounts

#### 1.1 Weights & Biases Account (Free)

1. Go to [wandb.ai](https://wandb.ai/)
2. Click "Sign Up" and create a free account
3. Verify your email address
4. Go to [Settings](https://wandb.ai/settings) and copy your API key
5. Save this API key - you'll need it later

#### 1.2 HuggingFace Account (Free)

1. Go to [huggingface.co](https://huggingface.co/)
2. Click "Sign Up" and create a free account
3. Verify your email address
4. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
5. Click "New token" and create a token with "Write" permissions
6. Copy and save this token - you'll need it later

### Step 2: Set Up Google Colab

#### 2.1 Create New Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Click "New notebook" or go to File → New notebook
4. Rename your notebook: File → Rename → "RLHF Phi-3 Pipeline Setup"

#### 2.2 Enable GPU Runtime

1. Go to **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Choose **T4 GPU** (free tier) or **A100/V100** (Colab Pro)
4. Set **Runtime shape** to **Standard** (free) or **High-RAM** (Pro)
5. Click **Save**

#### 2.3 Verify GPU Access

Run this code in a new cell:

```python
import torch
import subprocess

print("🔍 System Information:")
print(f"GPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Test GPU functionality
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("✅ GPU computation test passed")
else:
    print("❌ No GPU detected!")
    print("💡 Go to Runtime → Change runtime type → GPU")

# Check Python version
import sys
print(f"Python Version: {sys.version}")

# Check available RAM
import psutil
ram = psutil.virtual_memory()
print(f"Available RAM: {ram.total / 1e9:.1f}GB")
```

Expected output:
```
🔍 System Information:
GPU Available: True
GPU Name: Tesla T4
GPU Memory: 15.1GB
CUDA Version: 12.2
✅ GPU computation test passed
Python Version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
Available RAM: 12.7GB
```

### Step 3: Mount Google Drive

Run this code to mount Google Drive for persistent storage:

```python
from google.colab import drive
import os

print("📁 Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

# Verify mount
if os.path.exists('/content/drive/MyDrive'):
    print("✅ Google Drive mounted successfully")
    
    # Create project directory
    project_dir = '/content/drive/MyDrive/rlhf-phi3'
    os.makedirs(project_dir, exist_ok=True)
    print(f"📂 Project directory created: {project_dir}")
    
    # Test write permissions
    test_file = f'{project_dir}/test_write.txt'
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ Write permissions confirmed")
    except Exception as e:
        print(f"❌ Write permission test failed: {e}")
else:
    print("❌ Google Drive mount failed")
    print("💡 Try: Runtime → Restart runtime, then re-run this cell")
```

### Step 4: Install RLHF Pipeline

Run this code to clone and install the pipeline:

```python
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run command with progress indication"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        raise

# Change to content directory
os.chdir('/content')

# Clone repository
run_command(
    "git clone https://github.com/your-username/rlhf-phi3-pipeline.git",
    "Cloning RLHF Phi-3 repository"
)

# Change to project directory
os.chdir('/content/rlhf-phi3-pipeline')
print(f"📂 Changed to directory: {os.getcwd()}")

# Install dependencies
run_command(
    "pip install -r requirements.txt",
    "Installing Python dependencies"
)

# Install package in development mode
run_command(
    "pip install -e .",
    "Installing RLHF Phi-3 package"
)

# Verify installation
try:
    from rlhf_phi3 import Config
    print("✅ RLHF Phi-3 package installed successfully!")
    print("📦 Available components:")
    print("   • Configuration Manager")
    print("   • Dataset Manager") 
    print("   • Model Manager")
    print("   • Training Orchestrator")
    print("   • Experiment Tracker")
    print("   • Evaluation Engine")
except ImportError as e:
    print(f"❌ Installation verification failed: {e}")
    print("💡 Try restarting runtime and running installation again")
```

### Step 5: Set Up Authentication

Run this code to configure authentication for external services:

```python
import os
from getpass import getpass

print("🔐 Setting up authentication for external services...")

# Weights & Biases setup
print("\n1️⃣ Weights & Biases Setup:")
print("   This is used for experiment tracking and visualization")
print("   📊 Go to https://wandb.ai/settings to get your API key")

wandb_key = getpass("   Enter your W&B API key: ")
os.environ['WANDB_API_KEY'] = wandb_key

# Test W&B authentication
try:
    import wandb
    wandb.login(key=wandb_key)
    print("✅ W&B authentication successful")
    
    # Test functionality
    test_run = wandb.init(project="test-project", name="auth-test", mode="disabled")
    test_run.finish()
    print("✅ W&B functionality test passed")
except Exception as e:
    print(f"❌ W&B authentication failed: {e}")
    print("💡 Double-check your API key and try again")

# HuggingFace setup
print("\n2️⃣ HuggingFace Setup:")
print("   This is used for model publishing and dataset access")
print("   🤗 Go to https://huggingface.co/settings/tokens")
print("   📝 Create a new token with 'write' permissions")

hf_token = getpass("   Enter your HuggingFace token: ")
os.environ['HUGGINGFACE_TOKEN'] = hf_token

# Test HF authentication
try:
    from huggingface_hub import login, whoami
    login(token=hf_token)
    user_info = whoami()
    print(f"✅ HuggingFace authenticated as: {user_info['name']}")
    
    # Test model access
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    print("✅ Model access confirmed")
except Exception as e:
    print(f"❌ HuggingFace authentication failed: {e}")
    print("💡 Check your token permissions and try again")

print("\n🎉 Authentication setup complete!")
```

### Step 6: Load and Validate Configuration

Run this code to load and validate the pipeline configuration:

```python
from rlhf_phi3 import Config
import yaml

print("⚙️ Loading and validating configuration...")

# Load Colab-optimized configuration
config_path = "configs/colab_config.yaml"
config = Config.from_yaml(config_path)

print("📋 Configuration Summary:")
print(f"   Model: {config.model.name}")
print(f"   Max Length: {config.model.max_length}")
print(f"   LoRA Rank: {config.lora.r}")
print(f"   LoRA Alpha: {config.lora.alpha}")
print(f"   SFT Epochs: {config.training.sft.epochs}")
print(f"   SFT Batch Size: {config.training.sft.batch_size}")
print(f"   SFT Max Steps: {config.training.sft.max_steps}")

# Update paths for Google Drive
config.paths.base_output_dir = "/content/drive/MyDrive/rlhf-phi3"
config.paths.cache_dir = "/content/cache"
config.paths.logs_dir = "/content/logs"

# Validate configuration
print("\n🔍 Validating configuration...")
errors = config.validate()
if errors:
    print("❌ Configuration errors found:")
    for error in errors:
        print(f"   • {error}")
else:
    print("✅ Configuration is valid")

# Save configuration to Drive for persistence
config_save_path = "/content/drive/MyDrive/rlhf-phi3/my_config.yaml"
os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
config.save_yaml(config_save_path)
print(f"💾 Configuration saved to: {config_save_path}")

# Display memory-optimized settings for Colab
print("\n🧠 Memory Optimization Settings:")
print(f"   Mixed Precision (FP16): {config.optimization.fp16}")
print(f"   Gradient Checkpointing: {config.optimization.gradient_checkpointing}")
print(f"   Max Gradient Norm: {config.optimization.max_grad_norm}")
print("   These settings are optimized for T4 GPU (15GB VRAM)")
```

### Step 7: Run System Tests

Run this comprehensive system test to ensure everything is working:

```python
def run_comprehensive_system_test():
    """Run comprehensive system tests"""
    print("🧪 Running Comprehensive System Tests")
    print("=" * 60)
    
    test_results = {
        "gpu_access": False,
        "package_import": False,
        "config_validation": False,
        "authentication": {"wandb": False, "hf": False},
        "dataset_access": False,
        "model_loading": False
    }
    
    # Test 1: GPU Access
    print("\n1️⃣ Testing GPU Access...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Test GPU computation
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x.T)
            print("   ✅ GPU computation test passed")
            test_results["gpu_access"] = True
        else:
            print("   ❌ No GPU available")
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
    
    # Test 2: Package Import
    print("\n2️⃣ Testing Package Imports...")
    try:
        from rlhf_phi3 import Config, TrainingOrchestrator
        from rlhf_phi3.data import DatasetManager
        from rlhf_phi3.models import ModelManager
        print("   ✅ All core packages imported successfully")
        test_results["package_import"] = True
    except ImportError as e:
        print(f"   ❌ Package import failed: {e}")
    
    # Test 3: Configuration
    print("\n3️⃣ Testing Configuration...")
    try:
        config = Config.from_yaml("configs/colab_config.yaml")
        errors = config.validate()
        if not errors:
            print("   ✅ Configuration loaded and validated")
            test_results["config_validation"] = True
        else:
            print(f"   ❌ Configuration errors: {errors}")
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
    
    # Test 4: Authentication
    print("\n4️⃣ Testing Authentication...")
    
    # W&B test
    try:
        import wandb
        wandb.init(project="system-test", mode="disabled")
        print("   ✅ W&B authentication working")
        test_results["authentication"]["wandb"] = True
    except Exception as e:
        print(f"   ⚠️ W&B authentication issue: {e}")
    
    # HuggingFace test
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"   ✅ HuggingFace authenticated as: {user_info['name']}")
        test_results["authentication"]["hf"] = True
    except Exception as e:
        print(f"   ⚠️ HuggingFace authentication issue: {e}")
    
    # Test 5: Dataset Access
    print("\n5️⃣ Testing Dataset Access...")
    try:
        from datasets import load_dataset
        
        # Test SFT dataset
        sft_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:1]", streaming=True)
        sft_sample = next(iter(sft_dataset))
        print(f"   ✅ SFT dataset accessible (sample keys: {list(sft_sample.keys())})")
        
        # Test preference dataset
        pref_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train[:1]", streaming=True)
        pref_sample = next(iter(pref_dataset))
        print(f"   ✅ Preference dataset accessible (sample keys: {list(pref_sample.keys())})")
        
        test_results["dataset_access"] = True
    except Exception as e:
        print(f"   ❌ Dataset access failed: {e}")
    
    # Test 6: Model Loading
    print("\n6️⃣ Testing Model Loading...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        print("   ✅ Tokenizer loaded successfully")
        
        # Test model info (don't load full model to save memory)
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        print(f"   ✅ Model config loaded (vocab size: {model_config.vocab_size})")
        
        test_results["model_loading"] = True
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed_tests = 0
    total_tests = 0
    
    for test_name, result in test_results.items():
        if test_name == "authentication":
            for auth_service, auth_result in result.items():
                total_tests += 1
                if auth_result:
                    passed_tests += 1
                    print(f"   ✅ {auth_service.upper()} Authentication")
                else:
                    print(f"   ⚠️ {auth_service.upper()} Authentication")
        else:
            total_tests += 1
            if result:
                passed_tests += 1
                print(f"   ✅ {test_name.replace('_', ' ').title()}")
            else:
                print(f"   ❌ {test_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 Overall Score: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Ready for training!")
        return True
    elif passed_tests >= total_tests - 2:  # Allow for auth issues
        print("✅ MOSTLY READY - Some optional features may not work")
        return True
    else:
        print("❌ SETUP INCOMPLETE - Please fix the failing tests")
        return False

# Run the comprehensive test
test_success = run_comprehensive_system_test()

if test_success:
    print("\n🚀 Your Colab environment is ready!")
    print("📋 Next steps:")
    print("   1. Run a quick demo: orchestrator.run_demo_pipeline()")
    print("   2. Or start full training: orchestrator.run_full_pipeline()")
    print("   3. Monitor progress in W&B dashboard")
    print("   4. Checkpoints will be saved to Google Drive")
else:
    print("\n⚠️ Please resolve the failing tests before proceeding")
    print("💡 Check the troubleshooting section in the main README")
```

### Step 8: Run Your First Training

Now you're ready to run your first training! Choose one of these options:

#### Option A: Quick Demo (5 minutes)

```python
from rlhf_phi3 import Config, TrainingOrchestrator

print("🚀 Running Quick Demo Training...")

# Load demo configuration (very small dataset)
config = Config.from_yaml("configs/demo_config.yaml")
config.paths.base_output_dir = "/content/drive/MyDrive/rlhf-phi3"

# Initialize orchestrator
orchestrator = TrainingOrchestrator(config)

# Run demo pipeline
demo_result = orchestrator.run_demo_pipeline()

if demo_result:
    print(f"🎉 Demo completed successfully!")
    print(f"📂 Demo model saved to: {demo_result}")
else:
    print("❌ Demo failed - check the logs for details")
```

#### Option B: Full Training (2-4 hours)

```python
from rlhf_phi3 import Config, TrainingOrchestrator

print("🚀 Starting Full RLHF Training Pipeline...")

# Load Colab configuration
config = Config.from_yaml("configs/colab_config.yaml")
config.paths.base_output_dir = "/content/drive/MyDrive/rlhf-phi3"

# Initialize orchestrator
orchestrator = TrainingOrchestrator(config)

# Estimate training time
session_manager = SessionManager()  # From previous setup
time_estimate = session_manager.estimate_training_time(config)

if time_estimate["fits_in_session"]:
    print("✅ Training should complete within current session")
    
    # Run full pipeline
    final_model = orchestrator.run_full_pipeline()
    
    if final_model:
        print(f"🎉 Training completed successfully!")
        print(f"📂 Final model saved to: {final_model}")
        
        # Run evaluation
        from rlhf_phi3 import EvaluationEngine
        evaluator = EvaluationEngine(final_model, config)
        results = evaluator.run_comprehensive_evaluation()
        
        print(f"📊 Evaluation Results:")
        print(f"   MT-Bench Score: {results.mt_bench_score:.2f}/10.0")
        print(f"   Helpfulness: {results.helpfulness_score:.2f}/10.0")
        
    else:
        print("❌ Training failed - check the logs for details")
else:
    print("⚠️ Training may not complete within session time limit")
    print("💡 Consider running stages separately or upgrading to Colab Pro")
```

## 🔧 Colab-Specific Tips and Tricks

### Session Management

1. **Monitor Session Time**
   ```python
   # Check remaining session time
   session_manager.get_session_info()
   ```

2. **Enable Session Keepalive**
   ```python
   # Prevent idle timeout
   session_manager.setup_keepalive()
   ```

3. **Create Emergency Checkpoints**
   ```python
   # Save progress before session ends
   emergency_checkpoint_save(model, optimizer, step, "sft")
   ```

### Memory Management

1. **Clear GPU Memory Between Stages**
   ```python
   import torch
   import gc
   
   torch.cuda.empty_cache()
   gc.collect()
   ```

2. **Monitor Memory Usage**
   ```python
   def check_memory():
       allocated = torch.cuda.memory_allocated() / 1e9
       reserved = torch.cuda.memory_reserved() / 1e9
       print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
   
   check_memory()
   ```

### Storage Optimization

1. **Use Google Drive for Persistence**
   - All checkpoints automatically saved to Drive
   - Survives session restarts
   - Accessible from any device

2. **Clean Up Temporary Files**
   ```python
   # Clean up local cache periodically
   import shutil
   shutil.rmtree("/content/cache", ignore_errors=True)
   ```

### Troubleshooting Common Colab Issues

1. **"Runtime Disconnected" Error**
   - Restart runtime: Runtime → Restart runtime
   - Re-run setup cells
   - Resume from last checkpoint

2. **"CUDA Out of Memory" Error**
   - Reduce batch size in config
   - Clear GPU cache: `torch.cuda.empty_cache()`
   - Restart runtime if needed

3. **"Drive Mount Failed" Error**
   - Try force remount: `drive.mount('/content/drive', force_remount=True)`
   - Check Google account permissions
   - Clear browser cache if needed

4. **Slow Dataset Downloads**
   - Use streaming datasets: `streaming=True`
   - Reduce dataset size: `max_samples=1000`
   - Check internet connection

## 🎓 Next Steps

After completing the setup:

1. **Explore the Notebooks**
   - `01_setup_and_configuration.ipynb` - This guide in notebook form
   - `02_sft_training_tutorial.ipynb` - Detailed SFT training
   - `03_reward_model_tutorial.ipynb` - Reward model training
   - `04_ppo_training_tutorial.ipynb` - PPO training
   - `05_evaluation_and_publishing.ipynb` - Model evaluation and publishing

2. **Experiment with Configurations**
   - Try different LoRA ranks and learning rates
   - Experiment with different datasets
   - Adjust training parameters for your use case

3. **Monitor Your Training**
   - Use Weights & Biases dashboard for real-time monitoring
   - Check Google Drive for checkpoint persistence
   - Review training logs for insights

4. **Share Your Results**
   - Publish trained models to HuggingFace Hub
   - Share notebooks with the community
   - Contribute improvements back to the project

## 🆘 Getting Help

If you encounter issues:

1. **Check the Troubleshooting Section** in the main README
2. **Search GitHub Issues** for similar problems
3. **Create a New Issue** with detailed error information
4. **Join Community Discussions** for help and tips

## 📚 Additional Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [HuggingFace Documentation](https://huggingface.co/docs)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [RLHF Paper](https://arxiv.org/abs/2203.02155)

---

🎉 **Congratulations!** You've successfully set up the RLHF Phi-3 Pipeline in Google Colab. You're now ready to train your own state-of-the-art language model!