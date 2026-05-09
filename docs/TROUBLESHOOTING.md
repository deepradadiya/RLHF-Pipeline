# RLHF Phi-3 Pipeline Troubleshooting Guide

This comprehensive troubleshooting guide covers common issues, their causes, and step-by-step solutions for the RLHF Phi-3 Pipeline.

## 🚨 Critical Issues (Immediate Action Required)

### GPU Memory Issues

**Problem**: `CUDA out of memory` errors during training

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training stops unexpectedly
- System becomes unresponsive
- Memory usage spikes to 100%

**Root Causes:**
- Batch size too large for available GPU memory
- Model parameters exceed GPU capacity
- Memory fragmentation from previous operations
- Multiple models loaded simultaneously

**Immediate Solutions:**

1. **Emergency Memory Clear**
   ```python
   import torch
   import gc
   
   # Clear GPU cache
   torch.cuda.empty_cache()
   
   # Force garbage collection
   gc.collect()
   
   # Check memory status
   if torch.cuda.is_available():
       allocated = torch.cuda.memory_allocated() / 1e9
       reserved = torch.cuda.memory_reserved() / 1e9
       print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
   ```

2. **Reduce Batch Size Immediately**
   ```python
   # Emergency configuration adjustment
   config.training.sft.batch_size = 1
   config.training.reward.batch_size = 1
   config.training.ppo.batch_size = 1
   
   # Compensate with gradient accumulation
   config.training.sft.gradient_accumulation_steps = 16
   config.training.reward.gradient_accumulation_steps = 32
   config.training.ppo.gradient_accumulation_steps = 64
   ```

3. **Enable All Memory Optimizations**
   ```python
   # Maximum memory efficiency settings
   config.optimization.fp16 = True
   config.optimization.gradient_checkpointing = True
   config.model.max_length = 512  # Reduce from 1024/2048
   config.lora.r = 4  # Reduce LoRA rank
   ```

**Memory Usage Guidelines by GPU:**

| GPU Model | VRAM | Recommended Settings |
|-----------|------|---------------------|
| **RTX 3060** | 12GB | batch_size=1, max_length=1024, lora_r=8 |
| **RTX 3070** | 8GB | batch_size=1, max_length=512, lora_r=4 |
| **RTX 3080** | 10GB | batch_size=1, max_length=1024, lora_r=8 |
| **RTX 3090** | 24GB | batch_size=2, max_length=2048, lora_r=16 |
| **RTX 4090** | 24GB | batch_size=4, max_length=2048, lora_r=16 |
| **Tesla T4** | 15GB | batch_size=1, max_length=1024, lora_r=8 |
| **Tesla V100** | 32GB | batch_size=4, max_length=2048, lora_r=16 |
| **A100 40GB** | 40GB | batch_size=8, max_length=4096, lora_r=32 |

### Session Timeout Issues (Google Colab)

**Problem**: Session times out after 12 hours (24 hours with Pro)

**Prevention:**
```python
import time
import datetime
from IPython.display import display, Javascript

class SessionManager:
    def __init__(self):
        self.start_time = time.time()
        self.session_limit_hours = 12  # 24 for Colab Pro
        
    def get_session_info(self):
        """Get current session duration and remaining time"""
        current_time = time.time()
        elapsed_hours = (current_time - self.start_time) / 3600
        remaining_hours = self.session_limit_hours - elapsed_hours
        
        print(f"⏱️ Session Status:")
        print(f"   Elapsed: {elapsed_hours:.1f} hours")
        print(f"   Remaining: {remaining_hours:.1f} hours")
        
        if remaining_hours < 2:
            print("🚨 CRITICAL: Less than 2 hours remaining!")
            return "critical"
        elif remaining_hours < 4:
            print("⚠️ WARNING: Less than 4 hours remaining")
            return "warning"
        else:
            print("✅ Session time OK")
            return "ok"
            
    def setup_keepalive(self):
        """Prevent idle timeout"""
        display(Javascript('''
            function ClickConnect(){
                console.log("Keeping session alive...");
                var connectButton = document.querySelector("colab-connect-button");
                if (connectButton) {
                    connectButton.click();
                }
            }
            setInterval(ClickConnect, 60000); // Every minute
        '''))
        print("🔄 Session keepalive activated")

# Initialize and use
session_manager = SessionManager()
session_manager.get_session_info()
session_manager.setup_keepalive()
```

**Recovery:**
```python
def resume_training_from_checkpoint():
    """Resume training from the latest checkpoint"""
    from rlhf_phi3 import TrainingOrchestrator, CheckpointManager
    
    print("🔄 Attempting to resume training...")
    
    # Load configuration from Drive
    config_path = "/content/drive/MyDrive/rlhf-phi3/my_config.yaml"
    if os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config.from_yaml("configs/colab_config.yaml")
    
    # Find latest checkpoint
    checkpoint_manager = CheckpointManager(config.paths.base_output_dir)
    
    stages = ["ppo", "reward", "sft"]  # Check in reverse order
    for stage in stages:
        checkpoints = checkpoint_manager.list_checkpoints(stage)
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"📂 Found checkpoint: {latest_checkpoint}")
            
            orchestrator = TrainingOrchestrator(config)
            return orchestrator.resume_from_stage(stage, latest_checkpoint)
    
    print("❌ No checkpoints found")
    return None
```

## 🔐 Authentication Issues

### Google Drive Authentication

**Problem**: Cannot mount Google Drive or access files

**Solutions:**

1. **Force Remount**
   ```python
   from google.colab import drive, auth
   
   # Try force remount
   drive.mount('/content/drive', force_remount=True)
   
   # If that fails, try manual authentication
   auth.authenticate_user()
   drive.mount('/content/drive', force_remount=True)
   ```

2. **Check Permissions**
   ```python
   import os
   
   def check_drive_permissions():
       drive_path = '/content/drive/MyDrive'
       
       if not os.path.exists(drive_path):
           print("❌ Google Drive not mounted")
           return False
       
       # Test directory creation
       test_dir = f"{drive_path}/test_permissions"
       try:
           os.makedirs(test_dir, exist_ok=True)
           os.rmdir(test_dir)
           print("✅ Directory permissions OK")
       except Exception as e:
           print(f"❌ Directory permission failed: {e}")
           return False
       
       # Test file operations
       test_file = f"{drive_path}/test_file.txt"
       try:
           with open(test_file, 'w') as f:
               f.write("test")
           with open(test_file, 'r') as f:
               content = f.read()
           os.remove(test_file)
           print("✅ File permissions OK")
           return True
       except Exception as e:
           print(f"❌ File permission failed: {e}")
           return False
   
   check_drive_permissions()
   ```

### Weights & Biases Authentication

**Problem**: Cannot log into W&B or track experiments

**Solutions:**

1. **Manual API Key Setup**
   ```python
   import wandb
   import os
   from getpass import getpass
   
   def fix_wandb_authentication():
       print("🔑 Setting up W&B authentication...")
       
       # Get API key
       print("1. Go to https://wandb.ai/settings")
       print("2. Copy your API key")
       api_key = getpass("Enter your W&B API key: ")
       
       try:
           wandb.login(key=api_key)
           os.environ['WANDB_API_KEY'] = api_key
           print("✅ W&B authentication successful")
           
           # Test functionality
           test_run = wandb.init(project="test", mode="disabled")
           test_run.finish()
           print("✅ W&B functionality confirmed")
           return True
       except Exception as e:
           print(f"❌ W&B authentication failed: {e}")
           return False
   
   fix_wandb_authentication()
   ```

2. **Check Project Access**
   ```python
   def check_wandb_project_access(project_name):
       try:
           run = wandb.init(project=project_name, mode="disabled")
           run.finish()
           print(f"✅ Access to project '{project_name}' confirmed")
           return True
       except Exception as e:
           print(f"❌ Cannot access project '{project_name}': {e}")
           print("💡 Check project name and permissions")
           return False
   
   check_wandb_project_access("rlhf-phi3-pipeline")
   ```

### HuggingFace Authentication

**Problem**: Cannot access models or publish to Hub

**Solutions:**

1. **Token Setup**
   ```python
   from huggingface_hub import login, logout, whoami
   from getpass import getpass
   
   def fix_huggingface_authentication():
       print("🤗 Setting up HuggingFace authentication...")
       
       # Get token
       print("1. Go to https://huggingface.co/settings/tokens")
       print("2. Create token with 'write' permissions")
       token = getpass("Enter your HuggingFace token: ")
       
       try:
           login(token=token)
           user_info = whoami()
           print(f"✅ Authenticated as: {user_info['name']}")
           return True
       except Exception as e:
           print(f"❌ Authentication failed: {e}")
           return False
   
   fix_huggingface_authentication()
   ```

2. **Test Model Access**
   ```python
   def test_model_access():
       try:
           from transformers import AutoTokenizer
           tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
           print("✅ Model access confirmed")
           return True
       except Exception as e:
           print(f"❌ Model access failed: {e}")
           return False
   
   test_model_access()
   ```

## 📊 Dataset Loading Issues

### Network and Download Problems

**Problem**: Dataset downloads fail or are very slow

**Solutions:**

1. **Test Network Connectivity**
   ```python
   import requests
   
   def test_network_connectivity():
       print("🌐 Testing network connectivity...")
       
       try:
           # Test basic internet
           response = requests.get("https://www.google.com", timeout=10)
           print("✅ Basic internet OK")
       except:
           print("❌ No internet connectivity")
           return False
       
       try:
           # Test HuggingFace Hub
           response = requests.get("https://huggingface.co", timeout=10)
           print("✅ HuggingFace Hub accessible")
           return True
       except:
           print("❌ HuggingFace Hub not accessible")
           return False
   
   test_network_connectivity()
   ```

2. **Robust Dataset Loading**
   ```python
   from datasets import load_dataset
   import time
   
   def robust_dataset_loading(dataset_name, split="train", max_retries=3):
       print(f"📥 Loading dataset: {dataset_name}")
       
       for attempt in range(max_retries):
           try:
               print(f"   Attempt {attempt + 1}/{max_retries}")
               
               dataset = load_dataset(
                   dataset_name,
                   split=split,
                   streaming=True,
                   trust_remote_code=True
               )
               
               # Test iteration
               sample = next(iter(dataset))
               print(f"✅ Dataset loaded successfully")
               return dataset
               
           except Exception as e:
               print(f"❌ Attempt {attempt + 1} failed: {e}")
               if attempt < max_retries - 1:
                   wait_time = 2 ** attempt
                   print(f"   Waiting {wait_time} seconds...")
                   time.sleep(wait_time)
       
       print("❌ All attempts failed")
       return None
   
   # Load datasets with retry
   sft_dataset = robust_dataset_loading("HuggingFaceH4/ultrachat_200k", "train_sft")
   ```

3. **Alternative Datasets**
   ```python
   def try_alternative_datasets(dataset_type="sft"):
       alternatives = {
           "sft": [
               ("HuggingFaceH4/ultrachat_200k", "train_sft"),
               ("tatsu-lab/alpaca", "train"),
               ("Open-Orca/OpenOrca", "train")
           ],
           "preference": [
               ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs"),
               ("Anthropic/hh-rlhf", "train"),
               ("lvwerra/stack-exchange-paired", "train")
           ]
       }
       
       for dataset_name, split in alternatives[dataset_type]:
           try:
               dataset = load_dataset(dataset_name, split=split, streaming=True)
               sample = next(iter(dataset))
               print(f"✅ Successfully loaded {dataset_name}")
               return dataset, dataset_name
           except Exception as e:
               print(f"❌ Failed to load {dataset_name}: {e}")
       
       return None, None
   
   # Try alternatives
   sft_dataset, sft_name = try_alternative_datasets("sft")
   ```

### Dataset Cache Management

**Problem**: Corrupted cache or insufficient disk space

**Solutions:**

1. **Clear Dataset Cache**
   ```python
   import shutil
   import os
   from datasets import config
   
   def clear_dataset_cache():
       cache_dir = config.HF_DATASETS_CACHE
       print(f"📂 Cache directory: {cache_dir}")
       
       if os.path.exists(cache_dir):
           cache_size = sum(
               os.path.getsize(os.path.join(dirpath, filename))
               for dirpath, dirnames, filenames in os.walk(cache_dir)
               for filename in filenames
           ) / (1024**3)
           
           print(f"💾 Cache size: {cache_size:.2f} GB")
           
           response = input("Clear cache? (y/n): ")
           if response.lower() == 'y':
               shutil.rmtree(cache_dir, ignore_errors=True)
               print("✅ Cache cleared")
           else:
               print("Cache kept")
       else:
           print("No cache found")
   
   clear_dataset_cache()
   ```

2. **Check Disk Space**
   ```python
   import shutil
   
   def check_disk_space():
       total, used, free = shutil.disk_usage("/content")
       print(f"💿 Disk Space:")
       print(f"   Total: {total / 1e9:.1f}GB")
       print(f"   Used: {used / 1e9:.1f}GB")
       print(f"   Free: {free / 1e9:.1f}GB")
       
       if free < 10e9:  # Less than 10GB
           print("⚠️ Low disk space - consider clearing cache")
           return False
       else:
           print("✅ Sufficient disk space")
           return True
   
   check_disk_space()
   ```

## 🔄 Training Convergence Issues

### Loss Not Decreasing

**Problem**: Training loss plateaus or increases

**Diagnosis:**
```python
def analyze_training_metrics(metrics_history):
    """Analyze training metrics for convergence issues"""
    if not metrics_history:
        print("No metrics to analyze")
        return
    
    recent_losses = metrics_history[-20:]  # Last 20 steps
    
    if len(recent_losses) > 10:
        import numpy as np
        loss_variance = np.var(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        print(f"📊 Loss Analysis:")
        print(f"   Recent loss variance: {loss_variance:.4f}")
        print(f"   Loss trend: {loss_trend:.4f}")
        
        if loss_variance > 0.1:
            print("⚠️ High loss variance - reduce learning rate")
        elif loss_trend > 0.01:
            print("⚠️ Loss increasing - reduce learning rate significantly")
        elif abs(loss_trend) < 0.001:
            print("⚠️ Loss plateaued - increase learning rate or check data")
        else:
            print("✅ Loss decreasing normally")

# Example usage:
# analyze_training_metrics(your_loss_history)
```

**Solutions:**

1. **Adjust Learning Rate**
   ```python
   # If loss is oscillating or increasing
   config.training.sft.learning_rate *= 0.5  # Reduce by 50%
   
   # If loss plateaued at high value
   config.training.sft.learning_rate *= 1.5  # Increase by 50%
   
   # If loss plateaued at low value
   config.training.sft.learning_rate *= 0.1  # Reduce significantly
   ```

2. **Adjust Batch Size and Accumulation**
   ```python
   # Increase effective batch size for stability
   config.training.sft.gradient_accumulation_steps *= 2
   
   # Or reduce batch size and increase accumulation
   config.training.sft.batch_size = max(1, config.training.sft.batch_size // 2)
   config.training.sft.gradient_accumulation_steps *= 2
   ```

3. **Modify LoRA Parameters**
   ```python
   # Increase capacity
   config.lora.r = min(64, config.lora.r * 2)
   config.lora.alpha = config.lora.r * 2
   
   # Or reduce overfitting
   config.lora.dropout = min(0.3, config.lora.dropout + 0.1)
   ```

### Gradient Issues

**Problem**: Exploding or vanishing gradients

**Detection:**
```python
def monitor_gradients(model):
    """Monitor gradient norms during training"""
    total_norm = 0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    print(f"📊 Gradient Analysis:")
    print(f"   Total gradient norm: {total_norm:.4f}")
    print(f"   Parameters with gradients: {param_count}")
    
    if total_norm > 10:
        print("⚠️ Exploding gradients detected")
        return "exploding"
    elif total_norm < 0.001:
        print("⚠️ Vanishing gradients detected")
        return "vanishing"
    else:
        print("✅ Gradient norms normal")
        return "normal"

# Use during training loop
# gradient_status = monitor_gradients(model)
```

**Solutions:**

1. **For Exploding Gradients**
   ```python
   # Reduce gradient clipping threshold
   config.optimization.max_grad_norm = 0.5
   
   # Reduce learning rate
   config.training.sft.learning_rate *= 0.1
   
   # Enable gradient checkpointing
   config.optimization.gradient_checkpointing = True
   ```

2. **For Vanishing Gradients**
   ```python
   # Increase learning rate
   config.training.sft.learning_rate *= 2
   
   # Reduce LoRA dropout
   config.lora.dropout = max(0.05, config.lora.dropout - 0.05)
   
   # Check LoRA scaling
   config.lora.alpha = config.lora.r * 2
   ```

## 🆘 Emergency Procedures

### Session About to Timeout

```python
def emergency_session_save(model=None, optimizer=None, step=0, stage="unknown"):
    """Emergency save when session is about to timeout"""
    import datetime
    
    print("🚨 EMERGENCY SESSION SAVE")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    emergency_dir = f"/content/drive/MyDrive/rlhf-phi3/emergency_saves"
    os.makedirs(emergency_dir, exist_ok=True)
    
    save_path = f"{emergency_dir}/emergency_{stage}_{timestamp}"
    os.makedirs(save_path, exist_ok=True)
    
    try:
        if model is not None:
            model.save_pretrained(f"{save_path}/model")
            print(f"✅ Model saved")
        
        if optimizer is not None:
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'stage': stage,
                'timestamp': timestamp
            }, f"{save_path}/optimizer.pt")
            print(f"✅ Optimizer saved")
        
        if 'config' in globals():
            config.save_yaml(f"{save_path}/config.yaml")
            print(f"✅ Config saved")
        
        print(f"🎉 Emergency save completed: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"❌ Emergency save failed: {e}")
        return None

# Use when session is about to timeout
# emergency_path = emergency_session_save(current_model, current_optimizer, current_step, "sft")
```

### Complete System Reset

```python
def emergency_system_reset():
    """Complete system reset when everything fails"""
    print("🚨 EMERGENCY SYSTEM RESET")
    
    response = input("This will clear all memory and restart runtime. Continue? (y/n): ")
    if response.lower() != 'y':
        return
    
    try:
        # Clear GPU memory
        import torch
        torch.cuda.empty_cache()
        
        # Clear variables
        import gc
        gc.collect()
        
        # Restart runtime (Colab specific)
        import os
        os.kill(os.getpid(), 9)
        
    except Exception as e:
        print(f"❌ Reset failed: {e}")

# Use only as last resort
# emergency_system_reset()
```

## 🔍 Diagnostic Tools

### Comprehensive System Check

```python
def run_full_diagnostics():
    """Run comprehensive system diagnostics"""
    print("🔍 Running Full System Diagnostics")
    print("=" * 50)
    
    # GPU Check
    print("\n1️⃣ GPU Status:")
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Memory usage
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   Allocated: {allocated:.1f}GB")
        print(f"   Reserved: {reserved:.1f}GB")
    else:
        print("   ❌ No GPU available")
    
    # Package Check
    print("\n2️⃣ Package Status:")
    packages = ['torch', 'transformers', 'peft', 'trl', 'datasets', 'wandb']
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {pkg}: {version}")
        except ImportError:
            print(f"   ❌ {pkg}: not installed")
    
    # Authentication Check
    print("\n3️⃣ Authentication Status:")
    
    # W&B
    try:
        import wandb
        if wandb.api.api_key:
            print("   ✅ W&B: authenticated")
        else:
            print("   ❌ W&B: not authenticated")
    except:
        print("   ❌ W&B: error")
    
    # HuggingFace
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"   ✅ HuggingFace: {user['name']}")
    except:
        print("   ❌ HuggingFace: not authenticated")
    
    # Storage Check
    print("\n4️⃣ Storage Status:")
    import shutil
    
    # Disk space
    total, used, free = shutil.disk_usage("/content")
    print(f"   Disk: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")
    
    # Google Drive
    if os.path.exists('/content/drive/MyDrive'):
        print("   ✅ Google Drive: mounted")
    else:
        print("   ❌ Google Drive: not mounted")
    
    # Network Check
    print("\n5️⃣ Network Status:")
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        print("   ✅ HuggingFace Hub: accessible")
    except:
        print("   ❌ HuggingFace Hub: not accessible")
    
    print("\n" + "=" * 50)
    print("Diagnostics complete!")

# Run full diagnostics
run_full_diagnostics()
```

### Configuration Validator

```python
def validate_configuration(config):
    """Validate configuration for common issues"""
    print("⚙️ Validating Configuration")
    
    issues = []
    warnings = []
    
    # Memory checks
    if config.training.sft.batch_size > 4:
        warnings.append("Large batch size may cause OOM on T4 GPU")
    
    if config.model.max_length > 2048:
        warnings.append("Long sequences may cause OOM on T4 GPU")
    
    # Learning rate checks
    if config.training.sft.learning_rate > 1e-3:
        warnings.append("High learning rate may cause instability")
    
    if config.training.sft.learning_rate < 1e-6:
        warnings.append("Very low learning rate may slow convergence")
    
    # LoRA checks
    if config.lora.r > 64:
        warnings.append("High LoRA rank increases memory usage")
    
    if config.lora.alpha / config.lora.r < 0.5:
        warnings.append("Low alpha/rank ratio may reduce LoRA effectiveness")
    
    # Dataset checks
    if config.datasets.sft.max_samples > 50000:
        warnings.append("Large dataset may exceed session time limits")
    
    # Print results
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"   • {issue}")
    
    if warnings:
        print("⚠️ Configuration Warnings:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not issues and not warnings:
        print("✅ Configuration looks good!")
    
    return len(issues) == 0

# Validate your configuration
# validate_configuration(config)
```

## 📞 Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide** thoroughly
2. **Search existing GitHub issues** for similar problems
3. **Try the diagnostic tools** above
4. **Test with demo configuration** to isolate the issue

### When Reporting Issues

Include this information:

```python
def collect_issue_report_info():
    """Collect information for issue reports"""
    import sys
    import torch
    import platform
    
    info = {
        "system": platform.platform(),
        "python": sys.version,
        "gpu": "None",
        "packages": {}
    }
    
    if torch.cuda.is_available():
        info["gpu"] = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)"
    
    packages = ['torch', 'transformers', 'peft', 'trl', 'datasets']
    for pkg in packages:
        try:
            module = __import__(pkg)
            info["packages"][pkg] = getattr(module, '__version__', 'unknown')
        except ImportError:
            info["packages"][pkg] = 'not installed'
    
    print("📋 Issue Report Information:")
    print(f"System: {info['system']}")
    print(f"Python: {info['python']}")
    print(f"GPU: {info['gpu']}")
    print("Packages:")
    for pkg, version in info['packages'].items():
        print(f"  {pkg}: {version}")
    
    return info

# Run this and include output in your issue report
collect_issue_report_info()
```

### Support Channels

1. **GitHub Issues** (Primary)
   - Detailed bug reports
   - Feature requests
   - Technical questions

2. **GitHub Discussions**
   - General questions
   - Community help
   - Best practices

3. **Documentation**
   - README guide
   - Notebook tutorials
   - API documentation

---

Remember: Most issues can be resolved by following this guide systematically. Start with the most common solutions and work through the diagnostic tools before seeking additional help.