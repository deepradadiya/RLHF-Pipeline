# RLHF Phi-3 Pipeline

A production-grade Reinforcement Learning from Human Feedback (RLHF) pipeline for Microsoft Phi-3 Mini (3.8B parameters). This comprehensive system is designed for Google Colab's T4 GPU constraints while delivering publishable results suitable for professional portfolios and research applications.

## 🎯 Overview

This pipeline implements a complete three-stage RLHF training process that transforms a base Phi-3 model into a helpful, harmless, and honest assistant:

### Training Stages

1. **🎓 Supervised Fine-Tuning (SFT)** 
   - Train on high-quality instruction-following datasets
   - Teaches the model to follow instructions and maintain conversation format
   - Uses PEFT/LoRA for memory-efficient training

2. **🏆 Reward Model Training**
   - Learn human preferences from comparison datasets
   - Trains a separate model to score response quality
   - Enables automated preference learning at scale

3. **🚀 Proximal Policy Optimization (PPO)**
   - Final RLHF training stage using the reward model
   - Optimizes the SFT model to maximize reward scores
   - Produces the final aligned model ready for deployment

### 🌟 Key Features

- **🧠 Memory Efficient**: PEFT/LoRA techniques reduce trainable parameters by 99%
- **💾 Persistent Checkpoints**: Google Drive integration for seamless session recovery
- **📊 Experiment Tracking**: Weights & Biases integration with comprehensive metrics
- **🎯 Professional Evaluation**: MT-Bench assessment for response quality
- **🤗 HuggingFace Integration**: One-click model publishing and sharing
- **⚡ Colab Optimized**: Designed for T4 GPU constraints and 12-hour sessions
- **🔒 Safety First**: Built-in content filtering and safety guardrails
- **📈 Production Ready**: Comprehensive testing and error handling

## 🚀 Quick Start

### Prerequisites

Before starting, ensure you have:
- **Google Account**: For Google Colab and Drive access
- **Weights & Biases Account**: For experiment tracking ([sign up free](https://wandb.ai/))
- **HuggingFace Account**: For model publishing ([sign up free](https://huggingface.co/))
- **Basic Python Knowledge**: Familiarity with Python and machine learning concepts

## 📋 Complete Setup Guide

This section provides comprehensive setup instructions for different environments and use cases.

### 🎯 Quick Setup Checklist

Before starting, ensure you have:
- [ ] **Google Account** (for Colab and Drive)
- [ ] **Weights & Biases Account** ([sign up free](https://wandb.ai/))
- [ ] **HuggingFace Account** ([sign up free](https://huggingface.co/))
- [ ] **Stable Internet Connection** (for dataset downloads)
- [ ] **Basic Python Knowledge** (familiarity with ML concepts helpful)

### 🚀 Environment-Specific Setup

#### Google Colab Setup (Recommended for Beginners)

**Why Choose Colab:**
- ✅ Free GPU access (T4 with 15GB VRAM)
- ✅ Pre-installed ML libraries
- ✅ No local setup required
- ✅ Automatic Google Drive integration
- ✅ Easy sharing and collaboration

**Complete Colab Setup Process:**

1. **Create New Colab Notebook**
   ```python
   # Step 1: Verify GPU access
   import torch
   print(f"🎮 GPU Available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU Name: {torch.cuda.get_device_name(0)}")
       print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
   else:
       print("❌ No GPU detected!")
       print("💡 Go to Runtime → Change runtime type → GPU")
   ```

2. **Mount Google Drive**
   ```python
   # Step 2: Mount Google Drive for persistent storage
   from google.colab import drive
   import os
   
   drive.mount('/content/drive')
   
   # Create project directory
   project_dir = '/content/drive/MyDrive/rlhf-phi3'
   os.makedirs(project_dir, exist_ok=True)
   print(f"📂 Project directory: {project_dir}")
   ```

3. **Install RLHF Pipeline**
   ```python
   # Step 3: Clone and install the pipeline
   !git clone https://github.com/your-username/rlhf-phi3-pipeline.git
   %cd rlhf-phi3-pipeline
   
   # Install dependencies
   !pip install -r requirements.txt
   !pip install -e .
   
   # Verify installation
   from rlhf_phi3 import Config
   print("✅ RLHF Phi-3 pipeline installed successfully!")
   ```

4. **Set Up Authentication**
   ```python
   # Step 4: Configure authentication
   import os
   from getpass import getpass
   
   # Weights & Biases
   print("🔑 Setting up Weights & Biases:")
   print("1. Go to https://wandb.ai/settings")
   print("2. Copy your API key")
   wandb_key = getpass("Enter W&B API key: ")
   os.environ['WANDB_API_KEY'] = wandb_key
   
   # HuggingFace
   print("\n🤗 Setting up HuggingFace:")
   print("1. Go to https://huggingface.co/settings/tokens")
   print("2. Create token with 'write' permissions")
   hf_token = getpass("Enter HF token: ")
   os.environ['HUGGINGFACE_TOKEN'] = hf_token
   
   # Test authentication
   import wandb
   wandb.login(key=wandb_key)
   
   from huggingface_hub import login
   login(token=hf_token)
   
   print("✅ Authentication complete!")
   ```

5. **Load Configuration and Start Training**
   ```python
   # Step 5: Load configuration and start training
   from rlhf_phi3 import Config, TrainingOrchestrator
   
   # Load Colab-optimized configuration
   config = Config.from_yaml("configs/colab_config.yaml")
   
   # Update paths for your Drive
   config.paths.base_output_dir = "/content/drive/MyDrive/rlhf-phi3"
   
   # Initialize orchestrator
   orchestrator = TrainingOrchestrator(config)
   
   # Run training (this will take several hours)
   final_model = orchestrator.run_full_pipeline()
   
   print(f"🎉 Training complete! Model saved to: {final_model}")
   ```

#### Local Development Setup

**Why Choose Local Setup:**
- ✅ Full control over environment
- ✅ No session time limits
- ✅ Better for development and debugging
- ✅ Can use more powerful hardware
- ✅ Offline capability after initial setup

**Prerequisites:**
- Python 3.8+ (3.9+ recommended)
- CUDA 11.8+ or 12.1+ with compatible drivers
- 16GB+ RAM (32GB+ recommended)
- 100GB+ free disk space
- GPU with 15GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)

**Detailed Local Setup:**

1. **System Requirements Check**
   ```bash
   # Check Python version
   python --version  # Should be 3.8+
   
   # Check CUDA installation
   nvidia-smi  # Should show GPU info
   nvcc --version  # Should show CUDA version
   
   # Check available disk space
   df -h  # Should have 100GB+ free
   ```

2. **Create Virtual Environment**
   ```bash
   # Using venv (recommended)
   python -m venv rlhf-env
   
   # Activate environment
   # On Linux/Mac:
   source rlhf-env/bin/activate
   # On Windows:
   rlhf-env\Scripts\activate
   
   # Verify activation
   which python  # Should point to venv
   ```

3. **Install PyTorch with CUDA**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Verify CUDA installation
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

4. **Clone and Install Pipeline**
   ```bash
   # Clone repository
   git clone https://github.com/your-username/rlhf-phi3-pipeline.git
   cd rlhf-phi3-pipeline
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install in development mode
   pip install -e .
   
   # Verify installation
   python -c "from rlhf_phi3 import Config; print('✅ Installation successful!')"
   ```

5. **Configure Environment Variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env file with your credentials
   nano .env  # or use your preferred editor
   ```

   **.env file contents:**
   ```bash
   # Weights & Biases
   WANDB_API_KEY=your_wandb_api_key_here
   
   # HuggingFace
   HUGGINGFACE_TOKEN=your_hf_token_here
   
   # Optional: Custom cache directories
   HF_HOME=/path/to/huggingface/cache
   WANDB_CACHE_DIR=/path/to/wandb/cache
   
   # Optional: Logging level
   LOG_LEVEL=INFO
   ```

6. **Run System Tests**
   ```bash
   # Test GPU access
   python -c "
   import torch
   print(f'GPU Available: {torch.cuda.is_available()}')
   if torch.cuda.is_available():
       print(f'GPU Name: {torch.cuda.get_device_name(0)}')
       print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
   "
   
   # Test package installation
   python -c "
   from rlhf_phi3.utils import system_check
   system_check.run_full_check()
   "
   
   # Run demo training
   python -c "
   from rlhf_phi3 import Config, TrainingOrchestrator
   config = Config.from_yaml('configs/demo_config.yaml')
   orchestrator = TrainingOrchestrator(config)
   orchestrator.run_demo_pipeline()
   "
   ```

#### Cloud Platform Setup (AWS, GCP, Azure)

**Why Choose Cloud Platforms:**
- ✅ Scalable compute resources
- ✅ Professional-grade GPUs (A100, H100)
- ✅ Pay-per-use pricing
- ✅ Pre-configured ML environments
- ✅ Easy scaling and collaboration

**AWS SageMaker Setup:**

1. **Create SageMaker Notebook Instance**
   ```python
   # In SageMaker notebook
   import boto3
   import sagemaker
   
   # Check instance type and GPU
   !nvidia-smi
   
   # Install pipeline
   !git clone https://github.com/your-username/rlhf-phi3-pipeline.git
   %cd rlhf-phi3-pipeline
   !pip install -r requirements.txt
   !pip install -e .
   ```

2. **Configure S3 Storage**
   ```python
   # Set up S3 bucket for checkpoints
   import boto3
   
   s3_bucket = "your-rlhf-bucket"
   s3_prefix = "rlhf-phi3-checkpoints"
   
   # Update configuration
   config = Config.from_yaml("configs/default_config.yaml")
   config.paths.base_output_dir = f"s3://{s3_bucket}/{s3_prefix}"
   ```

**Google Cloud Platform Setup:**

1. **Create Vertex AI Workbench**
   ```bash
   # In Vertex AI notebook
   # GPU instance recommended: n1-standard-4 with 1x NVIDIA Tesla T4
   
   # Install pipeline
   git clone https://github.com/your-username/rlhf-phi3-pipeline.git
   cd rlhf-phi3-pipeline
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Configure Cloud Storage**
   ```python
   # Set up GCS bucket
   from google.cloud import storage
   
   bucket_name = "your-rlhf-bucket"
   
   # Update configuration
   config = Config.from_yaml("configs/default_config.yaml")
   config.paths.base_output_dir = f"gs://{bucket_name}/rlhf-phi3"
   ```

### 🔧 Configuration Customization

#### Basic Configuration Options

```yaml
# configs/my_custom_config.yaml

# Model settings
model:
  name: "microsoft/Phi-3-mini-4k-instruct"
  max_length: 2048  # Reduce for memory constraints
  device: "auto"

# LoRA settings (adjust for performance vs. memory trade-off)
lora:
  r: 16        # Higher = more parameters, better performance
  alpha: 32    # Usually 2x the rank
  dropout: 0.1 # Regularization
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training settings
training:
  sft:
    epochs: 3
    learning_rate: 2e-4
    batch_size: 4      # Adjust based on GPU memory
    gradient_accumulation_steps: 4
    max_steps: 1000
  
  reward:
    epochs: 1
    learning_rate: 1e-4
    batch_size: 2
    gradient_accumulation_steps: 8
    max_steps: 500
  
  ppo:
    learning_rate: 1e-5
    batch_size: 1
    mini_batch_size: 1
    gradient_accumulation_steps: 16
    ppo_epochs: 4
    max_steps: 1000

# Dataset settings
datasets:
  sft:
    name: "HuggingFaceH4/ultrachat_200k"
    split: "train_sft"
    max_samples: 10000  # Reduce for faster training
  
  preference:
    name: "HuggingFaceH4/ultrafeedback_binarized"
    split: "train_prefs"
    max_samples: 5000

# Paths (customize for your environment)
paths:
  base_output_dir: "/path/to/your/output"
  cache_dir: "/path/to/cache"
  logs_dir: "/path/to/logs"

# Experiment tracking
wandb:
  project: "my-rlhf-project"
  entity: "your-username"
  tags: ["phi3", "rlhf", "custom"]
```

#### Advanced Configuration Options

```yaml
# Advanced optimization settings
optimization:
  optimizer_type: "adamw_torch"  # or "adamw_hf", "sgd"
  scheduler_type: "cosine"       # or "linear", "constant"
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: true                     # Enable mixed precision
  gradient_checkpointing: true   # Trade compute for memory
  dataloader_num_workers: 4      # Parallel data loading

# Memory optimization
memory:
  use_flash_attention: true      # Faster attention computation
  offload_optimizer: false       # Offload optimizer to CPU
  offload_params: false          # Offload parameters to CPU
  pin_memory: true               # Faster GPU transfers

# Checkpointing
checkpointing:
  save_steps: 100               # Save every N steps
  save_total_limit: 3           # Keep only N checkpoints
  resume_from_checkpoint: null  # Path to resume from
  save_optimizer: true          # Save optimizer state

# Evaluation
evaluation:
  eval_steps: 200               # Evaluate every N steps
  eval_strategy: "steps"        # or "epoch"
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  
  mt_bench:
    num_samples: 100
    temperature: 0.7
    max_new_tokens: 512
    do_sample: true

# Logging
logging:
  level: "INFO"                 # DEBUG, INFO, WARNING, ERROR
  log_steps: 10                 # Log every N steps
  report_to: ["wandb"]          # Also supports "tensorboard"
  
# Safety and filtering
safety:
  enable_content_filter: true
  filter_threshold: 0.8
  safety_model: "unitary/toxic-bert"
```

### 🎮 Hardware Optimization Guide

#### GPU Memory Optimization

| GPU Model | VRAM | Recommended Settings |
|-----------|------|---------------------|
| **RTX 3060** | 12GB | batch_size=1, max_length=1024, lora_r=8 |
| **RTX 3070** | 8GB | batch_size=1, max_length=512, lora_r=4 |
| **RTX 3080** | 10GB | batch_size=1, max_length=1024, lora_r=8 |
| **RTX 3090** | 24GB | batch_size=2, max_length=2048, lora_r=16 |
| **RTX 4070** | 12GB | batch_size=1, max_length=1024, lora_r=8 |
| **RTX 4080** | 16GB | batch_size=2, max_length=1536, lora_r=12 |
| **RTX 4090** | 24GB | batch_size=4, max_length=2048, lora_r=16 |
| **Tesla T4** | 15GB | batch_size=1, max_length=1024, lora_r=8 |
| **Tesla V100** | 32GB | batch_size=4, max_length=2048, lora_r=16 |
| **A100 40GB** | 40GB | batch_size=8, max_length=4096, lora_r=32 |
| **A100 80GB** | 80GB | batch_size=16, max_length=4096, lora_r=64 |

#### CPU and RAM Recommendations

| System RAM | Recommended Usage |
|------------|-------------------|
| **16GB** | Basic training, small datasets |
| **32GB** | Standard training, medium datasets |
| **64GB+** | Large datasets, multiple experiments |

#### Storage Requirements

| Component | Space Required |
|-----------|----------------|
| **Base Installation** | 5GB |
| **Model Cache** | 10-20GB |
| **Dataset Cache** | 20-50GB |
| **Checkpoints** | 10-30GB per stage |
| **Logs and Outputs** | 5-10GB |
| **Total Recommended** | 100GB+ |

### 🧪 Testing Your Setup

#### Quick Validation Test

```python
def validate_complete_setup():
    """Comprehensive setup validation"""
    print("🧪 Running Complete Setup Validation")
    print("=" * 50)
    
    # 1. Import test
    try:
        from rlhf_phi3 import Config, TrainingOrchestrator
        print("✅ Package imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # 2. GPU test
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️ No GPU available (CPU training will be very slow)")
    
    # 3. Configuration test
    try:
        config = Config.from_yaml("configs/colab_config.yaml")
        errors = config.validate()
        if errors:
            print(f"❌ Configuration errors: {errors}")
        else:
            print("✅ Configuration valid")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # 4. Authentication test
    auth_status = {"wandb": False, "hf": False}
    
    try:
        import wandb
        wandb.init(project="test", mode="disabled")
        auth_status["wandb"] = True
        print("✅ W&B authentication OK")
    except:
        print("⚠️ W&B authentication not configured")
    
    try:
        from huggingface_hub import whoami
        whoami()
        auth_status["hf"] = True
        print("✅ HuggingFace authentication OK")
    except:
        print("⚠️ HuggingFace authentication not configured")
    
    # 5. Dataset connectivity test
    try:
        from datasets import load_dataset
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:1]", streaming=True)
        next(iter(dataset))
        print("✅ Dataset connectivity OK")
    except Exception as e:
        print(f"⚠️ Dataset connectivity issue: {e}")
    
    # 6. Model loading test
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        print("✅ Model loading OK")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    
    # Summary
    if all([auth_status["wandb"], auth_status["hf"]]):
        print("🎉 Setup validation PASSED - Ready for training!")
        return True
    else:
        print("⚠️ Setup validation PARTIAL - Some features may not work")
        print("💡 Complete authentication setup for full functionality")
        return True

# Run validation
validate_complete_setup()
```

#### Demo Training Test

```python
def run_demo_training():
    """Run a quick demo to test the complete pipeline"""
    print("🚀 Running Demo Training Test")
    
    try:
        from rlhf_phi3 import Config, TrainingOrchestrator
        
        # Load demo configuration
        config = Config.from_yaml("configs/demo_config.yaml")
        
        # Override for very quick test
        config.datasets.sft.max_samples = 10
        config.datasets.preference.max_samples = 5
        config.training.sft.max_steps = 5
        config.training.reward.max_steps = 3
        config.training.ppo.max_steps = 3
        
        print("⚙️ Demo configuration loaded")
        
        # Initialize orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Run demo pipeline
        print("🎯 Starting demo training...")
        result = orchestrator.run_demo_pipeline()
        
        if result:
            print("✅ Demo training completed successfully!")
            print(f"📂 Demo model saved to: {result}")
            return True
        else:
            print("❌ Demo training failed")
            return False
            
    except Exception as e:
        print(f"❌ Demo training error: {e}")
        return False

# Run demo (uncomment to test)
# run_demo_training()
```

Your setup is now complete! Choose your preferred environment and follow the corresponding setup guide. The troubleshooting section above will help you resolve any issues you encounter.

The easiest way to get started is using Google Colab's free T4 GPU. This comprehensive guide will walk you through every step of the setup process.

#### 🚀 Quick Start (5 Minutes)

1. **Open the Setup Notebook**
   ```
   https://colab.research.google.com/github/your-username/rlhf-phi3-pipeline/blob/main/notebooks/01_setup_and_configuration.ipynb
   ```

2. **Follow the Interactive Setup**
   - The notebook will guide you through authentication
   - Automatically configure Google Drive integration
   - Set up experiment tracking with Weights & Biases
   - Install all required dependencies

3. **Start Training**
   - Run the complete pipeline with a single command
   - Monitor progress through interactive dashboards
   - Checkpoints automatically saved to Google Drive

#### 📋 Detailed Google Colab Setup Guide

**Prerequisites:**
- Google account (free)
- Stable internet connection
- Web browser (Chrome recommended)

**Step 1: Create New Colab Notebook**

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `New notebook` or `File` → `New notebook`
3. Rename your notebook: `File` → `Rename` → "RLHF Phi-3 Pipeline"

**Step 2: Enable GPU Runtime**

1. **Change Runtime Type**
   - Go to `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Choose `T4 GPU` (free tier) or `A100/V100` (Colab Pro)
   - Set `Runtime shape` to `Standard` (free) or `High-RAM` (Pro)
   - Click `Save`

2. **Verify GPU Access**
   ```python
   import torch
   import subprocess
   
   print("🔍 System Information:")
   print(f"GPU Available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU Name: {torch.cuda.get_device_name(0)}")
       print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
       print(f"CUDA Version: {torch.version.cuda}")
   else:
       print("❌ No GPU detected. Please check runtime settings.")
   
   # Check Python version
   import sys
   print(f"Python Version: {sys.version}")
   
   # Check available RAM
   import psutil
   ram = psutil.virtual_memory()
   print(f"Available RAM: {ram.total / 1e9:.1f}GB")
   ```

**Step 3: Mount Google Drive**

```python
from google.colab import drive
import os

# Mount Google Drive
print("📁 Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

# Verify mount
if os.path.exists('/content/drive/MyDrive'):
    print("✅ Google Drive mounted successfully")
    
    # Create project directory
    project_dir = '/content/drive/MyDrive/rlhf-phi3'
    os.makedirs(project_dir, exist_ok=True)
    print(f"📂 Project directory created: {project_dir}")
else:
    print("❌ Google Drive mount failed")
    print("💡 Try: Runtime → Restart runtime, then re-run this cell")
```

**Step 4: Clone Repository and Install Dependencies**

```python
import subprocess
import sys

def run_command(cmd, description):
    """Run command with progress indication"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        raise

# Clone repository
run_command(
    "git clone https://github.com/your-username/rlhf-phi3-pipeline.git",
    "Cloning repository"
)

# Change to project directory
os.chdir('/content/rlhf-phi3-pipeline')
print(f"📂 Changed to directory: {os.getcwd()}")

# Install dependencies
run_command(
    "pip install -r requirements.txt",
    "Installing dependencies"
)

# Install package in development mode
run_command(
    "pip install -e .",
    "Installing RLHF Phi-3 package"
)

# Verify installation
try:
    from rlhf_phi3 import Config
    print("✅ RLHF Phi-3 package installed successfully")
except ImportError as e:
    print(f"❌ Installation verification failed: {e}")
```

**Step 5: Set Up Authentication**

```python
import os
from getpass import getpass

print("🔐 Setting up authentication...")

# Weights & Biases setup
print("\n1. Weights & Biases Setup:")
print("   - Go to https://wandb.ai/settings")
print("   - Copy your API key")
wandb_key = getpass("   Enter your W&B API key: ")
os.environ['WANDB_API_KEY'] = wandb_key

# Test W&B authentication
try:
    import wandb
    wandb.login(key=wandb_key)
    print("✅ W&B authentication successful")
except Exception as e:
    print(f"❌ W&B authentication failed: {e}")

# HuggingFace setup
print("\n2. HuggingFace Setup:")
print("   - Go to https://huggingface.co/settings/tokens")
print("   - Create a new token with 'write' permissions")
hf_token = getpass("   Enter your HuggingFace token: ")
os.environ['HUGGINGFACE_TOKEN'] = hf_token

# Test HF authentication
try:
    from huggingface_hub import login, whoami
    login(token=hf_token)
    user_info = whoami()
    print(f"✅ HuggingFace authenticated as: {user_info['name']}")
except Exception as e:
    print(f"❌ HuggingFace authentication failed: {e}")

print("\n🎉 Authentication setup complete!")
```

**Step 6: Load and Validate Configuration**

```python
from rlhf_phi3 import Config
import yaml

# Load Colab-optimized configuration
config_path = "configs/colab_config.yaml"
config = Config.from_yaml(config_path)

print("⚙️ Configuration loaded:")
print(f"   Model: {config.model.name}")
print(f"   Max Length: {config.model.max_length}")
print(f"   LoRA Rank: {config.lora.r}")
print(f"   SFT Epochs: {config.training.sft.epochs}")
print(f"   Output Directory: {config.paths.base_output_dir}")

# Validate configuration
errors = config.validate()
if errors:
    print("❌ Configuration errors:")
    for error in errors:
        print(f"   • {error}")
else:
    print("✅ Configuration is valid")

# Save configuration to Drive
config_save_path = "/content/drive/MyDrive/rlhf-phi3/my_config.yaml"
config.save_yaml(config_save_path)
print(f"💾 Configuration saved to: {config_save_path}")
```

**Step 7: Run Quick System Test**

```python
from rlhf_phi3.utils import system_check

print("🧪 Running system tests...")

# Run comprehensive system check
try:
    system_check.run_full_check(config)
    print("✅ All system checks passed!")
    print("\n🚀 Ready to start training!")
    
    # Display next steps
    print("\n📋 Next Steps:")
    print("1. Run the demo pipeline: orchestrator.run_demo_pipeline()")
    print("2. Or start full training: orchestrator.run_full_pipeline()")
    print("3. Monitor progress in W&B dashboard")
    print("4. Checkpoints will be saved to Google Drive")
    
except Exception as e:
    print(f"❌ System check failed: {e}")
    print("💡 Check the troubleshooting section below")
```

**Colab-Specific Optimizations:**

| Feature | Free Tier | Colab Pro | Colab Pro+ |
|---------|-----------|-----------|------------|
| **Session Limit** | 12 hours | 24 hours | 24 hours |
| **GPU Options** | T4 (15GB) | T4, V100 (32GB) | A100 (40GB) |
| **RAM** | 12GB | 25GB | 51GB |
| **Storage** | 100GB+ | 100GB+ | 100GB+ |
| **Priority** | Standard | High | Highest |

**Memory Management Tips:**
- **T4 GPU (15GB)**: Use `batch_size=1-2`, `max_length=1024-2048`
- **V100 GPU (32GB)**: Use `batch_size=2-4`, `max_length=2048-4096`
- **A100 GPU (40GB+)**: Use `batch_size=4-8`, `max_length=4096+`

**Session Management:**
- Sessions timeout automatically (12-24 hours depending on tier)
- Checkpoints are saved every 50-100 steps to Google Drive
- Use session keepalive techniques for long training runs
- Monitor session time with built-in warnings

**Storage Optimization:**
- All checkpoints saved to Google Drive (persistent)
- Local storage is temporary and cleared on session end
- Use streaming datasets to minimize local storage usage
- Compress checkpoints before uploading to Drive

### Option 2: Local Installation

For local development or custom environments:

```bash
# Clone the repository
git clone https://github.com/your-username/rlhf-phi3-pipeline.git
cd rlhf-phi3-pipeline

# Create virtual environment (recommended)
python -m venv rlhf-env
source rlhf-env/bin/activate  # On Windows: rlhf-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

#### Local Installation Guide

**Prerequisites:**
- Python 3.8+ (3.9+ recommended)
- CUDA 11.8+ or 12.1+ with compatible GPU drivers
- Git for version control
- 50GB+ free disk space

**Detailed Local Setup:**

1. **Check Python Version**
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Verify CUDA Installation**
   ```bash
   nvidia-smi  # Check GPU status
   nvcc --version  # Check CUDA version
   ```

3. **Create Virtual Environment**
   ```bash
   # Using venv (recommended)
   python -m venv rlhf-env
   source rlhf-env/bin/activate  # Linux/Mac
   # OR
   rlhf-env\Scripts\activate  # Windows
   
   # Using conda (alternative)
   conda create -n rlhf-env python=3.9
   conda activate rlhf-env
   ```

4. **Install PyTorch with CUDA**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Verify installation
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

5. **Install Pipeline Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

6. **Set Up Environment Variables**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env file with your credentials
   nano .env  # or use your preferred editor
   ```

**Environment Variables (.env file):**
```bash
# Weights & Biases
WANDB_API_KEY=your_wandb_api_key_here

# HuggingFace
HUGGINGFACE_TOKEN=your_hf_token_here

# Optional: Custom cache directories
HF_HOME=/path/to/huggingface/cache
WANDB_CACHE_DIR=/path/to/wandb/cache

# Optional: Logging level
LOG_LEVEL=INFO
```

**Verification Steps:**
```bash
# Test installation
python -c "from rlhf_phi3 import Config; print('✅ Installation successful!')"

# Run system check
python -c "
from rlhf_phi3.utils import system_check
system_check.run_full_check()
"

# Test GPU access
python -c "
import torch
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

### 🎮 Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **GPU** | T4 (15GB VRAM) | A100 (40GB+) | Free in Google Colab |
| **RAM** | 12GB | 32GB+ | System memory |
| **Storage** | 50GB | 100GB+ | For checkpoints and datasets |
| **Internet** | Stable connection | High-speed | For dataset downloads |

### ⚡ 5-Minute Demo

Want to see it in action? Run this minimal example:

```python
from rlhf_phi3 import Config, TrainingOrchestrator

# Load optimized configuration for quick demo
config = Config.from_yaml("configs/demo_config.yaml")

# Initialize the training orchestrator
orchestrator = TrainingOrchestrator(config)

# Run a minimal training demo (uses toy datasets)
demo_model = orchestrator.run_demo_pipeline()

print(f"Demo completed! Model saved to: {demo_model}")
```

This demo runs in ~5 minutes and demonstrates all pipeline components with minimal datasets.

## 📁 Project Structure

```
rlhf_phi3/                    # Main package directory
├── config/                   # Configuration management
│   └── config_manager.py     # Centralized config with validation
├── data/                     # Dataset loading and preprocessing
│   └── dataset_manager.py    # HuggingFace integration & streaming
├── models/                   # Model management and PEFT
│   └── model_manager.py      # Phi-3 loading, LoRA, checkpoints
├── training/                 # Training orchestration
│   ├── training_orchestrator.py  # Pipeline coordination
│   ├── sft_trainer.py        # Supervised fine-tuning
│   ├── reward_trainer.py     # Reward model training
│   └── ppo_trainer.py        # PPO/RLHF training
├── checkpoints/              # Checkpoint persistence
│   └── checkpoint_manager.py # Google Drive integration
├── tracking/                 # Experiment tracking
│   └── experiment_tracker.py # Weights & Biases integration
├── evaluation/               # Model evaluation
│   └── evaluation_engine.py  # MT-Bench & quality assessment
├── publishing/               # Model publishing
│   └── model_publisher.py    # HuggingFace Hub integration
└── utils/                    # Common utilities
    └── error_handler.py      # Comprehensive error handling

notebooks/                    # Interactive tutorials
├── 01_setup_and_configuration.ipynb
├── 02_sft_training_tutorial.ipynb
├── 03_reward_model_tutorial.ipynb
├── 04_ppo_training_tutorial.ipynb
└── 05_evaluation_and_publishing.ipynb

tests/                        # Comprehensive test suite
├── unit/                     # Unit tests for components
├── property/                 # Property-based tests
├── integration/              # End-to-end integration tests
└── fixtures/                 # Test data and utilities

configs/                      # Configuration files
├── default_config.yaml       # Standard configuration
├── colab_config.yaml         # Optimized for Google Colab
└── demo_config.yaml          # Quick demo configuration
```

## 💻 Usage Examples

### Complete Pipeline Execution

```python
from rlhf_phi3 import Config, TrainingOrchestrator

# Load configuration (automatically validates all parameters)
config = Config.from_yaml("configs/default_config.yaml")

# Initialize the training orchestrator
orchestrator = TrainingOrchestrator(config)

# Run the complete three-stage pipeline
final_model_path = orchestrator.run_full_pipeline()

print(f"🎉 Training complete! Model saved to: {final_model_path}")
```

### Stage-by-Stage Execution

For more control over the training process:

```python
# Run individual stages with checkpointing
sft_checkpoint = orchestrator.run_sft_stage()
print(f"✅ SFT completed: {sft_checkpoint}")

reward_checkpoint = orchestrator.run_reward_stage(sft_checkpoint)
print(f"✅ Reward model completed: {reward_checkpoint}")

final_model = orchestrator.run_ppo_stage(sft_checkpoint, reward_checkpoint)
print(f"🚀 PPO completed: {final_model}")
```

### Resume from Checkpoint

If training is interrupted, easily resume from any checkpoint:

```python
# Resume from a specific stage
final_model = orchestrator.resume_from_stage(
    stage="ppo",
    checkpoint_path="/path/to/checkpoint"
)
```

### Custom Configuration

Create custom configurations for different use cases:

```python
from rlhf_phi3 import Config, LoRAConfig, ModelConfig

# Create custom LoRA configuration
lora_config = LoRAConfig(
    r=32,                    # Higher rank for better performance
    alpha=64,                # Scaled alpha
    dropout=0.05,            # Lower dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
)

# Create custom model configuration
model_config = ModelConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_length=4096,         # Full context length
    device="cuda",
    torch_dtype="float16"
)

# Build complete configuration
config = Config(
    model=model_config,
    lora=lora_config,
    sft_epochs=5,            # More training epochs
    batch_size=2,            # Adjust for your GPU
    learning_rate=2e-4       # Custom learning rate
)

# Validate configuration
if config.validate():
    orchestrator = TrainingOrchestrator(config)
    final_model = orchestrator.run_full_pipeline()
```

### Evaluation and Publishing

```python
from rlhf_phi3 import EvaluationEngine, ModelPublisher

# Evaluate the trained model
evaluator = EvaluationEngine(final_model, config)
results = evaluator.run_comprehensive_evaluation()

print(f"📊 MT-Bench Score: {results.mt_bench_score:.2f}/10.0")
print(f"🎯 Helpfulness: {results.helpfulness_score:.2f}/10.0")

# Publish to HuggingFace Hub
publisher = ModelPublisher(config)
model_url = publisher.publish_model(
    model_path=final_model,
    model_name="my-rlhf-phi3-model",
    evaluation_results=results
)

print(f"🤗 Model published: {model_url}")
```

## ⚙️ Configuration

The pipeline uses YAML configuration files for easy customization. All configurations include automatic validation and helpful error messages.

### Configuration Files

- **`configs/default_config.yaml`** - Production-ready settings
- **`configs/colab_config.yaml`** - Optimized for Google Colab T4 GPU
- **`configs/demo_config.yaml`** - Quick demo with toy datasets

### Key Configuration Sections

#### Model Configuration
```yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  max_length: 2048                    # Context window
  device: "cuda"
  torch_dtype: "float16"              # Memory optimization
```

#### LoRA Configuration
```yaml
lora:
  r: 16                               # LoRA rank (higher = more parameters)
  alpha: 32                           # LoRA scaling factor
  dropout: 0.1                        # Regularization
  target_modules:                     # Which layers to adapt
    - "q_proj"
    - "k_proj" 
    - "v_proj"
    - "o_proj"
```

#### Training Configuration
```yaml
training:
  sft_epochs: 3                       # Supervised fine-tuning epochs
  reward_epochs: 1                    # Reward model epochs
  ppo_steps: 1000                     # PPO optimization steps
  
  batch_size: 4                       # Per-device batch size
  gradient_accumulation_steps: 4      # Effective batch size multiplier
  learning_rate: 5e-5                 # Base learning rate
  
  warmup_steps: 100                   # Learning rate warmup
  max_grad_norm: 1.0                  # Gradient clipping
```

#### Dataset Configuration
```yaml
datasets:
  sft_dataset: "HuggingFaceH4/ultrachat_200k"
  preference_dataset: "HuggingFaceH4/ultrafeedback_binarized"
  streaming: true                     # Memory-efficient loading
  max_samples: 10000                  # Limit for quick training
```

#### Paths and Storage
```yaml
paths:
  base_output_dir: "/content/drive/MyDrive/rlhf-phi3"
  checkpoint_dir: "checkpoints"
  logs_dir: "logs"
  
experiment_tracking:
  wandb_project: "rlhf-phi3-pipeline"
  wandb_entity: "your-username"       # Your W&B username
  log_every_n_steps: 10
```

### Configuration Validation

The pipeline automatically validates all configuration parameters:

```python
config = Config.from_yaml("configs/custom_config.yaml")

# Check for validation errors
errors = config.validate()
if errors:
    for error in errors:
        print(f"❌ {error}")
else:
    print("✅ Configuration is valid!")
```

### Environment Variables

Set these environment variables for authentication:

```bash
# Weights & Biases (for experiment tracking)
export WANDB_API_KEY="your_wandb_api_key"

# HuggingFace (for model publishing)
export HUGGINGFACE_TOKEN="your_hf_token"

# Optional: Custom cache directory
export HF_HOME="/path/to/cache"
```

## 📋 Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Google Colab |
|-----------|---------|-------------|--------------|
| **GPU** | T4 (15GB VRAM) | A100 (40GB+) | ✅ T4 Free |
| **System RAM** | 12GB | 32GB+ | ✅ 12GB |
| **Storage** | 50GB | 100GB+ | ✅ 100GB+ |
| **Internet** | Stable | High-speed | ✅ Fast |

### Software Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **CUDA**: 11.8+ or 12.1+ 
- **PyTorch**: 2.0+ with CUDA support
- **Transformers**: 4.36+ (for Phi-3 support)

### Key Dependencies

```
torch>=2.0.0                 # PyTorch with CUDA
transformers>=4.36.0         # HuggingFace Transformers (Phi-3 support)
peft>=0.7.0                  # Parameter-Efficient Fine-Tuning
trl>=0.7.0                   # Transformer Reinforcement Learning
datasets>=2.14.0             # HuggingFace Datasets
accelerate>=0.24.0           # Distributed training utilities
wandb>=0.16.0                # Experiment tracking
bitsandbytes>=0.41.0         # Quantization and optimization
evaluate>=0.4.0              # Evaluation metrics
```

### Account Requirements

1. **Google Account** (Free)
   - For Google Colab access
   - For Google Drive checkpoint storage

2. **Weights & Biases Account** (Free tier available)
   - Sign up at [wandb.ai](https://wandb.ai/)
   - Used for experiment tracking and visualization

3. **HuggingFace Account** (Free)
   - Sign up at [huggingface.co](https://huggingface.co/)
   - Required for model publishing and some datasets

## 🛠️ Troubleshooting

This comprehensive troubleshooting guide covers the most common issues you might encounter and provides step-by-step solutions.

### 🚨 Critical Issues (Immediate Action Required)

#### GPU Memory Issues

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
- Inefficient memory usage patterns
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

**Long-term Solutions:**

1. **Automatic Memory Management**
   ```python
   class MemoryManager:
       def __init__(self, target_memory_percent=85):
           self.target_memory_percent = target_memory_percent
           
       def optimize_batch_size(self, config):
           """Automatically adjust batch size based on available memory"""
           if not torch.cuda.is_available():
               return config
               
           total_memory = torch.cuda.get_device_properties(0).total_memory
           target_memory = total_memory * (self.target_memory_percent / 100)
           
           # Start with current batch size and reduce if needed
           current_batch_size = config.training.sft.batch_size
           
           while current_batch_size > 0:
               # Estimate memory usage (rough approximation)
               estimated_memory = self.estimate_memory_usage(config, current_batch_size)
               
               if estimated_memory <= target_memory:
                   break
                   
               current_batch_size = max(1, current_batch_size // 2)
               
           # Update configuration
           config.training.sft.batch_size = current_batch_size
           config.training.reward.batch_size = current_batch_size
           config.training.ppo.batch_size = current_batch_size
           
           # Adjust gradient accumulation to maintain effective batch size
           original_effective_batch = 4 * 4  # original batch_size * grad_accum
           new_grad_accum = max(1, original_effective_batch // current_batch_size)
           
           config.training.sft.gradient_accumulation_steps = new_grad_accum
           config.training.reward.gradient_accumulation_steps = new_grad_accum * 2
           config.training.ppo.gradient_accumulation_steps = new_grad_accum * 4
           
           return config
           
       def estimate_memory_usage(self, config, batch_size):
           """Rough estimation of memory usage"""
           # This is a simplified estimation
           model_params = 3.8e9  # Phi-3 Mini parameters
           bytes_per_param = 2 if config.optimization.fp16 else 4
           
           model_memory = model_params * bytes_per_param
           activation_memory = batch_size * config.model.max_length * 4096 * bytes_per_param
           optimizer_memory = model_memory * 2  # Adam optimizer
           
           return model_memory + activation_memory + optimizer_memory
   
   # Use automatic memory management
   memory_manager = MemoryManager()
   config = memory_manager.optimize_batch_size(config)
   ```

2. **Memory Monitoring Dashboard**
   ```python
   import matplotlib.pyplot as plt
   import time
   from threading import Thread
   
   class MemoryMonitor:
       def __init__(self):
           self.monitoring = False
           self.memory_history = []
           
       def start_monitoring(self):
           """Start continuous memory monitoring"""
           self.monitoring = True
           monitor_thread = Thread(target=self._monitor_loop)
           monitor_thread.daemon = True
           monitor_thread.start()
           
       def stop_monitoring(self):
           """Stop memory monitoring"""
           self.monitoring = False
           
       def _monitor_loop(self):
           """Continuous monitoring loop"""
           while self.monitoring:
               if torch.cuda.is_available():
                   allocated = torch.cuda.memory_allocated() / 1e9
                   reserved = torch.cuda.memory_reserved() / 1e9
                   total = torch.cuda.get_device_properties(0).total_memory / 1e9
                   
                   self.memory_history.append({
                       'timestamp': time.time(),
                       'allocated': allocated,
                       'reserved': reserved,
                       'total': total,
                       'usage_percent': (allocated / total) * 100
                   })
                   
                   # Alert if memory usage is high
                   if (allocated / total) > 0.9:
                       print(f"⚠️ High GPU memory usage: {allocated:.1f}GB / {total:.1f}GB ({(allocated/total)*100:.1f}%)")
                       
               time.sleep(5)  # Check every 5 seconds
               
       def plot_memory_usage(self):
           """Plot memory usage over time"""
           if not self.memory_history:
               print("No memory data to plot")
               return
               
           timestamps = [h['timestamp'] for h in self.memory_history]
           allocated = [h['allocated'] for h in self.memory_history]
           reserved = [h['reserved'] for h in self.memory_history]
           
           plt.figure(figsize=(12, 6))
           plt.plot(timestamps, allocated, label='Allocated', color='red')
           plt.plot(timestamps, reserved, label='Reserved', color='orange')
           plt.axhline(y=self.memory_history[0]['total'], color='black', linestyle='--', label='Total GPU Memory')
           plt.xlabel('Time')
           plt.ylabel('Memory (GB)')
           plt.title('GPU Memory Usage Over Time')
           plt.legend()
           plt.grid(True)
           plt.show()
   
   # Start memory monitoring
   monitor = MemoryMonitor()
   monitor.start_monitoring()
   ```

**Memory Usage Guidelines by GPU:**

| GPU Model | VRAM | Recommended Settings |
|-----------|------|---------------------|
| **T4** | 15GB | batch_size=1, max_length=1024, lora_r=8 |
| **V100** | 32GB | batch_size=2-4, max_length=2048, lora_r=16 |
| **A100** | 40GB+ | batch_size=4-8, max_length=4096, lora_r=32 |

**Advanced Memory Optimization Techniques:**

1. **Gradient Checkpointing**
   ```python
   # Trade compute for memory
   config.optimization.gradient_checkpointing = True
   
   # Custom checkpointing for specific layers
   def apply_gradient_checkpointing(model):
       if hasattr(model, 'gradient_checkpointing_enable'):
           model.gradient_checkpointing_enable()
       return model
   ```

2. **CPU Offloading**
   ```python
   # Offload optimizer states to CPU
   config.optimization.offload_optimizer = True
   
   # Offload parameters to CPU when not in use
   config.optimization.offload_params = True
   ```

3. **Memory-Efficient Attention**
   ```python
   # Use Flash Attention for memory efficiency
   config.model.use_flash_attention = True
   
   # Enable memory-efficient attention patterns
   config.model.attention_implementation = "flash_attention_2"
   ```

```python
# Emergency memory optimization
import torch
torch.cuda.empty_cache()

# Reduce batch size in config
config.batch_size = 1
config.gradient_accumulation_steps = 16  # Maintain effective batch size

# Enable additional memory optimizations
config.model.torch_dtype = "float16"
config.model.max_length = 1024  # Reduce from 2048
config.training.gradient_checkpointing = True
```

**Memory Usage Guidelines**:
- **T4 GPU (15GB)**: batch_size=1-2, max_length=1024-2048
- **V100 GPU (32GB)**: batch_size=2-4, max_length=2048-4096  
- **A100 GPU (40GB+)**: batch_size=4-8, max_length=4096+

**Advanced Memory Optimization:**
```python
# Enable memory-efficient attention
config.model.use_flash_attention = True

# Use gradient checkpointing
config.training.gradient_checkpointing = True

# Enable CPU offloading for large models
config.model.offload_to_cpu = True

# Use DeepSpeed ZeRO for multi-GPU setups
config.training.use_deepspeed = True
config.training.deepspeed_config = "configs/deepspeed_zero2.json"
```

**Monitoring Memory Usage**:
```python
import torch
import psutil

def print_memory_usage():
    """Print comprehensive memory usage statistics"""
    if torch.cuda.is_available():
        # GPU Memory
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total_gpu:.1f}GB total")
        print(f"   Usage: {(allocated/total_gpu)*100:.1f}% allocated, {(reserved/total_gpu)*100:.1f}% reserved")
    
    # System Memory
    memory = psutil.virtual_memory()
    print(f"💾 System RAM: {memory.used/1e9:.1f}GB used / {memory.total/1e9:.1f}GB total ({memory.percent:.1f}%)")
    
    # Disk Space
    import shutil
    disk = shutil.disk_usage(".")
    print(f"💿 Disk Space: {disk.free/1e9:.1f}GB free / {disk.total/1e9:.1f}GB total")

# Call this function to monitor memory usage
print_memory_usage()
```

#### ⏰ Session Timeout Issues

**Problem**: Google Colab session times out after 12 hours (24 hours with Pro)

**Symptoms:**
- Session disconnects unexpectedly
- "Runtime disconnected" message
- Loss of variables and loaded models
- Training progress lost if not checkpointed

**Prevention Strategies:**

1. **Session Time Monitoring**
   ```python
   import time
   import datetime
   from IPython.display import display, Javascript, HTML
   
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
           print(f"   Limit: {self.session_limit_hours} hours")
           
           if remaining_hours < 2:
               print("🚨 CRITICAL: Less than 2 hours remaining!")
               print("   Recommended actions:")
               print("   1. Save current progress immediately")
               print("   2. Create emergency checkpoint")
               print("   3. Prepare for session restart")
               return "critical"
           elif remaining_hours < 4:
               print("⚠️ WARNING: Less than 4 hours remaining")
               print("   Consider saving progress soon")
               return "warning"
           else:
               print("✅ Session time OK")
               return "ok"
               
       def setup_keepalive(self):
           """Prevent idle timeout by simulating activity"""
           display(Javascript('''
               function ClickConnect(){
                   console.log("Keeping session alive...");
                   var connectButton = document.querySelector("colab-connect-button");
                   if (connectButton) {
                       connectButton.click();
                   }
               }
               setInterval(ClickConnect, 60000); // Every minute
               console.log("Session keepalive activated");
           '''))
           print("🔄 Session keepalive activated")
           
       def estimate_training_time(self, config):
           """Estimate remaining training time"""
           # Rough estimates based on configuration
           samples_per_epoch = config.datasets.sft.max_samples or 10000
           batch_size = config.training.sft.batch_size
           grad_accum = config.training.sft.gradient_accumulation_steps
           
           effective_batch_size = batch_size * grad_accum
           steps_per_epoch = samples_per_epoch // effective_batch_size
           
           # Time estimates (seconds per step, varies by GPU)
           gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
           time_per_step = {
               "T4": 2.0,
               "V100": 1.0, 
               "A100": 0.5
           }.get("T4", 2.0)  # Default to T4 timing
           
           # Calculate stage durations
           sft_time = (steps_per_epoch * config.training.sft.epochs * time_per_step) / 3600
           reward_time = (steps_per_epoch * config.training.reward.epochs * time_per_step) / 3600
           ppo_time = (config.training.ppo.max_steps * time_per_step * 2) / 3600  # PPO is slower
           
           total_time = sft_time + reward_time + ppo_time
           
           print(f"⏱️ Estimated Training Times:")
           print(f"   SFT Stage: {sft_time:.1f} hours")
           print(f"   Reward Stage: {reward_time:.1f} hours")
           print(f"   PPO Stage: {ppo_time:.1f} hours")
           print(f"   Total: {total_time:.1f} hours")
           
           current_time = time.time()
           elapsed_hours = (current_time - self.start_time) / 3600
           remaining_hours = self.session_limit_hours - elapsed_hours
           
           if total_time > remaining_hours:
               print(f"⚠️ Training time ({total_time:.1f}h) exceeds remaining session time ({remaining_hours:.1f}h)")
               print("💡 Recommendations:")
               print("   • Reduce max_samples in config")
               print("   • Use smaller batch sizes with more gradient accumulation")
               print("   • Run stages separately across multiple sessions")
               print("   • Consider upgrading to Colab Pro for longer sessions")
           else:
               print("✅ Training should complete within current session")
               
           return {
               "sft_hours": sft_time,
               "reward_hours": reward_time,
               "ppo_hours": ppo_time,
               "total_hours": total_time,
               "fits_in_session": total_time <= remaining_hours
           }
   
   # Initialize session manager
   session_manager = SessionManager()
   session_manager.get_session_info()
   session_manager.setup_keepalive()
   ```

2. **Smart Training Scheduling**
   ```python
   def create_session_aware_config(base_config, remaining_hours):
       """Adjust configuration based on remaining session time"""
       config = base_config.copy()
       
       if remaining_hours < 2:
           # Emergency mode - minimal training
           config.training.sft.epochs = 1
           config.training.sft.max_steps = 100
           config.training.reward.max_steps = 50
           config.training.ppo.max_steps = 50
           config.datasets.sft.max_samples = 500
           config.datasets.preference.max_samples = 250
           print("🚨 Emergency mode: Minimal training configuration")
           
       elif remaining_hours < 4:
           # Quick mode - reduced training
           config.training.sft.epochs = 2
           config.training.sft.max_steps = 300
           config.training.reward.max_steps = 150
           config.training.ppo.max_steps = 150
           config.datasets.sft.max_samples = 1500
           config.datasets.preference.max_samples = 750
           print("⚠️ Quick mode: Reduced training configuration")
           
       else:
           # Normal mode - full training
           print("✅ Normal mode: Full training configuration")
           
       return config
   ```

**Recovery Solutions:**

1. **Automatic Checkpoint Resume**
   ```python
   def resume_training_from_checkpoint():
       """Resume training from the latest checkpoint"""
       from rlhf_phi3 import TrainingOrchestrator, CheckpointManager
       
       print("🔄 Attempting to resume training from checkpoint...")
       
       # Load configuration from Drive
       config_path = "/content/drive/MyDrive/rlhf-phi3/my_config.yaml"
       if os.path.exists(config_path):
           config = Config.from_yaml(config_path)
           print(f"✅ Configuration loaded from: {config_path}")
       else:
           print("❌ No saved configuration found. Using default Colab config.")
           config = Config.from_yaml("configs/colab_config.yaml")
       
       # Find latest checkpoint
       checkpoint_manager = CheckpointManager(config.paths.base_output_dir)
       
       # Check each stage for checkpoints
       stages = ["ppo", "reward", "sft"]  # Check in reverse order
       latest_checkpoint = None
       resume_stage = None
       
       for stage in stages:
           checkpoints = checkpoint_manager.list_checkpoints(stage)
           if checkpoints:
               latest_checkpoint = checkpoints[-1]  # Most recent
               resume_stage = stage
               break
       
       if latest_checkpoint:
           print(f"📂 Found checkpoint: {latest_checkpoint}")
           print(f"🎯 Resuming from {resume_stage} stage")
           
           orchestrator = TrainingOrchestrator(config)
           
           try:
               if resume_stage == "sft":
                   final_model = orchestrator.resume_sft_training(latest_checkpoint)
               elif resume_stage == "reward":
                   final_model = orchestrator.resume_reward_training(latest_checkpoint)
               elif resume_stage == "ppo":
                   final_model = orchestrator.resume_ppo_training(latest_checkpoint)
                   
               print(f"🎉 Training resumed successfully!")
               return final_model
               
           except Exception as e:
               print(f"❌ Resume failed: {e}")
               print("💡 Try starting from an earlier checkpoint or beginning fresh")
               return None
       else:
           print("❌ No checkpoints found. Starting training from beginning.")
           return None
   
   # Attempt resume
   resumed_model = resume_training_from_checkpoint()
   ```

2. **Emergency Checkpoint Creation**
   ```python
   def create_emergency_checkpoint(model, optimizer, step, stage):
       """Create emergency checkpoint when session is about to timeout"""
       import datetime
       
       timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
       emergency_dir = f"/content/drive/MyDrive/rlhf-phi3/emergency_checkpoints"
       os.makedirs(emergency_dir, exist_ok=True)
       
       checkpoint_path = f"{emergency_dir}/emergency_{stage}_{timestamp}"
       
       try:
           # Save model state
           model.save_pretrained(f"{checkpoint_path}/model")
           
           # Save optimizer state
           torch.save({
               'optimizer_state_dict': optimizer.state_dict(),
               'step': step,
               'stage': stage,
               'timestamp': timestamp,
               'session_info': session_manager.get_session_info()
           }, f"{checkpoint_path}/optimizer.pt")
           
           # Save configuration
           config.save_yaml(f"{checkpoint_path}/config.yaml")
           
           print(f"🚨 Emergency checkpoint created: {checkpoint_path}")
           print("💾 All progress saved to Google Drive")
           
           return checkpoint_path
           
       except Exception as e:
           print(f"❌ Emergency checkpoint failed: {e}")
           return None
   ```

**Session Planning Guide:**

1. **Pre-Training Session Assessment**
   ```python
   def plan_training_session():
       """Plan training based on available session time"""
       session_info = session_manager.get_session_info()
       time_estimate = session_manager.estimate_training_time(config)
       
       if time_estimate["fits_in_session"]:
           print("✅ Recommended: Run full pipeline")
           return "full_pipeline"
       elif time_estimate["sft_hours"] < session_manager.session_limit_hours - 1:
           print("⚠️ Recommended: Run SFT stage only")
           return "sft_only"
       else:
           print("🚨 Recommended: Use demo mode or reduce dataset size")
           return "demo_mode"
   
   # Get recommendation
   recommendation = plan_training_session()
   ```

2. **Multi-Session Training Strategy**
   ```python
   def create_multi_session_plan(config):
       """Create a plan for training across multiple sessions"""
       print("📋 Multi-Session Training Plan:")
       print("\n🎯 Session 1 (SFT Stage):")
       print("   1. Run SFT training")
       print("   2. Save SFT checkpoint to Drive")
       print("   3. Validate SFT model performance")
       
       print("\n🎯 Session 2 (Reward Model):")
       print("   1. Load SFT checkpoint")
       print("   2. Run reward model training")
       print("   3. Save reward checkpoint to Drive")
       print("   4. Validate reward model")
       
       print("\n🎯 Session 3 (PPO Training):")
       print("   1. Load SFT and reward checkpoints")
       print("   2. Run PPO training")
       print("   3. Save final model")
       print("   4. Run evaluation and publish")
       
       print("\n💡 Tips for Multi-Session Training:")
       print("   • Always verify checkpoints before ending sessions")
       print("   • Keep configuration files in Google Drive")
       print("   • Monitor training progress with W&B")
       print("   • Test checkpoint loading before long training runs")
   ```

```python
# Session management utilities
import time
import datetime
from IPython.display import Javascript

def get_session_info():
    """Get current session duration and remaining time estimate"""
    # Note: This is an approximation as Colab doesn't expose exact session start time
    import psutil
    boot_time = psutil.boot_time()
    current_time = time.time()
    uptime_hours = (current_time - boot_time) / 3600
    
    print(f"⏱️  Estimated session duration: {uptime_hours:.1f} hours")
    
    if uptime_hours > 10:
        print("⚠️  Session approaching 12-hour limit. Consider saving and resuming.")
        return True
    return False

def session_keepalive():
    """Prevent session timeout by simulating activity"""
    display(Javascript('''
        function ClickConnect(){
            console.log("Keeping session alive...");
            document.querySelector("colab-toolbar-button#connect").click()
        }
        setInterval(ClickConnect, 60000)
    '''))
    print("🔄 Session keepalive activated (clicks connect every minute)")

# Resume from checkpoint
def resume_training_from_checkpoint():
    """Resume training from the latest checkpoint"""
    from rlhf_phi3 import TrainingOrchestrator, CheckpointManager
    
    # Load configuration
    config = Config.from_yaml("/content/drive/MyDrive/rlhf-phi3/my_config.yaml")
    
    # Find latest checkpoint
    checkpoint_manager = CheckpointManager(config.paths.checkpoint_dir)
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint("sft")  # or "reward", "ppo"
    
    if latest_checkpoint:
        print(f"📂 Found checkpoint: {latest_checkpoint}")
        orchestrator = TrainingOrchestrator(config)
        final_model = orchestrator.resume_from_stage(
            stage="sft",  # Adjust based on your checkpoint
            checkpoint_path=latest_checkpoint
        )
        return final_model
    else:
        print("❌ No checkpoints found. Starting from beginning.")
        return None

# Check session status
if get_session_info():
    session_keepalive()
```

**Session Planning Guide:**
```python
def estimate_training_time(config):
    """Estimate training time for each stage"""
    
    # Rough estimates based on configuration
    samples_per_epoch = config.datasets.max_samples or 10000
    batch_size = config.training.batch_size
    grad_accum = config.training.gradient_accumulation_steps
    
    effective_batch_size = batch_size * grad_accum
    steps_per_epoch = samples_per_epoch // effective_batch_size
    
    # Time estimates (seconds per step, varies by GPU)
    time_per_step = {
        "T4": 2.0,    # seconds
        "V100": 1.0,
        "A100": 0.5
    }
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    step_time = time_per_step.get("T4", 2.0)  # Default to T4 timing
    
    # Calculate stage durations
    sft_time = (steps_per_epoch * config.training.sft_epochs * step_time) / 3600
    reward_time = (steps_per_epoch * config.training.reward_epochs * step_time) / 3600
    ppo_time = (config.training.ppo_steps * step_time * 2) / 3600  # PPO is slower
    
    total_time = sft_time + reward_time + ppo_time
    
    print(f"⏱️  Estimated Training Times:")
    print(f"   SFT Stage: {sft_time:.1f} hours")
    print(f"   Reward Stage: {reward_time:.1f} hours") 
    print(f"   PPO Stage: {ppo_time:.1f} hours")
    print(f"   Total: {total_time:.1f} hours")
    
    if total_time > 12:
        print("⚠️  Training may exceed 12-hour Colab limit. Consider:")
        print("   • Reducing max_samples")
        print("   • Using smaller batch sizes with more gradient accumulation")
        print("   • Running stages separately across multiple sessions")
    
    return {
        "sft_hours": sft_time,
        "reward_hours": reward_time,
        "ppo_hours": ppo_time,
        "total_hours": total_time
    }

# Run estimation
estimate_training_time(config)
```

#### 🔐 Authentication Issues

**Problem**: Google Drive, Weights & Biases, or HuggingFace authentication failures

**Symptoms:**
- "Authentication failed" errors
- "Permission denied" messages
- Unable to save checkpoints to Drive
- Cannot log metrics to W&B
- Model publishing fails

**Google Drive Authentication:**

1. **Basic Drive Mounting Issues**
   ```python
   from google.colab import drive, auth
   import os
   
   def fix_drive_mounting():
       """Comprehensive Drive mounting with error handling"""
       print("🔄 Attempting Google Drive mount...")
       
       try:
           # Force remount with authentication
           drive.mount('/content/drive', force_remount=True)
           
           # Verify mount
           if os.path.exists('/content/drive/MyDrive'):
               print("✅ Google Drive mounted successfully")
               
               # Test write permissions
               test_file = '/content/drive/MyDrive/test_write.txt'
               try:
                   with open(test_file, 'w') as f:
                       f.write("test")
                   os.remove(test_file)
                   print("✅ Write permissions confirmed")
                   return True
               except Exception as e:
                   print(f"❌ Write permission test failed: {e}")
                   return False
           else:
               print("❌ Drive mount failed - directory not accessible")
               return False
               
       except Exception as e:
           print(f"❌ Drive mounting failed: {e}")
           print("💡 Trying alternative authentication...")
           
           try:
               # Alternative: Manual authentication
               auth.authenticate_user()
               drive.mount('/content/drive', force_remount=True)
               
               if os.path.exists('/content/drive/MyDrive'):
                   print("✅ Alternative authentication successful")
                   return True
               else:
                   print("❌ Alternative authentication also failed")
                   return False
                   
           except Exception as e2:
               print(f"❌ Alternative authentication failed: {e2}")
               return False
   
   # Fix drive mounting
   if not fix_drive_mounting():
       print("🚨 Drive mounting failed. Using local storage as fallback.")
       print("⚠️ WARNING: Checkpoints will be lost when session ends!")
   ```

2. **Drive Permission Issues**
   ```python
   def check_drive_permissions():
       """Check and fix Drive permissions"""
       drive_path = '/content/drive/MyDrive'
       
       if not os.path.exists(drive_path):
           print("❌ Google Drive not mounted")
           return False
           
       # Check if we can create directories
       test_dir = f"{drive_path}/rlhf-phi3-test"
       try:
           os.makedirs(test_dir, exist_ok=True)
           os.rmdir(test_dir)
           print("✅ Directory creation permissions OK")
       except Exception as e:
           print(f"❌ Directory creation failed: {e}")
           return False
           
       # Check file operations
       test_file = f"{drive_path}/test_permissions.txt"
       try:
           # Write test
           with open(test_file, 'w') as f:
               f.write("permission test")
           
           # Read test
           with open(test_file, 'r') as f:
               content = f.read()
               
           # Delete test
           os.remove(test_file)
           
           print("✅ File operation permissions OK")
           return True
           
       except Exception as e:
           print(f"❌ File operation failed: {e}")
           return False
   
   check_drive_permissions()
   ```

**Weights & Biases Authentication:**

1. **W&B Login Issues**
   ```python
   import wandb
   import os
   from getpass import getpass
   
   def fix_wandb_authentication():
       """Comprehensive W&B authentication with error handling"""
       print("🔄 Setting up Weights & Biases authentication...")
       
       # Method 1: Try existing API key
       if 'WANDB_API_KEY' in os.environ:
           try:
               wandb.login(key=os.environ['WANDB_API_KEY'])
               print("✅ W&B authenticated with existing API key")
               return True
           except Exception as e:
               print(f"❌ Existing API key failed: {e}")
       
       # Method 2: Interactive login
       try:
           wandb.login()
           print("✅ W&B interactive login successful")
           return True
       except Exception as e:
           print(f"❌ Interactive login failed: {e}")
       
       # Method 3: Manual API key entry
       print("\n🔑 Manual API Key Setup:")
       print("1. Go to https://wandb.ai/settings")
       print("2. Copy your API key")
       print("3. Paste it below")
       
       api_key = getpass("Enter your W&B API key: ")
       
       try:
           wandb.login(key=api_key)
           os.environ['WANDB_API_KEY'] = api_key
           print("✅ Manual API key authentication successful")
           return True
       except Exception as e:
           print(f"❌ Manual API key failed: {e}")
           return False
   
   def test_wandb_functionality():
       """Test W&B functionality"""
       try:
           # Test run creation
           test_run = wandb.init(
               project="test-project",
               name="authentication-test",
               mode="disabled"  # Don't actually log
           )
           test_run.finish()
           print("✅ W&B functionality test passed")
           return True
       except Exception as e:
           print(f"❌ W&B functionality test failed: {e}")
           return False
   
   # Fix W&B authentication
   if fix_wandb_authentication():
       test_wandb_functionality()
   ```

2. **W&B Project Access Issues**
   ```python
   def check_wandb_project_access(project_name, entity=None):
       """Check if we can access the specified W&B project"""
       try:
           # Try to initialize a test run
           run = wandb.init(
               project=project_name,
               entity=entity,
               name="access-test",
               mode="disabled"
           )
           run.finish()
           print(f"✅ Access to project '{project_name}' confirmed")
           return True
       except Exception as e:
           print(f"❌ Cannot access project '{project_name}': {e}")
           print("💡 Solutions:")
           print("   1. Check project name spelling")
           print("   2. Verify you have access to the project")
           print("   3. Create the project if it doesn't exist")
           print("   4. Check entity/team name if specified")
           return False
   
   # Test project access
   check_wandb_project_access("rlhf-phi3-pipeline")
   ```

**HuggingFace Authentication:**

1. **HF Token Issues**
   ```python
   from huggingface_hub import login, logout, whoami
   import os
   from getpass import getpass
   
   def fix_huggingface_authentication():
       """Comprehensive HuggingFace authentication"""
       print("🔄 Setting up HuggingFace authentication...")
       
       # Method 1: Try existing token
       if 'HUGGINGFACE_TOKEN' in os.environ:
           try:
               login(token=os.environ['HUGGINGFACE_TOKEN'])
               user_info = whoami()
               print(f"✅ HF authenticated as: {user_info['name']}")
               return True
           except Exception as e:
               print(f"❌ Existing token failed: {e}")
       
       # Method 2: Interactive login
       try:
           login()
           user_info = whoami()
           print(f"✅ HF interactive login successful as: {user_info['name']}")
           return True
       except Exception as e:
           print(f"❌ Interactive login failed: {e}")
       
       # Method 3: Manual token entry
       print("\n🔑 Manual Token Setup:")
       print("1. Go to https://huggingface.co/settings/tokens")
       print("2. Create a new token with 'write' permissions")
       print("3. Copy the token")
       print("4. Paste it below")
       
       token = getpass("Enter your HuggingFace token: ")
       
       try:
           login(token=token)
           os.environ['HUGGINGFACE_TOKEN'] = token
           user_info = whoami()
           print(f"✅ Manual token authentication successful as: {user_info['name']}")
           return True
       except Exception as e:
           print(f"❌ Manual token failed: {e}")
           return False
   
   def test_huggingface_functionality():
       """Test HuggingFace functionality"""
       try:
           from huggingface_hub import HfApi
           api = HfApi()
           
           # Test API access
           user_info = api.whoami()
           print(f"✅ HF API access confirmed for: {user_info['name']}")
           
           # Test model access
           model_info = api.model_info("microsoft/Phi-3-mini-4k-instruct")
           print("✅ Model access confirmed")
           
           return True
       except Exception as e:
           print(f"❌ HF functionality test failed: {e}")
           return False
   
   # Fix HF authentication
   if fix_huggingface_authentication():
       test_huggingface_functionality()
   ```

2. **Token Permission Issues**
   ```python
   def check_token_permissions():
       """Check HuggingFace token permissions"""
       try:
           from huggingface_hub import HfApi
           api = HfApi()
           
           # Get token info
           token_info = api.whoami()
           
           print("🔍 Token Information:")
           print(f"   Username: {token_info.get('name', 'Unknown')}")
           print(f"   Email: {token_info.get('email', 'Unknown')}")
           
           # Check if token has write permissions
           auth_header = api._get_token_headers()
           if auth_header:
               print("✅ Token has authentication headers")
           else:
               print("❌ Token missing authentication headers")
               
           # Try to access user repositories
           try:
               repos = api.list_repos(token=api.token, type="model")
               print(f"✅ Can access repositories ({len(list(repos))} found)")
           except Exception as e:
               print(f"❌ Cannot access repositories: {e}")
               
           return True
           
       except Exception as e:
           print(f"❌ Token permission check failed: {e}")
           print("💡 Solutions:")
           print("   1. Regenerate token with 'write' permissions")
           print("   2. Check token hasn't expired")
           print("   3. Verify account is in good standing")
           return False
   
   check_token_permissions()
   ```

**Authentication Recovery Procedures:**

1. **Complete Authentication Reset**
   ```python
   def reset_all_authentication():
       """Reset all authentication and start fresh"""
       print("🔄 Resetting all authentication...")
       
       # Clear environment variables
       auth_vars = ['WANDB_API_KEY', 'HUGGINGFACE_TOKEN', 'HF_TOKEN']
       for var in auth_vars:
           if var in os.environ:
               del os.environ[var]
               print(f"   Cleared {var}")
       
       # Logout from services
       try:
           wandb.finish()
           print("   Logged out of W&B")
       except:
           pass
           
       try:
           from huggingface_hub import logout
           logout()
           print("   Logged out of HuggingFace")
       except:
           pass
       
       # Remount Drive
       try:
           from google.colab import drive
           drive.mount('/content/drive', force_remount=True)
           print("   Remounted Google Drive")
       except:
           pass
       
       print("✅ Authentication reset complete")
       print("💡 Now run the setup cells again to re-authenticate")
   
   # Uncomment to reset authentication
   # reset_all_authentication()
   ```

2. **Authentication Status Dashboard**
   ```python
   def check_all_authentication():
       """Check status of all authentication services"""
       print("🔍 Authentication Status Dashboard")
       print("=" * 50)
       
       # Google Drive
       print("\n📁 Google Drive:")
       if os.path.exists('/content/drive/MyDrive'):
           print("   ✅ Mounted and accessible")
           try:
               test_file = '/content/drive/MyDrive/.auth_test'
               with open(test_file, 'w') as f:
                   f.write('test')
               os.remove(test_file)
               print("   ✅ Write permissions confirmed")
           except:
               print("   ❌ Write permissions failed")
       else:
           print("   ❌ Not mounted or inaccessible")
       
       # Weights & Biases
       print("\n📊 Weights & Biases:")
       try:
           import wandb
           if wandb.api.api_key:
               print("   ✅ API key configured")
               # Test with a dummy run
               run = wandb.init(project="test", mode="disabled")
               run.finish()
               print("   ✅ Functionality confirmed")
           else:
               print("   ❌ No API key found")
       except Exception as e:
           print(f"   ❌ Error: {e}")
       
       # HuggingFace
       print("\n🤗 HuggingFace:")
       try:
           from huggingface_hub import whoami
           user_info = whoami()
           print(f"   ✅ Authenticated as: {user_info['name']}")
           print("   ✅ Functionality confirmed")
       except Exception as e:
           print(f"   ❌ Error: {e}")
       
       print("\n" + "=" * 50)
   
   # Run authentication check
   check_all_authentication()
   ```

**Verifying Authentication**:
```python
# Test Google Drive access
import os
if os.path.exists('/content/drive/MyDrive'):
    print("✅ Google Drive mounted successfully")
else:
    print("❌ Google Drive not accessible")

# Test W&B authentication
try:
    import wandb
    wandb.init(project="test", mode="disabled")
    print("✅ W&B authentication successful")
except Exception as e:
    print(f"❌ W&B authentication failed: {e}")

# Test HuggingFace authentication
try:
    from huggingface_hub import whoami
    user = whoami()
    print(f"✅ HuggingFace authenticated as: {user['name']}")
except Exception as e:
    print(f"❌ HuggingFace authentication failed: {e}")
```

#### 📊 Dataset Loading Issues

**Problem**: Dataset download failures, corruption, or format errors

**Symptoms:**
- `ConnectionError` during dataset download
- `DatasetNotFoundError` for specified datasets
- Corrupted or incomplete dataset files
- Format validation failures
- Slow download speeds or timeouts

**Common Dataset Issues:**

1. **Network and Download Problems**
   ```python
   import requests
   from datasets import load_dataset
   import time
   
   def test_network_connectivity():
       """Test network connectivity to HuggingFace Hub"""
       print("🌐 Testing network connectivity...")
       
       try:
           # Test basic internet
           response = requests.get("https://www.google.com", timeout=10)
           print("✅ Basic internet connectivity OK")
       except Exception as e:
           print(f"❌ No internet connectivity: {e}")
           return False
       
       try:
           # Test HuggingFace Hub
           response = requests.get("https://huggingface.co", timeout=10)
           print("✅ HuggingFace Hub accessible")
       except Exception as e:
           print(f"❌ HuggingFace Hub not accessible: {e}")
           return False
       
       try:
           # Test datasets API
           response = requests.get("https://datasets-server.huggingface.co", timeout=10)
           print("✅ Datasets server accessible")
           return True
       except Exception as e:
           print(f"❌ Datasets server not accessible: {e}")
           return False
   
   def robust_dataset_loading(dataset_name, split="train", max_retries=3, streaming=True):
       """Load dataset with retry logic and error handling"""
       print(f"📥 Loading dataset: {dataset_name}")
       
       for attempt in range(max_retries):
           try:
               print(f"   Attempt {attempt + 1}/{max_retries}")
               
               # Load with streaming for memory efficiency
               dataset = load_dataset(
                   dataset_name,
                   split=split,
                   streaming=streaming,
                   trust_remote_code=True  # Some datasets require this
               )
               
               # Test that we can iterate over the dataset
               sample = next(iter(dataset))
               print(f"✅ Dataset loaded successfully")
               print(f"   Sample keys: {list(sample.keys())}")
               return dataset
               
           except Exception as e:
               print(f"❌ Attempt {attempt + 1} failed: {e}")
               if attempt < max_retries - 1:
                   wait_time = 2 ** attempt  # Exponential backoff
                   print(f"   Waiting {wait_time} seconds before retry...")
                   time.sleep(wait_time)
               else:
                   print("❌ All attempts failed")
                   return None
   
   # Test network and load datasets
   if test_network_connectivity():
       sft_dataset = robust_dataset_loading("HuggingFaceH4/ultrachat_200k", "train_sft")
       pref_dataset = robust_dataset_loading("HuggingFaceH4/ultrafeedback_binarized", "train_prefs")
   ```

2. **Dataset Cache Management**
   ```python
   import shutil
   import os
   from datasets import config
   
   def manage_dataset_cache():
       """Manage HuggingFace dataset cache"""
       cache_dir = config.HF_DATASETS_CACHE
       print(f"📂 Dataset cache directory: {cache_dir}")
       
       if os.path.exists(cache_dir):
           # Check cache size
           cache_size = sum(
               os.path.getsize(os.path.join(dirpath, filename))
               for dirpath, dirnames, filenames in os.walk(cache_dir)
               for filename in filenames
           ) / (1024**3)  # Convert to GB
           
           print(f"💾 Cache size: {cache_size:.2f} GB")
           
           if cache_size > 10:  # If cache is larger than 10GB
               print("⚠️ Large cache detected. Consider cleaning.")
               
               response = input("Clear dataset cache? (y/n): ")
               if response.lower() == 'y':
                   shutil.rmtree(cache_dir, ignore_errors=True)
                   print("✅ Cache cleared")
               else:
                   print("Cache kept")
           else:
               print("✅ Cache size is reasonable")
       else:
           print("📂 No cache directory found (will be created)")
   
   def clear_specific_dataset_cache(dataset_name):
       """Clear cache for a specific dataset"""
       cache_dir = config.HF_DATASETS_CACHE
       
       # Find dataset-specific cache files
       dataset_cache_files = []
       if os.path.exists(cache_dir):
           for root, dirs, files in os.walk(cache_dir):
               for file in files:
                   if dataset_name.replace("/", "___") in file:
                       dataset_cache_files.append(os.path.join(root, file))
       
       if dataset_cache_files:
           print(f"🗑️ Found {len(dataset_cache_files)} cache files for {dataset_name}")
           for file in dataset_cache_files:
               try:
                   os.remove(file)
                   print(f"   Removed: {os.path.basename(file)}")
               except Exception as e:
                   print(f"   Failed to remove {file}: {e}")
           print("✅ Dataset cache cleared")
       else:
           print(f"No cache files found for {dataset_name}")
   
   # Manage cache
   manage_dataset_cache()
   ```

3. **Alternative Dataset Sources**
   ```python
   def get_alternative_datasets():
       """Provide alternative datasets if primary ones fail"""
       alternatives = {
           "sft_datasets": [
               {
                   "name": "HuggingFaceH4/ultrachat_200k",
                   "split": "train_sft",
                   "description": "Primary SFT dataset"
               },
               {
                   "name": "tatsu-lab/alpaca",
                   "split": "train",
                   "description": "Smaller alternative (52K samples)"
               },
               {
                   "name": "Open-Orca/OpenOrca",
                   "split": "train",
                   "description": "Large instruction dataset (4M samples)"
               }
           ],
           "preference_datasets": [
               {
                   "name": "HuggingFaceH4/ultrafeedback_binarized",
                   "split": "train_prefs",
                   "description": "Primary preference dataset"
               },
               {
                   "name": "Anthropic/hh-rlhf",
                   "split": "train",
                   "description": "Anthropic's helpful/harmless dataset"
               },
               {
                   "name": "lvwerra/stack-exchange-paired",
                   "split": "train",
                   "description": "Stack Exchange Q&A pairs"
               }
           ]
       }
       
       return alternatives
   
   def try_alternative_datasets(dataset_type="sft"):
       """Try loading alternative datasets"""
       alternatives = get_alternative_datasets()
       datasets_to_try = alternatives.get(f"{dataset_type}_datasets", [])
       
       print(f"🔄 Trying alternative {dataset_type} datasets...")
       
       for dataset_info in datasets_to_try:
           print(f"\n📥 Trying: {dataset_info['name']}")
           print(f"   Description: {dataset_info['description']}")
           
           try:
               dataset = load_dataset(
                   dataset_info['name'],
                   split=dataset_info['split'],
                   streaming=True
               )
               
               # Test dataset
               sample = next(iter(dataset))
               print(f"✅ Successfully loaded {dataset_info['name']}")
               print(f"   Sample keys: {list(sample.keys())}")
               return dataset, dataset_info['name']
               
           except Exception as e:
               print(f"❌ Failed to load {dataset_info['name']}: {e}")
               continue
       
       print(f"❌ All alternative {dataset_type} datasets failed")
       return None, None
   
   # Try alternatives if needed
   # sft_dataset, sft_name = try_alternative_datasets("sft")
   # pref_dataset, pref_name = try_alternative_datasets("preference")
   ```

**Dataset Validation and Preprocessing:**

1. **Dataset Format Validation**
   ```python
   def validate_sft_dataset(dataset):
       """Validate SFT dataset format"""
       print("🔍 Validating SFT dataset format...")
       
       try:
           sample = next(iter(dataset))
           
           # Check required fields
           required_fields = ['messages']  # For chat format
           alternative_fields = ['instruction', 'input', 'output']  # For instruction format
           
           has_chat_format = all(field in sample for field in required_fields)
           has_instruction_format = all(field in sample for field in alternative_fields)
           
           if has_chat_format:
               print("✅ Dataset has chat format (messages field)")
               
               # Validate messages structure
               messages = sample['messages']
               if isinstance(messages, list) and len(messages) > 0:
                   print("✅ Messages field is properly formatted")
                   
                   # Check message structure
                   first_message = messages[0]
                   if isinstance(first_message, dict) and 'role' in first_message and 'content' in first_message:
                       print("✅ Message structure is valid")
                       return True
                   else:
                       print("❌ Invalid message structure")
                       return False
               else:
                   print("❌ Messages field is not a list or is empty")
                   return False
                   
           elif has_instruction_format:
               print("✅ Dataset has instruction format")
               print("💡 Will convert to chat format during preprocessing")
               return True
           else:
               print("❌ Dataset format not recognized")
               print(f"   Available fields: {list(sample.keys())}")
               return False
               
       except Exception as e:
           print(f"❌ Dataset validation failed: {e}")
           return False
   
   def validate_preference_dataset(dataset):
       """Validate preference dataset format"""
       print("🔍 Validating preference dataset format...")
       
       try:
           sample = next(iter(dataset))
           
           # Check required fields for preference data
           required_fields = ['chosen', 'rejected']
           alternative_fields = ['prompt', 'chosen', 'rejected']
           
           has_basic_format = all(field in sample for field in required_fields)
           has_full_format = all(field in sample for field in alternative_fields)
           
           if has_full_format:
               print("✅ Dataset has full preference format (prompt, chosen, rejected)")
               return True
           elif has_basic_format:
               print("✅ Dataset has basic preference format (chosen, rejected)")
               print("💡 Will extract prompts during preprocessing")
               return True
           else:
               print("❌ Dataset format not recognized")
               print(f"   Available fields: {list(sample.keys())}")
               print("   Required: 'chosen' and 'rejected' fields")
               return False
               
       except Exception as e:
           print(f"❌ Dataset validation failed: {e}")
           return False
   
   # Validate datasets
   # if sft_dataset:
   #     validate_sft_dataset(sft_dataset)
   # if pref_dataset:
   #     validate_preference_dataset(pref_dataset)
   ```

2. **Dataset Preprocessing Troubleshooting**
   ```python
   def debug_dataset_preprocessing(dataset, dataset_type="sft"):
       """Debug dataset preprocessing issues"""
       print(f"🔧 Debugging {dataset_type} dataset preprocessing...")
       
       try:
           # Get multiple samples for analysis
           samples = []
           for i, sample in enumerate(dataset):
               samples.append(sample)
               if i >= 4:  # Get 5 samples
                   break
           
           print(f"📊 Analyzed {len(samples)} samples")
           
           # Analyze sample structure
           for i, sample in enumerate(samples):
               print(f"\n📝 Sample {i + 1}:")
               print(f"   Keys: {list(sample.keys())}")
               
               for key, value in sample.items():
                   if isinstance(value, str):
                       print(f"   {key}: '{value[:100]}...' (length: {len(value)})")
                   elif isinstance(value, list):
                       print(f"   {key}: list with {len(value)} items")
                       if value and isinstance(value[0], dict):
                           print(f"      First item keys: {list(value[0].keys())}")
                   else:
                       print(f"   {key}: {type(value)} - {str(value)[:50]}...")
           
           return samples
           
       except Exception as e:
           print(f"❌ Preprocessing debug failed: {e}")
           return None
   
   def test_tokenization(samples, tokenizer_name="microsoft/Phi-3-mini-4k-instruct"):
       """Test tokenization on sample data"""
       print("🔤 Testing tokenization...")
       
       try:
           from transformers import AutoTokenizer
           tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
           
           if tokenizer.pad_token is None:
               tokenizer.pad_token = tokenizer.eos_token
           
           for i, sample in enumerate(samples[:2]):  # Test first 2 samples
               print(f"\n🧪 Testing sample {i + 1}:")
               
               # Extract text based on format
               if 'messages' in sample:
                   # Chat format
                   text = tokenizer.apply_chat_template(
                       sample['messages'],
                       tokenize=False,
                       add_generation_prompt=False
                   )
               elif 'instruction' in sample:
                   # Instruction format
                   text = f"Instruction: {sample['instruction']}\nResponse: {sample.get('output', '')}"
               else:
                   print("   ❌ Unknown format for tokenization")
                   continue
               
               # Tokenize
               tokens = tokenizer(
                   text,
                   truncation=True,
                   padding=False,
                   max_length=2048
               )
               
               print(f"   Text length: {len(text)} characters")
               print(f"   Token count: {len(tokens['input_ids'])}")
               print(f"   First 10 tokens: {tokens['input_ids'][:10]}")
               
           print("✅ Tokenization test completed")
           
       except Exception as e:
           print(f"❌ Tokenization test failed: {e}")
   
   # Debug preprocessing
   # if sft_dataset:
   #     sft_samples = debug_dataset_preprocessing(sft_dataset, "sft")
   #     if sft_samples:
   #         test_tokenization(sft_samples)
   ```

**Dataset Recovery Procedures:**

1. **Offline Dataset Preparation**
   ```python
   def prepare_offline_datasets():
       """Prepare datasets for offline use"""
       print("💾 Preparing datasets for offline use...")
       
       datasets_to_download = [
           ("HuggingFaceH4/ultrachat_200k", "train_sft"),
           ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs"),
           ("tatsu-lab/alpaca", "train")  # Backup
       ]
       
       download_dir = "/content/drive/MyDrive/rlhf-phi3/datasets"
       os.makedirs(download_dir, exist_ok=True)
       
       for dataset_name, split in datasets_to_download:
           try:
               print(f"📥 Downloading {dataset_name} ({split})...")
               
               # Load and save dataset
               dataset = load_dataset(dataset_name, split=split)
               
               # Save to disk
               save_path = f"{download_dir}/{dataset_name.replace('/', '_')}_{split}"
               dataset.save_to_disk(save_path)
               
               print(f"✅ Saved to: {save_path}")
               
           except Exception as e:
               print(f"❌ Failed to download {dataset_name}: {e}")
   
   def load_offline_dataset(dataset_path):
       """Load dataset from offline storage"""
       try:
           from datasets import load_from_disk
           dataset = load_from_disk(dataset_path)
           print(f"✅ Loaded offline dataset from: {dataset_path}")
           return dataset
       except Exception as e:
           print(f"❌ Failed to load offline dataset: {e}")
           return None
   
   # Prepare offline datasets if needed
   # prepare_offline_datasets()
   ```

```python
# Clear HuggingFace cache
import shutil
shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)

# Use streaming datasets for large datasets
config.datasets.streaming = True
config.datasets.max_samples = 10000  # Limit samples for testing

# Alternative datasets for testing
config.datasets.sft_dataset = "tatsu-lab/alpaca"  # Smaller alternative
config.datasets.preference_dataset = "Anthropic/hh-rlhf"  # Alternative preference dataset
```

**Dataset Troubleshooting**:
- **Network timeouts**: Increase timeout values
- **Disk space**: Ensure sufficient space for downloads
- **Format validation**: Check dataset format compatibility
- **Access permissions**: Verify dataset access rights

**Testing Dataset Loading**:
```python
from datasets import load_dataset

def test_dataset_loading(dataset_name, split="train"):
    try:
        # Test with small sample first
        dataset = load_dataset(dataset_name, split=f"{split}[:100]", streaming=True)
        sample = next(iter(dataset))
        print(f"✅ Dataset {dataset_name} loaded successfully")
        print(f"   Sample keys: {list(sample.keys())}")
        return True
    except Exception as e:
        print(f"❌ Dataset {dataset_name} failed to load: {e}")
        return False

# Test your datasets
test_dataset_loading("HuggingFaceH4/ultrachat_200k")
test_dataset_loading("HuggingFaceH4/ultrafeedback_binarized")
```

#### 🔄 Training Convergence Issues

**Problem**: Loss not decreasing, training unstable, or poor model performance

**Symptoms:**
- Training loss plateaus or increases
- Loss becomes NaN or infinity
- Gradient norms explode or vanish
- Model outputs become repetitive or nonsensical
- Evaluation metrics don't improve

**Convergence Diagnostics:**

1. **Training Metrics Analysis**
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   
   def analyze_training_metrics(run_id=None):
       """Analyze training metrics from W&B or local logs"""
       print("📊 Analyzing training metrics...")
       
       if run_id:
           # Load from W&B
           try:
               import wandb
               api = wandb.Api()
               run = api.run(f"your-entity/rlhf-phi3-pipeline/{run_id}")
               
               history = run.scan_history(keys=[
                   "train/loss", "train/learning_rate", "train/grad_norm",
                   "eval/loss", "train/epoch"
               ])
               
               metrics = {
                   "steps": [],
                   "train_loss": [],
                   "eval_loss": [],
                   "learning_rate": [],
                   "grad_norm": []
               }
               
               for row in history:
                   if row.get("train/loss") is not None:
                       metrics["steps"].append(row["_step"])
                       metrics["train_loss"].append(row["train/loss"])
                       metrics["eval_loss"].append(row.get("eval/loss"))
                       metrics["learning_rate"].append(row.get("train/learning_rate", 0))
                       metrics["grad_norm"].append(row.get("train/grad_norm", 0))
               
           except Exception as e:
               print(f"❌ Failed to load W&B metrics: {e}")
               return None
       else:
           print("💡 Provide run_id to analyze W&B metrics")
           return None
       
       # Analyze metrics
       if metrics["train_loss"]:
           print(f"📈 Training Analysis:")
           print(f"   Steps: {len(metrics['steps'])}")
           print(f"   Initial loss: {metrics['train_loss'][0]:.4f}")
           print(f"   Final loss: {metrics['train_loss'][-1]:.4f}")
           print(f"   Loss reduction: {metrics['train_loss'][0] - metrics['train_loss'][-1]:.4f}")
           
           # Check for convergence issues
           recent_losses = metrics["train_loss"][-10:]  # Last 10 steps
           if len(recent_losses) > 5:
               loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
               
               if loss_trend > 0.01:
                   print("⚠️ Loss is increasing (possible overfitting or high LR)")
               elif abs(loss_trend) < 0.001:
                   print("⚠️ Loss has plateaued (possible underfitting or low LR)")
               else:
                   print("✅ Loss is decreasing normally")
           
           # Check gradient norms
           if metrics["grad_norm"]:
               avg_grad_norm = np.mean([g for g in metrics["grad_norm"] if g > 0])
               max_grad_norm = max([g for g in metrics["grad_norm"] if g > 0])
               
               print(f"   Average gradient norm: {avg_grad_norm:.4f}")
               print(f"   Max gradient norm: {max_grad_norm:.4f}")
               
               if max_grad_norm > 10:
                   print("⚠️ High gradient norms detected (possible exploding gradients)")
               elif avg_grad_norm < 0.001:
                   print("⚠️ Very low gradient norms (possible vanishing gradients)")
       
       return metrics
   
   def plot_training_diagnostics(metrics):
       """Plot comprehensive training diagnostics"""
       if not metrics or not metrics["train_loss"]:
           print("No metrics to plot")
           return
       
       fig, axes = plt.subplots(2, 2, figsize=(15, 10))
       
       # Training loss
       axes[0, 0].plot(metrics["steps"], metrics["train_loss"], label="Train Loss", color="blue")
       if any(metrics["eval_loss"]):
           eval_steps = [s for s, e in zip(metrics["steps"], metrics["eval_loss"]) if e is not None]
           eval_losses = [e for e in metrics["eval_loss"] if e is not None]
           axes[0, 0].plot(eval_steps, eval_losses, label="Eval Loss", color="red")
       axes[0, 0].set_title("Training Loss")
       axes[0, 0].set_xlabel("Steps")
       axes[0, 0].set_ylabel("Loss")
       axes[0, 0].legend()
       axes[0, 0].grid(True)
       
       # Learning rate
       axes[0, 1].plot(metrics["steps"], metrics["learning_rate"], color="green")
       axes[0, 1].set_title("Learning Rate Schedule")
       axes[0, 1].set_xlabel("Steps")
       axes[0, 1].set_ylabel("Learning Rate")
       axes[0, 1].grid(True)
       
       # Gradient norms
       grad_steps = [s for s, g in zip(metrics["steps"], metrics["grad_norm"]) if g > 0]
       grad_norms = [g for g in metrics["grad_norm"] if g > 0]
       if grad_norms:
           axes[1, 0].plot(grad_steps, grad_norms, color="orange")
           axes[1, 0].set_title("Gradient Norms")
           axes[1, 0].set_xlabel("Steps")
           axes[1, 0].set_ylabel("Gradient Norm")
           axes[1, 0].grid(True)
       
       # Loss smoothed (moving average)
       if len(metrics["train_loss"]) > 10:
           window = min(50, len(metrics["train_loss"]) // 10)
           smoothed_loss = np.convolve(metrics["train_loss"], np.ones(window)/window, mode='valid')
           smoothed_steps = metrics["steps"][window-1:]
           axes[1, 1].plot(smoothed_steps, smoothed_loss, color="purple")
           axes[1, 1].set_title(f"Smoothed Loss (window={window})")
           axes[1, 1].set_xlabel("Steps")
           axes[1, 1].set_ylabel("Smoothed Loss")
           axes[1, 1].grid(True)
       
       plt.tight_layout()
       plt.show()
   
   # Example usage:
   # metrics = analyze_training_metrics("your-run-id")
   # if metrics:
   #     plot_training_diagnostics(metrics)
   ```

2. **Hyperparameter Optimization**
   ```python
   def suggest_hyperparameter_fixes(current_config, metrics=None):
       """Suggest hyperparameter adjustments based on training issues"""
       print("🔧 Hyperparameter Optimization Suggestions:")
       
       suggestions = []
       
       # Learning rate analysis
       current_lr = current_config.training.sft.learning_rate
       print(f"\n📚 Learning Rate Analysis (current: {current_lr})")
       
       if metrics and metrics["train_loss"]:
           recent_losses = metrics["train_loss"][-20:]
           if len(recent_losses) > 10:
               loss_variance = np.var(recent_losses)
               loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
               
               if loss_variance > 0.1:
                   suggestions.append({
                       "issue": "High loss variance (oscillating)",
                       "fix": "Reduce learning rate by 50%",
                       "new_value": current_lr * 0.5
                   })
               elif loss_trend > 0.01:
                   suggestions.append({
                       "issue": "Loss increasing",
                       "fix": "Reduce learning rate by 70%",
                       "new_value": current_lr * 0.3
                   })
               elif abs(loss_trend) < 0.001 and recent_losses[-1] > 2.0:
                   suggestions.append({
                       "issue": "Loss plateaued at high value",
                       "fix": "Increase learning rate by 50%",
                       "new_value": current_lr * 1.5
                   })
       
       # Batch size analysis
       current_batch = current_config.training.sft.batch_size
       current_grad_accum = current_config.training.sft.gradient_accumulation_steps
       effective_batch = current_batch * current_grad_accum
       
       print(f"\n📦 Batch Size Analysis:")
       print(f"   Current batch size: {current_batch}")
       print(f"   Gradient accumulation: {current_grad_accum}")
       print(f"   Effective batch size: {effective_batch}")
       
       if effective_batch < 8:
           suggestions.append({
               "issue": "Very small effective batch size",
               "fix": "Increase gradient accumulation steps",
               "new_value": max(8 // current_batch, current_grad_accum * 2)
           })
       elif effective_batch > 64:
           suggestions.append({
               "issue": "Very large effective batch size",
               "fix": "Reduce gradient accumulation or batch size",
               "new_value": min(32 // current_batch, current_grad_accum)
           })
       
       # LoRA configuration analysis
       current_lora_r = current_config.lora.r
       current_lora_alpha = current_config.lora.alpha
       
       print(f"\n🎯 LoRA Configuration Analysis:")
       print(f"   Current rank (r): {current_lora_r}")
       print(f"   Current alpha: {current_lora_alpha}")
       print(f"   Alpha/r ratio: {current_lora_alpha / current_lora_r}")
       
       if current_lora_r < 8:
           suggestions.append({
               "issue": "Very low LoRA rank (limited capacity)",
               "fix": "Increase LoRA rank",
               "new_value": min(16, current_lora_r * 2)
           })
       elif current_lora_alpha / current_lora_r < 1:
           suggestions.append({
               "issue": "Low alpha/rank ratio",
               "fix": "Increase LoRA alpha",
               "new_value": current_lora_r * 2
           })
       
       # Print suggestions
       if suggestions:
           print(f"\n💡 Recommended Fixes:")
           for i, suggestion in enumerate(suggestions, 1):
               print(f"   {i}. {suggestion['issue']}")
               print(f"      Fix: {suggestion['fix']}")
               if 'new_value' in suggestion:
                   print(f"      Suggested value: {suggestion['new_value']}")
       else:
           print("\n✅ No obvious hyperparameter issues detected")
       
       return suggestions
   
   def apply_suggested_fixes(config, suggestions):
       """Apply suggested hyperparameter fixes"""
       print("🔧 Applying suggested fixes...")
       
       for suggestion in suggestions:
           if "learning rate" in suggestion["fix"].lower():
               config.training.sft.learning_rate = suggestion["new_value"]
               config.training.reward.learning_rate = suggestion["new_value"] * 0.5
               config.training.ppo.learning_rate = suggestion["new_value"] * 0.2
               print(f"   Updated learning rates")
               
           elif "gradient accumulation" in suggestion["fix"].lower():
               config.training.sft.gradient_accumulation_steps = suggestion["new_value"]
               config.training.reward.gradient_accumulation_steps = suggestion["new_value"] * 2
               config.training.ppo.gradient_accumulation_steps = suggestion["new_value"] * 4
               print(f"   Updated gradient accumulation steps")
               
           elif "lora rank" in suggestion["fix"].lower():
               config.lora.r = suggestion["new_value"]
               print(f"   Updated LoRA rank to {suggestion['new_value']}")
               
           elif "lora alpha" in suggestion["fix"].lower():
               config.lora.alpha = suggestion["new_value"]
               print(f"   Updated LoRA alpha to {suggestion['new_value']}")
       
       print("✅ Fixes applied to configuration")
       return config
   
   # Example usage:
   # suggestions = suggest_hyperparameter_fixes(config, metrics)
   # if suggestions:
   #     config = apply_suggested_fixes(config, suggestions)
   ```

**Emergency Procedures:**

1. **Session About to Timeout**
   ```python
   def emergency_session_save(model=None, optimizer=None, step=0, stage="unknown"):
       """Emergency save when session is about to timeout"""
       import datetime
       
       print("🚨 EMERGENCY SESSION SAVE INITIATED")
       
       timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
       emergency_dir = f"/content/drive/MyDrive/rlhf-phi3/emergency_saves"
       os.makedirs(emergency_dir, exist_ok=True)
       
       save_path = f"{emergency_dir}/emergency_{stage}_{timestamp}"
       os.makedirs(save_path, exist_ok=True)
       
       saved_items = []
       
       try:
           # Save model if provided
           if model is not None:
               model_path = f"{save_path}/model"
               model.save_pretrained(model_path)
               saved_items.append("model")
               print(f"✅ Model saved to: {model_path}")
           
           # Save optimizer if provided
           if optimizer is not None:
               optimizer_path = f"{save_path}/optimizer.pt"
               torch.save({
                   'optimizer_state_dict': optimizer.state_dict(),
                   'step': step,
                   'stage': stage,
                   'timestamp': timestamp
               }, optimizer_path)
               saved_items.append("optimizer")
               print(f"✅ Optimizer saved to: {optimizer_path}")
           
           # Save current configuration
           if 'config' in globals():
               config_path = f"{save_path}/config.yaml"
               config.save_yaml(config_path)
               saved_items.append("config")
               print(f"✅ Config saved to: {config_path}")
           
           # Save session info
           session_info = {
               'stage': stage,
               'step': step,
               'timestamp': timestamp,
               'saved_items': saved_items,
               'session_duration': time.time() - session_manager.start_time if 'session_manager' in globals() else 0
           }
           
           info_path = f"{save_path}/session_info.json"
           import json
           with open(info_path, 'w') as f:
               json.dump(session_info, f, indent=2)
           
           print(f"🎉 Emergency save completed!")
           print(f"📂 Save location: {save_path}")
           print(f"💾 Saved items: {', '.join(saved_items)}")
           
           return save_path
           
       except Exception as e:
           print(f"❌ Emergency save failed: {e}")
           return None
   
   def emergency_system_reset():
       """Complete system reset when everything fails"""
       print("🚨 EMERGENCY SYSTEM RESET")
       print("This will clear all GPU memory and restart Python runtime")
       
       response = input("Are you sure? This will lose all unsaved progress (y/n): ")
       if response.lower() != 'y':
           print("Reset cancelled")
           return
       
       try:
           # Clear GPU memory
           import torch
           torch.cuda.empty_cache()
           
           # Force garbage collection
           import gc
           gc.collect()
           
           # Clear variables
           for var in list(globals().keys()):
               if not var.startswith('_'):
                   del globals()[var]
           
           print("✅ Memory cleared")
           
           # Restart Python runtime (Colab specific)
           import os
           print("🔄 Restarting Python runtime...")
           os.kill(os.getpid(), 9)
           
       except Exception as e:
           print(f"❌ Reset failed: {e}")
   ```

2. **Data Recovery Procedures**
   ```python
   def recover_from_corrupted_checkpoint():
       """Recover from corrupted checkpoints"""
       print("🔄 Attempting checkpoint recovery...")
       
       checkpoint_dir = "/content/drive/MyDrive/rlhf-phi3/checkpoints"
       emergency_dir = "/content/drive/MyDrive/rlhf-phi3/emergency_saves"
       
       recovery_candidates = []
       
       # Check regular checkpoints
       if os.path.exists(checkpoint_dir):
           for stage in ["sft", "reward", "ppo"]:
               stage_dir = f"{checkpoint_dir}/{stage}"
               if os.path.exists(stage_dir):
                   checkpoints = [f for f in os.listdir(stage_dir) if f.startswith("checkpoint")]
                   for cp in checkpoints:
                       cp_path = f"{stage_dir}/{cp}"
                       recovery_candidates.append({
                           "path": cp_path,
                           "type": "regular",
                           "stage": stage,
                           "name": cp
                       })
       
       # Check emergency saves
       if os.path.exists(emergency_dir):
           emergency_saves = [f for f in os.listdir(emergency_dir) if f.startswith("emergency")]
           for save in emergency_saves:
               save_path = f"{emergency_dir}/{save}"
               recovery_candidates.append({
                   "path": save_path,
                   "type": "emergency",
                   "stage": save.split("_")[1] if len(save.split("_")) > 1 else "unknown",
                   "name": save
               })
       
       if not recovery_candidates:
           print("❌ No recovery candidates found")
           return None
       
       print(f"📂 Found {len(recovery_candidates)} recovery candidates:")
       for i, candidate in enumerate(recovery_candidates):
           print(f"   {i+1}. {candidate['name']} ({candidate['type']}, {candidate['stage']})")
       
       # Test each candidate
       valid_candidates = []
       for candidate in recovery_candidates:
           try:
               # Basic validation
               if candidate["type"] == "regular":
                   model_path = f"{candidate['path']}/pytorch_model.bin"
                   config_path = f"{candidate['path']}/config.json"
                   
                   if os.path.exists(model_path) and os.path.exists(config_path):
                       valid_candidates.append(candidate)
                       print(f"✅ {candidate['name']} appears valid")
                   else:
                       print(f"❌ {candidate['name']} missing files")
               
               elif candidate["type"] == "emergency":
                   model_path = f"{candidate['path']}/model"
                   
                   if os.path.exists(model_path):
                       valid_candidates.append(candidate)
                       print(f"✅ {candidate['name']} appears valid")
                   else:
                       print(f"❌ {candidate['name']} missing model")
                       
           except Exception as e:
               print(f"❌ {candidate['name']} validation failed: {e}")
       
       if valid_candidates:
           print(f"\n✅ Found {len(valid_candidates)} valid recovery options")
           return valid_candidates
       else:
           print("❌ No valid recovery options found")
           return None
   
   def create_manual_backup():
       """Create manual backup of important files"""
       import shutil
       import datetime
       
       timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
       backup_dir = f"/content/drive/MyDrive/rlhf-phi3-backup-{timestamp}"
       source_dir = "/content/drive/MyDrive/rlhf-phi3"
       
       try:
           if os.path.exists(source_dir):
               shutil.copytree(source_dir, backup_dir)
               print(f"✅ Manual backup created: {backup_dir}")
               
               # Calculate backup size
               backup_size = sum(
                   os.path.getsize(os.path.join(dirpath, filename))
                   for dirpath, dirnames, filenames in os.walk(backup_dir)
                   for filename in filenames
               ) / (1024**3)  # Convert to GB
               
               print(f"💾 Backup size: {backup_size:.2f} GB")
               return backup_dir
           else:
               print("❌ Source directory not found")
               return None
               
       except Exception as e:
           print(f"❌ Backup creation failed: {e}")
           return None
   ```

### 🆘 Getting Help and Support

When you encounter issues not covered in this troubleshooting guide:

#### 📞 Support Channels

1. **GitHub Issues** (Primary Support)
   - Create detailed issue reports
   - Search existing issues first
   - Include error logs and system info
   - Use issue templates for consistency

2. **Community Discussions**
   - Join GitHub Discussions for Q&A
   - Share experiences and solutions
   - Help other users with similar issues

3. **Documentation**
   - Check the comprehensive README
   - Review notebook tutorials
   - Read API documentation

#### 🐛 Reporting Issues

**Before Reporting:**
1. Check this troubleshooting guide
2. Search existing GitHub issues
3. Try the suggested solutions
4. Test with demo configuration

**When Reporting, Include:**
```python
def collect_system_info():
    """Collect comprehensive system information for issue reports"""
    import sys
    import torch
    import platform
    import subprocess
    
    info = {
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable
        },
        "gpu": {},
        "packages": {},
        "environment": {}
    }
    
    # GPU information
    if torch.cuda.is_available():
        info["gpu"] = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "cuda_version": torch.version.cuda
        }
    else:
        info["gpu"]["available"] = False
    
    # Package versions
    packages = ['torch', 'transformers', 'peft', 'trl', 'datasets', 'wandb', 'accelerate']
    for pkg in packages:
        try:
            module = __import__(pkg)
            info["packages"][pkg] = getattr(module, '__version__', 'unknown')
        except ImportError:
            info["packages"][pkg] = 'not installed'
    
    # Environment variables
    env_vars = ['CUDA_VISIBLE_DEVICES', 'WANDB_API_KEY', 'HUGGINGFACE_TOKEN']
    for var in env_vars:
        info["environment"][var] = "set" if os.environ.get(var) else "not set"
    
    return info

def generate_issue_report(error_message, steps_to_reproduce):
    """Generate formatted issue report"""
    system_info = collect_system_info()
    
    report = f"""
## Issue Description
{error_message}

## Steps to Reproduce
{steps_to_reproduce}

## System Information
- Platform: {system_info['system']['platform']}
- Python: {system_info['system']['python_version']}
- GPU Available: {system_info['gpu'].get('available', False)}
- GPU Name: {system_info['gpu'].get('device_name', 'N/A')}
- CUDA Version: {system_info['gpu'].get('cuda_version', 'N/A')}

## Package Versions
"""
    
    for pkg, version in system_info['packages'].items():
        report += f"- {pkg}: {version}\n"
    
    report += "\n## Environment Variables\n"
    for var, status in system_info['environment'].items():
        report += f"- {var}: {status}\n"
    
    return report

# Example usage:
# error_msg = "CUDA out of memory during SFT training"
# steps = "1. Load colab config\n2. Start SFT training\n3. Error occurs after 50 steps"
# report = generate_issue_report(error_msg, steps)
# print(report)
```

#### 💡 Self-Help Resources

1. **Configuration Validator**
   ```python
   def run_comprehensive_diagnostics():
       """Run all diagnostic checks"""
       print("🔍 Running Comprehensive Diagnostics")
       print("=" * 50)
       
       # System checks
       test_network_connectivity()
       check_all_authentication()
       
       # Configuration checks
       if 'config' in globals():
           errors = config.validate()
           if errors:
               print("❌ Configuration errors found:")
               for error in errors:
                   print(f"   • {error}")
           else:
               print("✅ Configuration is valid")
       
       # Memory checks
       print_memory_usage()
       
       # Dataset checks
       print("\n📊 Dataset Connectivity:")
       test_datasets = [
           "HuggingFaceH4/ultrachat_200k",
           "HuggingFaceH4/ultrafeedback_binarized"
       ]
       
       for dataset_name in test_datasets:
           try:
               dataset = load_dataset(dataset_name, split="train[:1]", streaming=True)
               next(iter(dataset))
               print(f"   ✅ {dataset_name}")
           except Exception as e:
               print(f"   ❌ {dataset_name}: {e}")
       
       print("\n" + "=" * 50)
       print("Diagnostics complete!")
   
   # Run diagnostics
   run_comprehensive_diagnostics()
   ```

Remember: Most issues can be resolved by following this troubleshooting guide systematically. Start with the most common solutions and work your way through the more advanced recovery procedures.

```python
# Debugging training issues
config.learning_rate = 2e-5      # Lower learning rate
config.warmup_steps = 200        # More warmup
config.max_grad_norm = 0.5       # Stricter gradient clipping

# LoRA adjustments for better convergence
config.lora.r = 32               # Higher rank for more capacity
config.lora.alpha = 64           # Scaled alpha
config.lora.dropout = 0.05       # Lower dropout

# Enable additional monitoring
config.training.logging_steps = 5    # More frequent logging
config.training.eval_steps = 50      # Regular evaluation
```

**Convergence Monitoring**:
```python
import matplotlib.pyplot as plt
import wandb

def plot_training_metrics(run_id):
    """Plot training metrics from W&B run"""
    api = wandb.Api()
    run = api.run(f"your-entity/rlhf-phi3-pipeline/{run_id}")
    
    # Get training loss
    history = run.scan_history(keys=["train/loss", "train/learning_rate"])
    steps, losses, lrs = [], [], []
    
    for row in history:
        if row.get("train/loss"):
            steps.append(row["_step"])
            losses.append(row["train/loss"])
            lrs.append(row.get("train/learning_rate", 0))
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(steps, losses)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    
    ax2.plot(steps, lrs)
    ax2.set_title("Learning Rate")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Learning Rate")
    
    plt.tight_layout()
    plt.show()

# Use this to analyze training progress
# plot_training_metrics("your-run-id")
```

**Common Convergence Issues**:
- **Learning rate too high**: Loss oscillates or increases
- **Learning rate too low**: Loss decreases very slowly
- **Insufficient warmup**: Training unstable at the beginning
- **Gradient explosion**: Loss becomes NaN or very large
- **Data quality issues**: Inconsistent or corrupted training data

### 🆘 Emergency Procedures

#### Session About to Timeout
```python
# Quick checkpoint save
from rlhf_phi3.utils import emergency_checkpoint_save
checkpoint_path = emergency_checkpoint_save(
    model=current_model,
    optimizer=current_optimizer,
    step=current_step
)
print(f"Emergency checkpoint saved: {checkpoint_path}")
```

#### Complete System Reset
```python
# Clear all GPU memory and restart
import torch
torch.cuda.empty_cache()
import gc
gc.collect()

# Restart Python runtime in Colab
import os
os.kill(os.getpid(), 9)
```

#### Data Recovery
```python
# Recover from corrupted checkpoints
from rlhf_phi3.checkpoints import CheckpointManager

checkpoint_manager = CheckpointManager(config.paths.checkpoint_dir)

# List available checkpoints
checkpoints = checkpoint_manager.list_checkpoints("sft")
print("Available checkpoints:")
for cp in checkpoints:
    print(f"  {cp}")

# Validate checkpoint integrity
for cp in checkpoints:
    if checkpoint_manager.validate_checkpoint(cp):
        print(f"✅ {cp} is valid")
        break
    else:
        print(f"❌ {cp} is corrupted")
```

#### Manual Backup Creation
```python
# Create manual backup of important files
import shutil
import datetime

backup_dir = f"/content/drive/MyDrive/rlhf-phi3-backup-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copytree("/content/drive/MyDrive/rlhf-phi3", backup_dir)
print(f"Backup created: {backup_dir}")
```

### 📞 Getting Help

1. **Check the logs**: Look for detailed error messages in W&B or console
2. **Review configuration**: Ensure all parameters are within valid ranges
3. **Test with demo config**: Try `configs/demo_config.yaml` first
4. **Check system resources**: Monitor GPU memory and disk space
5. **Search existing issues**: Check GitHub issues for similar problems
6. **Join the community**: Participate in discussions and ask questions

#### Useful Debugging Commands

```python
# Check GPU status
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Check disk space
import shutil
total, used, free = shutil.disk_usage("/content")
print(f"Disk space: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")

# Monitor memory usage
import psutil
memory = psutil.virtual_memory()
print(f"RAM usage: {memory.percent}% ({memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB)")

# Check Python environment
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Check key package versions
packages = ['torch', 'transformers', 'peft', 'trl', 'datasets', 'wandb']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"{pkg}: {version}")
    except ImportError:
        print(f"{pkg}: not installed")
```

#### Configuration Validation
```python
# Comprehensive configuration check
from rlhf_phi3 import Config

def validate_environment(config):
    """Validate environment and configuration"""
    issues = []
    
    # Check GPU
    if not torch.cuda.is_available():
        issues.append("No GPU available")
    elif torch.cuda.get_device_properties(0).total_memory < 10e9:
        issues.append("GPU memory < 10GB (may cause OOM errors)")
    
    # Check disk space
    _, _, free = shutil.disk_usage("/content" if 'google.colab' in sys.modules else ".")
    if free < 20e9:
        issues.append("Low disk space (< 20GB free)")
    
    # Check configuration
    config_errors = config.validate()
    issues.extend(config_errors)
    
    # Check authentication
    try:
        import wandb
        wandb.init(project="test", mode="disabled")
    except:
        issues.append("W&B authentication not configured")
    
    try:
        from huggingface_hub import whoami
        whoami()
    except:
        issues.append("HuggingFace authentication not configured")
    
    return issues

# Run validation
config = Config.from_yaml("configs/colab_config.yaml")
issues = validate_environment(config)

if issues:
    print("⚠️  Issues found:")
    for issue in issues:
        print(f"   • {issue}")
else:
    print("✅ Environment validation passed!")
```

### 🐛 Reporting Issues

If you encounter issues not covered here:

1. **Collect information**:
   - Error messages and stack traces
   - Configuration file used
   - System specifications
   - Steps to reproduce

2. **Create an issue** on GitHub with:
   - Clear description of the problem
   - Expected vs actual behavior
   - Environment details
   - Minimal code to reproduce

3. **Include logs** from:
   - Console output
   - Weights & Biases dashboard
   - Google Colab session

## 📚 Documentation

### 📖 Complete Documentation Suite

Our comprehensive documentation covers everything from basic setup to advanced usage:

#### 🚀 Getting Started
- **[Complete Setup Guide](docs/README.md)** - Documentation overview and navigation
- **[Google Colab Setup Guide](docs/COLAB_SETUP_GUIDE.md)** - Detailed Colab setup with screenshots
- **[Installation Guide](docs/INSTALLATION.md)** - Local, cloud, and Docker installation
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Comprehensive problem-solving

#### 📖 Interactive Tutorials

Step-by-step Jupyter notebooks for hands-on learning:

1. **[Setup and Configuration](notebooks/01_setup_and_configuration.ipynb)**
   - Environment setup and authentication
   - Configuration validation and customization
   - System requirements and verification

2. **[SFT Training Tutorial](notebooks/02_sft_training_tutorial.ipynb)**
   - Supervised fine-tuning walkthrough
   - Dataset preparation and validation
   - Training monitoring and optimization

3. **[Reward Model Tutorial](notebooks/03_reward_model_tutorial.ipynb)**
   - Human preference learning concepts
   - Reward model training and evaluation
   - Quality assessment techniques

4. **[PPO Training Tutorial](notebooks/04_ppo_training_tutorial.ipynb)**
   - Reinforcement learning from human feedback
   - Policy optimization strategies
   - Advanced hyperparameter tuning

5. **[Evaluation and Publishing](notebooks/05_evaluation_and_publishing.ipynb)**
   - Comprehensive model evaluation
   - MT-Bench assessment and benchmarking
   - HuggingFace Hub publishing workflow

#### 🔧 Technical Reference

Detailed technical documentation for advanced users:

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and components
- **[Configuration Guide](docs/CONFIGURATION.md)** - Advanced configuration options
- **[Performance Optimization](docs/PERFORMANCE.md)** - Memory and speed optimization
- **[Advanced Usage](docs/ADVANCED.md)** - Advanced features and customization

#### 🤝 Community and Development

Resources for contributors and the community:

- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute to the project
- **[Development Setup](docs/DEVELOPMENT.md)** - Setting up development environment
- **[Testing Guide](docs/TESTING.md)** - Running and writing tests
- **[FAQ](docs/FAQ.md)** - Frequently asked questions

### 🎓 Learning Resources

Educational content to help you understand the concepts:

- **RLHF Concepts**: Understanding reinforcement learning from human feedback
- **PEFT Techniques**: Parameter-efficient fine-tuning with LoRA
- **Memory Optimization**: Techniques for training large models efficiently
- **Evaluation Metrics**: Understanding MT-Bench and quality assessments
- **Best Practices**: Production-ready training and deployment strategies

## 🧪 Testing

The pipeline includes a comprehensive test suite ensuring reliability and correctness:

### Test Categories

```bash
# Unit tests - Test individual components
pytest tests/unit/ -v

# Property-based tests - Test universal properties
pytest tests/property/ -v

# Integration tests - Test end-to-end workflows
pytest tests/integration/ -v

# All tests with coverage report
pytest --cov=rlhf_phi3 --cov-report=html --cov-report=term
```

### Test Coverage

- **Unit Tests**: >90% coverage for core components
- **Property Tests**: 39 universal correctness properties
- **Integration Tests**: Complete pipeline validation
- **Performance Tests**: Memory and speed benchmarks

### Running Specific Tests

```bash
# Test configuration management
pytest tests/unit/test_config_manager.py -v

# Test dataset processing
pytest tests/unit/test_dataset_manager.py -v

# Test training orchestration
pytest tests/integration/test_training_stages_integration.py -v

# Test with specific markers
pytest -m "not slow" -v  # Skip slow tests
pytest -m "gpu" -v       # Run GPU-specific tests
```

### Property-Based Testing

The pipeline uses Hypothesis for property-based testing to ensure correctness across all possible inputs:

```python
# Example: Configuration serialization property
@given(st.integers(min_value=1, max_value=256))
def test_lora_config_serialization_roundtrip(lora_r):
    """Test that LoRA config serialization preserves all data."""
    config = LoRAConfig(r=lora_r)
    serialized = config.to_dict()
    deserialized = LoRAConfig.from_dict(serialized)
    assert config == deserialized
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{rlhf_phi3_pipeline,
  title={RLHF Phi-3 Pipeline: Production-Grade RLHF for Microsoft Phi-3},
  author={RLHF Pipeline Team},
  year={2024},
  url={https://github.com/your-username/rlhf-phi3-pipeline}
}
```

## Acknowledgments

- Microsoft for the Phi-3 model family
- HuggingFace for the transformers library and model hub
- The TRL team for RLHF training utilities
- Google Colab for accessible GPU compute