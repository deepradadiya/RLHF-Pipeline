# Installation Guide

This comprehensive installation guide covers all supported environments and platforms for the RLHF Phi-3 Pipeline.

## 🎯 Installation Options

Choose the installation method that best fits your needs:

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Google Colab** | Beginners, Quick Start | Free GPU, No setup | Session limits, Less control |
| **Local Installation** | Development, Full Control | No limits, Full customization | Requires setup, Hardware needed |
| **Docker** | Reproducible Environments | Consistent, Portable | Docker knowledge required |
| **Cloud Platforms** | Scalable Training | Powerful hardware, Scalable | Cost, Setup complexity |

## 🚀 Google Colab Installation (Recommended for Beginners)

### Prerequisites
- Google account (free)
- Web browser (Chrome recommended)
- Stable internet connection

### Quick Setup
```python
# 1. Enable GPU runtime: Runtime → Change runtime type → GPU
# 2. Run this setup cell:

import torch
print(f"GPU Available: {torch.cuda.is_available()}")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install pipeline
!git clone https://github.com/your-username/rlhf-phi3-pipeline.git
%cd rlhf-phi3-pipeline
!pip install -r requirements.txt
!pip install -e .

# Verify installation
from rlhf_phi3 import Config
print("✅ Installation complete!")
```

For detailed Colab setup, see the [Google Colab Setup Guide](COLAB_SETUP_GUIDE.md).

## 💻 Local Installation

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8+ (3.9+ recommended)
- **RAM**: 16GB system memory
- **Storage**: 100GB free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8+ or 12.1+

#### Recommended Requirements
- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.9 or 3.10
- **RAM**: 32GB+ system memory
- **Storage**: 200GB+ SSD storage
- **GPU**: RTX 3090, RTX 4090, or A100
- **CUDA**: 12.1+

### Step 1: System Preparation

#### Linux (Ubuntu/Debian)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv git curl wget

# Install NVIDIA drivers (if not already installed)
sudo apt install -y nvidia-driver-535  # or latest version

# Install CUDA (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Verify CUDA installation
nvidia-smi
nvcc --version
```

#### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python@3.9 git

# Note: CUDA is not available on macOS
# You can still run CPU training (very slow) or use MPS on Apple Silicon
```

#### Windows
```powershell
# Install Python from python.org or Microsoft Store
# Install Git from git-scm.com
# Install CUDA from NVIDIA developer website

# Verify installations
python --version
git --version
nvidia-smi
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv rlhf-env

# Activate environment
# Linux/macOS:
source rlhf-env/bin/activate
# Windows:
rlhf-env\Scripts\activate

# Verify activation
which python  # Should point to venv
python --version
```

### Step 3: Install PyTorch with CUDA

Choose the appropriate PyTorch installation for your CUDA version:

#### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only (Not Recommended)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Apple Silicon (MPS)
```bash
pip install torch torchvision torchaudio
```

#### Verify PyTorch Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### Step 4: Install RLHF Pipeline

```bash
# Clone repository
git clone https://github.com/your-username/rlhf-phi3-pipeline.git
cd rlhf-phi3-pipeline

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
python -c "from rlhf_phi3 import Config; print('✅ Installation successful!')"
```

### Step 5: Configure Environment

#### Create Environment File
```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env  # or use your preferred editor
```

#### Environment Variables (.env)
```bash
# Weights & Biases (for experiment tracking)
WANDB_API_KEY=your_wandb_api_key_here

# HuggingFace (for model access and publishing)
HUGGINGFACE_TOKEN=your_hf_token_here

# Optional: Custom cache directories
HF_HOME=/path/to/huggingface/cache
WANDB_CACHE_DIR=/path/to/wandb/cache
TRANSFORMERS_CACHE=/path/to/transformers/cache

# Optional: Logging configuration
LOG_LEVEL=INFO
PYTHONPATH=/path/to/rlhf-phi3-pipeline

# Optional: CUDA configuration
CUDA_VISIBLE_DEVICES=0  # Use first GPU only
```

### Step 6: Verify Installation

Run the comprehensive verification script:

```python
# verification_script.py
import sys
import torch
import os
from pathlib import Path

def verify_installation():
    """Comprehensive installation verification"""
    print("🔍 Verifying RLHF Phi-3 Pipeline Installation")
    print("=" * 60)
    
    # Python version check
    print(f"\n1️⃣ Python Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # PyTorch check
    print(f"\n2️⃣ PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Test GPU computation
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x.T)
            print("✅ GPU computation test passed")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
            return False
    else:
        print("⚠️ CUDA not available - CPU training will be very slow")
    
    # Package imports
    print("\n3️⃣ Package Imports:")
    required_packages = [
        'rlhf_phi3',
        'transformers',
        'peft',
        'trl',
        'datasets',
        'accelerate',
        'wandb',
        'evaluate'
    ]
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {package}: {version}")
        except ImportError as e:
            print(f"   ❌ {package}: {e}")
            return False
    
    # Configuration test
    print("\n4️⃣ Configuration Test:")
    try:
        from rlhf_phi3 import Config
        config = Config.from_yaml("configs/default_config.yaml")
        errors = config.validate()
        if errors:
            print(f"   ❌ Configuration errors: {errors}")
            return False
        else:
            print("   ✅ Configuration valid")
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False
    
    # Model access test
    print("\n5️⃣ Model Access Test:")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        print("   ✅ Model access confirmed")
    except Exception as e:
        print(f"   ❌ Model access failed: {e}")
        return False
    
    # Dataset access test
    print("\n6️⃣ Dataset Access Test:")
    try:
        from datasets import load_dataset
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:1]", streaming=True)
        sample = next(iter(dataset))
        print("   ✅ Dataset access confirmed")
    except Exception as e:
        print(f"   ❌ Dataset access failed: {e}")
        return False
    
    # Storage check
    print("\n7️⃣ Storage Check:")
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    print(f"   Free space: {free_gb:.1f}GB")
    if free_gb < 50:
        print("   ⚠️ Low disk space - recommend 100GB+ free")
    else:
        print("   ✅ Sufficient storage")
    
    print("\n" + "=" * 60)
    print("🎉 Installation verification completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Set up authentication (W&B and HuggingFace tokens)")
    print("2. Run demo training: python -c \"from rlhf_phi3 import Config, TrainingOrchestrator; orchestrator = TrainingOrchestrator(Config.from_yaml('configs/demo_config.yaml')); orchestrator.run_demo_pipeline()\"")
    print("3. Explore the notebook tutorials")
    
    return True

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
```

Run the verification:
```bash
python verification_script.py
```

## 🐳 Docker Installation

### Prerequisites
- Docker installed and running
- NVIDIA Docker runtime (for GPU support)

### Build Docker Image

#### Option 1: Use Pre-built Image
```bash
# Pull pre-built image
docker pull your-username/rlhf-phi3-pipeline:latest

# Run container with GPU support
docker run --gpus all -it --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  your-username/rlhf-phi3-pipeline:latest
```

#### Option 2: Build from Source
```bash
# Clone repository
git clone https://github.com/your-username/rlhf-phi3-pipeline.git
cd rlhf-phi3-pipeline

# Build Docker image
docker build -t rlhf-phi3-pipeline .

# Run container
docker run --gpus all -it --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  rlhf-phi3-pipeline
```

### Docker Compose (Recommended)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  rlhf-phi3:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - WANDB_API_KEY=${WANDB_API_KEY}
      - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
    volumes:
      - ./data:/workspace/data
      - ./outputs:/workspace/outputs
      - ./configs:/workspace/configs
    ports:
      - "8888:8888"  # Jupyter notebook
      - "6006:6006"  # TensorBoard
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Run with Docker Compose:
```bash
# Set environment variables
export WANDB_API_KEY=your_key_here
export HUGGINGFACE_TOKEN=your_token_here

# Start services
docker-compose up -d

# Access Jupyter notebook at http://localhost:8888
```

## ☁️ Cloud Platform Installation

### AWS SageMaker

#### Setup SageMaker Notebook Instance
```python
import boto3
import sagemaker

# Create SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create notebook instance
sm_client = boto3.client('sagemaker')

response = sm_client.create_notebook_instance(
    NotebookInstanceName='rlhf-phi3-pipeline',
    InstanceType='ml.p3.2xlarge',  # GPU instance
    RoleArn=role,
    DefaultCodeRepository='https://github.com/your-username/rlhf-phi3-pipeline.git'
)
```

#### Install in SageMaker Notebook
```python
# In SageMaker notebook cell
import sys
!{sys.executable} -m pip install -r requirements.txt
!{sys.executable} -m pip install -e .

# Verify installation
from rlhf_phi3 import Config
print("✅ Installation complete!")
```

### Google Cloud Platform (Vertex AI)

#### Create Vertex AI Workbench
```bash
# Using gcloud CLI
gcloud notebooks instances create rlhf-phi3-instance \
  --location=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator-type=NVIDIA_TESLA_T4 \
  --accelerator-core-count=1 \
  --boot-disk-size=100GB \
  --data-disk-size=200GB
```

#### Install in Vertex AI Notebook
```bash
# In terminal
git clone https://github.com/your-username/rlhf-phi3-pipeline.git
cd rlhf-phi3-pipeline
pip install -r requirements.txt
pip install -e .
```

### Azure Machine Learning

#### Create Compute Instance
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ComputeInstance

# Create ML client
ml_client = MLClient.from_config()

# Create compute instance
compute_instance = ComputeInstance(
    name="rlhf-phi3-instance",
    size="Standard_NC6s_v3",  # GPU instance
    idle_time_before_shutdown_minutes=30
)

ml_client.compute.begin_create_or_update(compute_instance)
```

## 🔧 Development Installation

For contributors and developers:

### Additional Development Dependencies
```bash
# Install development dependencies
pip install -e ".[dev]"

# Or install manually
pip install pytest hypothesis black flake8 mypy pre-commit

# Set up pre-commit hooks
pre-commit install
```

### Development Tools Setup
```bash
# Code formatting
black rlhf_phi3/ tests/

# Linting
flake8 rlhf_phi3/ tests/

# Type checking
mypy rlhf_phi3/

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rlhf_phi3 --cov-report=html
```

## 🚨 Troubleshooting Installation

### Common Issues

#### CUDA Not Found
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# If missing, install CUDA toolkit
# Ubuntu:
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA website
```

#### PyTorch CUDA Mismatch
```bash
# Uninstall and reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Permission Errors
```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/
```

#### Memory Issues During Installation
```bash
# Use no-cache option
pip install --no-cache-dir -r requirements.txt

# Or install packages individually
pip install torch
pip install transformers
# ... etc
```

#### Network/Firewall Issues
```bash
# Use trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Or configure proxy
pip install --proxy http://proxy.server:port -r requirements.txt
```

### Verification Failures

If verification fails, check:

1. **Python Version**: Ensure Python 3.8+
2. **CUDA Compatibility**: Match PyTorch and CUDA versions
3. **Memory**: Ensure sufficient RAM and disk space
4. **Network**: Check internet connectivity for downloads
5. **Permissions**: Ensure write access to installation directory

### Getting Help

If installation fails:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Search [GitHub Issues](https://github.com/your-username/rlhf-phi3-pipeline/issues)
3. Create a new issue with:
   - Operating system and version
   - Python version
   - CUDA version (if applicable)
   - Complete error messages
   - Steps you've already tried

## 📋 Post-Installation Checklist

After successful installation:

- [ ] ✅ Python 3.8+ installed
- [ ] ✅ PyTorch with CUDA support working
- [ ] ✅ All required packages installed
- [ ] ✅ RLHF pipeline imports successfully
- [ ] ✅ Configuration files load without errors
- [ ] ✅ Model and dataset access confirmed
- [ ] ✅ Environment variables configured
- [ ] ✅ Authentication tokens set up
- [ ] ✅ Sufficient storage space available
- [ ] ✅ Demo training runs successfully

## 🎯 Next Steps

After installation:

1. **Set up authentication** for W&B and HuggingFace
2. **Run the demo training** to verify everything works
3. **Explore the notebook tutorials** for hands-on learning
4. **Read the configuration guide** to customize settings
5. **Join the community** for support and discussions

---

Congratulations! You've successfully installed the RLHF Phi-3 Pipeline. You're now ready to start training your own language models!