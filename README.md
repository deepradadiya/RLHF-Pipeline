# RLHF Phi-3 Pipeline

A production-grade Reinforcement Learning from Human Feedback (RLHF) pipeline for Microsoft Phi-3 Mini (3.8B parameters). Designed for Google Colab's T4 GPU constraints with comprehensive checkpoint management and experiment tracking.

## Overview

This pipeline implements a complete three-stage RLHF training process:

1. **Supervised Fine-Tuning (SFT)** - Train on instruction-following datasets
2. **Reward Model Training** - Learn human preferences from comparison data  
3. **Proximal Policy Optimization (PPO)** - Final RLHF training stage

## Key Features

- 🚀 **Memory Efficient**: PEFT/LoRA techniques reduce trainable parameters by 99%
- 💾 **Persistent Checkpoints**: Google Drive integration for session recovery
- 📊 **Experiment Tracking**: Weights & Biases integration with comprehensive logging
- 🎯 **MT-Bench Evaluation**: Professional-grade model assessment
- 🤗 **HuggingFace Integration**: Seamless model publishing and sharing
- 🔧 **Colab Optimized**: Designed for T4 GPU constraints and 12-hour sessions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rlhf-phi3-pipeline.git
cd rlhf-phi3-pipeline

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Google Colab Setup

1. Mount Google Drive for checkpoint persistence
2. Install the pipeline in Colab
3. Configure authentication for Weights & Biases and HuggingFace

See the [Colab Setup Guide](notebooks/01_setup_and_configuration.ipynb) for detailed instructions.

## Project Structure

```
rlhf_phi3/
├── config/          # Configuration management
├── data/            # Dataset loading and preprocessing  
├── models/          # Model management and PEFT
├── training/        # Training orchestration
├── checkpoints/     # Checkpoint persistence
├── tracking/        # Experiment tracking
├── evaluation/      # Model evaluation
├── publishing/      # Model publishing
└── utils/           # Common utilities

tests/               # Comprehensive test suite
notebooks/           # Tutorial notebooks
configs/             # Configuration files
```

## Usage

### Basic Pipeline Execution

```python
from rlhf_phi3 import Config, TrainingOrchestrator

# Load configuration
config = Config.from_yaml("configs/default_config.yaml")

# Initialize and run the pipeline
orchestrator = TrainingOrchestrator(config)
final_model_path = orchestrator.run_full_pipeline()

print(f"Training complete! Model saved to: {final_model_path}")
```

### Stage-by-Stage Execution

```python
# Run individual stages
sft_checkpoint = orchestrator.run_sft_stage()
reward_checkpoint = orchestrator.run_reward_stage(sft_checkpoint)
final_model = orchestrator.run_ppo_stage(sft_checkpoint, reward_checkpoint)
```

## Configuration

The pipeline uses YAML configuration files for easy customization:

- `configs/default_config.yaml` - Standard configuration
- `configs/colab_config.yaml` - Optimized for Google Colab

Key configuration sections:
- Model settings (Phi-3 variant, context length)
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters per stage
- Dataset selection and preprocessing
- Checkpoint and logging settings

## Requirements

### Hardware Requirements
- **GPU**: NVIDIA T4 (15GB VRAM) or better
- **RAM**: 12GB+ system memory
- **Storage**: 50GB+ for checkpoints and datasets

### Software Requirements
- Python 3.8+
- CUDA 11.8+ or 12.1+
- PyTorch 2.0+
- Transformers 4.36+

## Documentation

- [Setup Guide](notebooks/01_setup_and_configuration.ipynb)
- [SFT Training Tutorial](notebooks/02_sft_training_tutorial.ipynb)
- [Reward Model Tutorial](notebooks/03_reward_model_tutorial.ipynb)
- [PPO Training Tutorial](notebooks/04_ppo_training_tutorial.ipynb)
- [Evaluation and Publishing](notebooks/05_evaluation_and_publishing.ipynb)

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
pytest tests/unit/

# Property-based tests  
pytest tests/property/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=rlhf_phi3 --cov-report=html
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