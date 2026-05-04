# RLHF Phi-3 Pipeline Notebooks

This directory contains Jupyter notebooks that provide step-by-step tutorials for using the RLHF Phi-3 pipeline.

## Notebooks Overview

1. **01_setup_and_configuration.ipynb** - Environment setup and configuration
2. **02_sft_training_tutorial.ipynb** - Supervised Fine-Tuning stage
3. **03_reward_model_tutorial.ipynb** - Reward model training stage  
4. **04_ppo_training_tutorial.ipynb** - PPO training stage
5. **05_evaluation_and_publishing.ipynb** - Model evaluation and publishing

## Usage

### Google Colab

1. Upload the notebooks to Google Colab
2. Follow the setup instructions in notebook 01
3. Run the notebooks in sequence

### Local Jupyter

1. Install Jupyter: `pip install jupyter`
2. Start Jupyter: `jupyter notebook`
3. Navigate to the notebooks directory
4. Run the notebooks in sequence

## Prerequisites

- Python 3.8+
- GPU with 15GB+ VRAM (T4 or better)
- Google Drive account (for Colab usage)
- Weights & Biases account (for experiment tracking)
- HuggingFace account (for model publishing)

## Notes

- The notebooks are designed to work in Google Colab's free tier
- Each notebook includes memory optimization techniques
- Checkpoints are automatically saved to Google Drive
- All notebooks include comprehensive error handling and recovery instructions