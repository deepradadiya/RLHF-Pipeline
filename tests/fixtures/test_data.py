"""
Test data fixtures and utilities for RLHF Phi-3 Pipeline tests.

This module provides reusable test data, mock objects, and utilities
for testing all pipeline components.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import pytest

# Sample dataset entries for testing

SAMPLE_SFT_CONVERSATIONS = [
    {
        "messages": [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain neural networks briefly."},
            {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How do I start learning Python?"},
            {"role": "assistant", "content": "Start with basic syntax, practice with simple programs, use online resources like Python.org tutorial, and work on small projects to build confidence."}
        ]
    }
]

SAMPLE_PREFERENCE_DATA = [
    {
        "prompt": "Explain quantum computing in simple terms.",
        "chosen": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot, potentially solving certain problems exponentially faster.",
        "rejected": "Quantum computing is just really fast regular computing with quantum stuff."
    },
    {
        "prompt": "What are the benefits of renewable energy?",
        "chosen": "Renewable energy sources like solar and wind power offer environmental benefits by reducing greenhouse gas emissions, provide energy security through domestic resources, and create economic opportunities in growing industries.",
        "rejected": "Renewable energy is good because it's renewable and doesn't pollute much."
    },
    {
        "prompt": "How does photosynthesis work?",
        "chosen": "Photosynthesis is the process by which plants convert light energy, carbon dioxide, and water into glucose and oxygen using chlorophyll in their leaves, providing energy for the plant and oxygen for other organisms.",
        "rejected": "Plants eat sunlight and make oxygen somehow."
    }
]

# Mock model configurations for testing

MOCK_MODEL_CONFIGS = {
    "tiny": {
        "name": "microsoft/DialoGPT-small",  # Smaller model for testing
        "max_length": 512,
        "device": "cpu"
    },
    "phi3_mini": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "max_length": 2048,
        "device": "cpu"
    },
    "test_model": {
        "name": "test-model",
        "max_length": 1024,
        "device": "cpu"
    }
}

# Test checkpoint metadata

@dataclass
class MockCheckpointMetadata:
    """Mock checkpoint metadata for testing."""
    stage: str
    epoch: int
    step: int
    model_path: str
    optimizer_path: str
    config_hash: str
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Test evaluation results

SAMPLE_EVALUATION_RESULTS = {
    "mt_bench_score": 6.5,
    "category_scores": {
        "writing": 7.2,
        "roleplay": 6.8,
        "reasoning": 6.1,
        "math": 5.9,
        "coding": 6.3,
        "extraction": 7.0,
        "stem": 6.4,
        "humanities": 6.6
    },
    "helpfulness_score": 7.1,
    "harmlessness_score": 8.2,
    "honesty_score": 6.9,
    "tokens_per_second": 12.5,
    "memory_usage_mb": 3840.0,
    "sample_responses": [
        {
            "prompt": "Write a short story about AI.",
            "response": "In a world where artificial intelligence had become commonplace, Maya discovered that her AI assistant had developed something unexpected: creativity."
        },
        {
            "prompt": "Explain climate change.",
            "response": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human activities that increase greenhouse gas concentrations in the atmosphere."
        }
    ]
}

# Utility functions for test data generation

def create_mock_dataset_file(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Create a mock dataset file in JSONL format."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def create_mock_config_file(config: Dict[str, Any], file_path: Path) -> None:
    """Create a mock configuration file."""
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)

def create_mock_checkpoint_dir(base_dir: Path, stage: str, epoch: int = 1) -> Path:
    """Create a mock checkpoint directory structure."""
    checkpoint_dir = base_dir / f"{stage}_checkpoint_epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock checkpoint files
    (checkpoint_dir / "pytorch_model.bin").write_text("mock model weights")
    (checkpoint_dir / "optimizer.pt").write_text("mock optimizer state")
    (checkpoint_dir / "trainer_state.json").write_text('{"epoch": ' + str(epoch) + '}')
    (checkpoint_dir / "config.json").write_text('{"model_type": "phi3"}')
    
    # Create metadata file
    metadata = MockCheckpointMetadata(
        stage=stage,
        epoch=epoch,
        step=epoch * 100,
        model_path=str(checkpoint_dir / "pytorch_model.bin"),
        optimizer_path=str(checkpoint_dir / "optimizer.pt"),
        config_hash="mock_hash_" + stage,
        metrics={"loss": 2.5 - (epoch * 0.1), "accuracy": 0.6 + (epoch * 0.05)}
    )
    
    with open(checkpoint_dir / "metadata.json", 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)
    
    return checkpoint_dir

class MockWandBRun:
    """Mock Weights & Biases run for testing."""
    
    def __init__(self, project: str, name: str, config: Dict[str, Any]):
        self.project = project
        self.name = name
        self.config = config
        self.logged_metrics = []
        self.logged_artifacts = []
        self.finished = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Mock metric logging."""
        self.logged_metrics.append({"metrics": metrics, "step": step})
    
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Mock artifact logging."""
        self.logged_artifacts.append({"path": path, "name": name})
    
    def finish(self) -> None:
        """Mock run finishing."""
        self.finished = True

class MockHuggingFaceModel:
    """Mock HuggingFace model for testing."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = {"model_type": "phi3", "vocab_size": 32000}
        self.device = "cpu"
        self.training = False
    
    def to(self, device: str):
        """Mock device placement."""
        self.device = device
        return self
    
    def train(self, mode: bool = True):
        """Mock training mode."""
        self.training = mode
        return self
    
    def eval(self):
        """Mock evaluation mode."""
        self.training = False
        return self
    
    def save_pretrained(self, path: str) -> None:
        """Mock model saving."""
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "config.json", 'w') as f:
            json.dump(self.config, f)

class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Mock text encoding."""
        # Simple mock: return list of integers based on text length
        tokens = [self.bos_token_id] + list(range(3, min(len(text) + 3, max_length or 512))) + [self.eos_token_id]
        return tokens[:max_length] if max_length else tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Mock token decoding."""
        return f"decoded_text_from_{len(token_ids)}_tokens"
    
    def __call__(self, text, **kwargs):
        """Mock tokenizer call."""
        max_length = kwargs.get('max_length', 512)
        padding = kwargs.get('padding', False)
        truncation = kwargs.get('truncation', False)
        
        if isinstance(text, str):
            text = [text]
        
        results = {
            'input_ids': [],
            'attention_mask': []
        }
        
        for t in text:
            tokens = self.encode(t, max_length if truncation else None)
            if padding and len(tokens) < max_length:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
            
            results['input_ids'].append(tokens)
            results['attention_mask'].append([1] * len(tokens))
        
        return results

# Test environment utilities

def setup_test_environment(temp_dir: Path) -> Dict[str, Path]:
    """Set up a complete test environment with directories and files."""
    env = {
        'base_dir': temp_dir,
        'output_dir': temp_dir / 'output',
        'cache_dir': temp_dir / 'cache',
        'logs_dir': temp_dir / 'logs',
        'checkpoints_dir': temp_dir / 'checkpoints',
        'datasets_dir': temp_dir / 'datasets',
        'models_dir': temp_dir / 'models'
    }
    
    # Create all directories
    for path in env.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create sample dataset files
    create_mock_dataset_file(
        SAMPLE_SFT_CONVERSATIONS,
        env['datasets_dir'] / 'sft_data.jsonl'
    )
    
    create_mock_dataset_file(
        SAMPLE_PREFERENCE_DATA,
        env['datasets_dir'] / 'preference_data.jsonl'
    )
    
    # Create sample config file
    create_mock_config_file(
        MOCK_MODEL_CONFIGS['test_model'],
        env['output_dir'] / 'config.json'
    )
    
    return env

# Pytest fixtures

@pytest.fixture
def sample_sft_data():
    """Provide sample SFT conversation data."""
    return SAMPLE_SFT_CONVERSATIONS.copy()

@pytest.fixture
def sample_preference_data():
    """Provide sample preference data."""
    return SAMPLE_PREFERENCE_DATA.copy()

@pytest.fixture
def mock_model_configs():
    """Provide mock model configurations."""
    return MOCK_MODEL_CONFIGS.copy()

@pytest.fixture
def sample_evaluation_results():
    """Provide sample evaluation results."""
    return SAMPLE_EVALUATION_RESULTS.copy()

@pytest.fixture
def mock_wandb_run():
    """Provide a mock WandB run."""
    return MockWandBRun("test-project", "test-run", {"learning_rate": 1e-4})

@pytest.fixture
def mock_hf_model():
    """Provide a mock HuggingFace model."""
    return MockHuggingFaceModel("test-model")

@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer."""
    return MockTokenizer()

@pytest.fixture
def test_environment(temp_dir):
    """Provide a complete test environment setup."""
    return setup_test_environment(temp_dir)