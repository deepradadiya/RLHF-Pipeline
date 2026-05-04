"""
Pytest configuration and shared fixtures for RLHF Phi-3 Pipeline tests.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from typing import Generator, Dict, Any
import yaml
from unittest.mock import Mock, patch

# Import test utilities and fixtures
from tests.fixtures.test_data import (
    SAMPLE_SFT_CONVERSATIONS, SAMPLE_PREFERENCE_DATA, MOCK_MODEL_CONFIGS,
    SAMPLE_EVALUATION_RESULTS, MockWandBRun, MockHuggingFaceModel, MockTokenizer
)
from tests.utils import create_minimal_valid_config, setup_test_environment

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration for testing."""
    return create_minimal_valid_config()

@pytest.fixture
def config_file(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a temporary config file for testing."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path

@pytest.fixture
def mock_model_path(temp_dir: Path) -> Path:
    """Create a mock model directory structure."""
    model_dir = temp_dir / "mock_model"
    model_dir.mkdir()
    
    # Create mock model files
    (model_dir / "config.json").write_text('{"model_type": "phi3"}')
    (model_dir / "pytorch_model.bin").write_text("mock model weights")
    (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
    
    return model_dir

# Additional fixtures for comprehensive testing

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
def test_environment(temp_dir):
    """Provide a complete test environment setup."""
    return setup_test_environment(temp_dir)

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
def mock_cuda_available():
    """Mock CUDA availability for testing."""
    with patch('torch.cuda.is_available', return_value=False):
        yield

@pytest.fixture
def mock_environment_vars():
    """Mock environment variables for testing."""
    test_env = {
        'WANDB_API_KEY': 'test_wandb_key',
        'HF_TOKEN': 'test_hf_token',
        'GOOGLE_APPLICATION_CREDENTIALS': '/tmp/test_credentials.json'
    }
    
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        yield test_env
    finally:
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

@pytest.fixture(scope="session")
def hypothesis_settings():
    """Configure Hypothesis settings for the test session."""
    from hypothesis import settings, Verbosity
    
    # Use CI profile if in CI environment
    profile = "ci" if os.getenv("CI") else "default"
    
    return settings(
        max_examples=50 if profile == "default" else 100,
        deadline=5000,  # 5 seconds
        verbosity=Verbosity.normal,
        suppress_health_check=[],
        report_multiple_bugs=True
    )

# Pytest markers for test categorization
pytest_plugins = []

# Configure pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "property: Property-based tests using Hypothesis")
    config.addinivalue_line("markers", "integration: Integration tests for component interactions")
    config.addinivalue_line("markers", "slow: Tests that require significant compute time")
    config.addinivalue_line("markers", "gpu: Tests that require GPU access")
    config.addinivalue_line("markers", "colab: Tests specific to Google Colab environment")
    config.addinivalue_line("markers", "network: Tests that require network access")
    config.addinivalue_line("markers", "auth: Tests that require authentication")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "property" in str(item.fspath):
            item.add_marker(pytest.mark.property)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker for tests with "slow" in name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add gpu marker for tests with "gpu" in name
        if "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Add network marker for tests with "network" or "api" in name
        if any(keyword in item.name.lower() for keyword in ["network", "api", "download", "upload"]):
            item.add_marker(pytest.mark.network)

# Fixtures for mocking external services

@pytest.fixture
def mock_transformers():
    """Mock transformers library components."""
    with patch('transformers.AutoModelForCausalLM') as mock_model, \
         patch('transformers.AutoTokenizer') as mock_tokenizer, \
         patch('transformers.AutoConfig') as mock_config:
        
        # Configure mocks
        mock_model.from_pretrained.return_value = MockHuggingFaceModel("test-model")
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()
        mock_config.from_pretrained.return_value = {"model_type": "phi3"}
        
        yield {
            'model': mock_model,
            'tokenizer': mock_tokenizer,
            'config': mock_config
        }

@pytest.fixture
def mock_datasets():
    """Mock datasets library."""
    with patch('datasets.load_dataset') as mock_load:
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(side_effect=lambda i: SAMPLE_SFT_CONVERSATIONS[i % len(SAMPLE_SFT_CONVERSATIONS)])
        mock_dataset.map = Mock(return_value=mock_dataset)
        mock_dataset.filter = Mock(return_value=mock_dataset)
        mock_dataset.select = Mock(return_value=mock_dataset)
        
        mock_load.return_value = mock_dataset
        
        yield mock_load

@pytest.fixture
def mock_peft():
    """Mock PEFT library components."""
    with patch('peft.LoraConfig') as mock_lora_config, \
         patch('peft.get_peft_model') as mock_get_peft_model, \
         patch('peft.PeftModel') as mock_peft_model:
        
        # Configure mocks
        mock_lora_config.return_value = Mock()
        mock_get_peft_model.return_value = MockHuggingFaceModel("peft-model")
        mock_peft_model.from_pretrained.return_value = MockHuggingFaceModel("peft-model")
        
        yield {
            'lora_config': mock_lora_config,
            'get_peft_model': mock_get_peft_model,
            'peft_model': mock_peft_model
        }

@pytest.fixture
def mock_accelerate():
    """Mock Accelerate library."""
    with patch('accelerate.Accelerator') as mock_accelerator:
        mock_acc_instance = Mock()
        mock_acc_instance.device = "cpu"
        mock_acc_instance.prepare = Mock(side_effect=lambda *args: args)
        mock_accelerator.return_value = mock_acc_instance
        
        yield mock_acc_instance