# Testing Framework Documentation

This document describes the comprehensive testing framework for the RLHF Phi-3 Pipeline project.

## Overview

The testing framework is designed to ensure high code quality, reliability, and correctness of the RLHF pipeline. It includes:

- **Unit Tests**: Test individual components in isolation
- **Property-Based Tests**: Test universal properties using Hypothesis
- **Integration Tests**: Test component interactions and end-to-end workflows
- **Coverage Reporting**: Ensure 90%+ test coverage
- **CI/CD Integration**: Automated testing in GitHub Actions

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── hypothesis_config.py        # Hypothesis configuration and strategies
├── utils.py                    # Test utilities and helper functions
├── README.md                   # This documentation
├── fixtures/
│   ├── __init__.py
│   └── test_data.py           # Test data fixtures and mock objects
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_config_manager.py
│   └── ...
├── property/                   # Property-based tests
│   ├── __init__.py
│   ├── test_config_manager_properties.py
│   └── ...
└── integration/                # Integration tests
    ├── __init__.py
    └── ...
```

## Test Categories

### Unit Tests (`tests/unit/`)

Unit tests verify individual components work correctly in isolation.

**Characteristics:**
- Fast execution (< 1 second per test)
- No external dependencies
- Mock external services
- Test specific examples and edge cases

**Example:**
```python
def test_config_validation_with_invalid_learning_rate():
    config = Config()
    config.training.sft.learning_rate = -1.0  # Invalid
    
    assert not config.validate_config()
    errors = config.get_validation_errors()
    assert any("learning rate" in error for error in errors)
```

### Property-Based Tests (`tests/property/`)

Property-based tests verify universal properties hold across all valid inputs using Hypothesis.

**Characteristics:**
- Test properties that should always be true
- Generate random test data
- Find edge cases automatically
- Validate correctness properties from design document

**Example:**
```python
@given(valid_config())
def test_configuration_serialization_round_trip(config: Config):
    """Configuration serialization should preserve all data."""
    # Save to file
    config.save_config(temp_path)
    
    # Load from file
    loaded_config = Config.load_config(temp_path)
    
    # Should be identical
    assert asdict(config) == asdict(loaded_config)
```

### Integration Tests (`tests/integration/`)

Integration tests verify components work together correctly.

**Characteristics:**
- Test component interactions
- May use real external services (with mocking)
- Longer execution time
- Test end-to-end workflows

**Example:**
```python
def test_full_sft_training_pipeline():
    """Test complete SFT training with minimal dataset."""
    orchestrator = TrainingOrchestrator(config)
    checkpoint_path = orchestrator.run_sft_stage()
    
    assert Path(checkpoint_path).exists()
    assert validate_checkpoint_integrity(checkpoint_path) == []
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.property`: Property-based tests  
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Tests requiring significant time/resources
- `@pytest.mark.gpu`: Tests requiring GPU access
- `@pytest.mark.colab`: Google Colab specific tests
- `@pytest.mark.network`: Tests requiring network access
- `@pytest.mark.auth`: Tests requiring authentication

## Running Tests

### Using the Test Runner

The `run_tests.py` script provides a convenient interface:

```bash
# Run all tests
python run_tests.py all

# Run specific test types
python run_tests.py unit
python run_tests.py property
python run_tests.py integration

# Run fast tests only
python run_tests.py fast

# Run with coverage
python run_tests.py all --coverage

# Run CI pipeline
python run_tests.py ci
```

### Using Make

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-property
make test-integration

# Run fast tests
make test-fast

# Generate coverage report
make coverage

# Run linting
make lint

# Format code
make format
```

### Using Pytest Directly

```bash
# Run all tests
pytest tests/

# Run specific test types
pytest tests/unit/ -m "unit"
pytest tests/property/ -m "property"
pytest tests/integration/ -m "integration"

# Run with coverage
pytest tests/ --cov=rlhf_phi3 --cov-report=html

# Run fast tests only
pytest tests/ -m "not slow and not gpu and not network"
```

### Using Tox

```bash
# Run tests across Python versions
tox

# Run specific environments
tox -e py39
tox -e lint
tox -e coverage
```

## Configuration

### Pytest Configuration

Main configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
addopts = "-ra -q --strict-markers --cov=rlhf_phi3 --cov-fail-under=90"
markers = [
    "unit: Unit tests",
    "property: Property-based tests",
    "integration: Integration tests",
    "slow: Slow tests",
    "gpu: GPU tests",
]
```

Additional configuration in `pytest.ini` for extended options.

### Hypothesis Configuration

Property-based testing configuration in `tests/hypothesis_config.py`:

```python
# Profiles for different environments
settings.register_profile("default", max_examples=50, deadline=5000)
settings.register_profile("ci", max_examples=100, deadline=10000)
settings.register_profile("thorough", max_examples=200, deadline=30000)
```

Set profile with environment variable:
```bash
export HYPOTHESIS_PROFILE=ci
```

### Coverage Configuration

Coverage settings in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["rlhf_phi3"]
omit = ["*/tests/*", "setup.py"]

[tool.coverage.report]
fail_under = 90
show_missing = true
```

## Fixtures and Test Data

### Shared Fixtures (`tests/conftest.py`)

Common fixtures available to all tests:

- `temp_dir`: Temporary directory for test files
- `sample_config`: Valid configuration for testing
- `config_file`: Temporary config file
- `mock_model_path`: Mock model directory
- `test_environment`: Complete test environment setup

### Test Data (`tests/fixtures/test_data.py`)

Reusable test data and mock objects:

- `SAMPLE_SFT_CONVERSATIONS`: Sample conversation data
- `SAMPLE_PREFERENCE_DATA`: Sample preference pairs
- `MockWandBRun`: Mock Weights & Biases run
- `MockHuggingFaceModel`: Mock HuggingFace model
- `MockTokenizer`: Mock tokenizer

### Test Utilities (`tests/utils.py`)

Helper functions for testing:

- `assert_config_equality()`: Deep config comparison
- `create_mock_checkpoint_structure()`: Mock checkpoint creation
- `validate_dataset_format()`: Dataset validation
- `mock_wandb()`: Context manager for mocking WandB
- `temporary_directory()`: Temporary directory context manager

## Property-Based Testing

### Hypothesis Strategies

Custom strategies for generating test data:

```python
# Configuration strategies
valid_config = composite(lambda draw: Config(...))
invalid_config = composite(lambda draw: InvalidConfig(...))

# Dataset strategies  
conversation_messages = composite(lambda draw: [...])
preference_pairs = composite(lambda draw: {...})

# Parameter strategies
learning_rates = floats(min_value=1e-6, max_value=1e-2)
batch_sizes = integers(min_value=1, max_value=64)
```

### Property Examples

Key properties tested:

1. **Configuration Serialization Round-Trip**: Any config can be saved and loaded identically
2. **Parameter Bounds Enforcement**: Invalid parameters are rejected
3. **Stage Configuration Subsetting**: Stage configs contain correct parameters
4. **Memory Adaptive Behavior**: System adapts to memory constraints
5. **Checkpoint Integrity**: Checkpoints preserve exact model state

## Mocking External Dependencies

### HuggingFace Transformers

```python
@pytest.fixture
def mock_transformers():
    with patch('transformers.AutoModelForCausalLM') as mock_model:
        mock_model.from_pretrained.return_value = MockHuggingFaceModel()
        yield mock_model
```

### Weights & Biases

```python
@contextmanager
def mock_wandb():
    with patch('wandb.init') as mock_init:
        mock_run = MockWandBRun()
        mock_init.return_value = mock_run
        yield mock_run
```

### Google Drive

```python
@contextmanager  
def mock_google_drive():
    with patch('google.auth.default') as mock_auth:
        yield mock_auth
```

## CI/CD Integration

### GitHub Actions

Automated testing in `.github/workflows/ci.yml`:

- **Lint Job**: Code quality checks
- **Test Job**: Cross-platform testing (Ubuntu, Windows, macOS)
- **GPU Tests**: Mocked GPU testing
- **Coverage**: Coverage reporting and upload
- **Security**: Vulnerability scanning

### Pre-commit Hooks

Automated checks before commits (`.pre-commit-config.yaml`):

- Code formatting (Black, isort)
- Linting (flake8, mypy)
- Security scanning (bandit)
- General file checks

## Performance Testing

### Execution Time Limits

```python
@assert_execution_time_under(5.0)  # 5 seconds max
def test_fast_operation():
    # Test implementation
    pass
```

### Memory Usage Monitoring

```python
def test_memory_efficient_operation():
    with measure_memory_usage() as memory:
        # Operation that should use < 1GB
        pass
    
    assert memory.peak_mb < 1000
```

## Best Practices

### Writing Unit Tests

1. **Test one thing**: Each test should verify one specific behavior
2. **Use descriptive names**: `test_config_validation_rejects_negative_learning_rate`
3. **Arrange-Act-Assert**: Clear test structure
4. **Mock external dependencies**: Keep tests isolated
5. **Test edge cases**: Empty inputs, boundary values, error conditions

### Writing Property Tests

1. **Focus on invariants**: Properties that should always hold
2. **Use appropriate strategies**: Generate realistic test data
3. **Add assumptions**: Filter out invalid combinations
4. **Test round-trip operations**: Serialize/deserialize, save/load
5. **Validate error handling**: Properties should hold even with errors

### Writing Integration Tests

1. **Test realistic scenarios**: Use representative data sizes
2. **Mock expensive operations**: Network calls, large model loading
3. **Verify end-to-end behavior**: Complete workflows work correctly
4. **Test error recovery**: System handles failures gracefully
5. **Use appropriate timeouts**: Don't let tests hang indefinitely

## Troubleshooting

### Common Issues

**Tests fail with import errors:**
```bash
# Ensure package is installed in development mode
pip install -e .
```

**Hypothesis tests are slow:**
```bash
# Use faster profile
export HYPOTHESIS_PROFILE=fast
```

**Coverage is below threshold:**
```bash
# Generate detailed coverage report
make coverage
# Open htmlcov/index.html to see uncovered lines
```

**GPU tests fail on CPU-only systems:**
```bash
# Run without GPU tests
pytest tests/ -m "not gpu"
```

### Debugging Test Failures

1. **Run with verbose output**: `pytest -v`
2. **Show full traceback**: `pytest --tb=long`
3. **Run single test**: `pytest tests/unit/test_config.py::test_specific_function`
4. **Use debugger**: `pytest --pdb`
5. **Check logs**: Look in test output for detailed error messages

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_component_behavior_condition`
2. **Add appropriate markers**: `@pytest.mark.unit`, etc.
3. **Update documentation**: Add new test categories to this README
4. **Ensure coverage**: New code should have corresponding tests
5. **Run full test suite**: `make ci` before submitting PR

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)