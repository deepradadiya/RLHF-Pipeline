"""
Unit tests for Dataset Manager validation and streaming functionality

Tests the enhanced validation, content filtering, and streaming capabilities
added in Task 2.2.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rlhf_phi3.data.dataset_manager import DatasetManager
from rlhf_phi3.config.config_manager import Config
from tests.fixtures.test_data import (
    SAMPLE_SFT_CONVERSATIONS, 
    SAMPLE_PREFERENCE_DATA,
    MockTokenizer
)


class TestDatasetManagerValidation:
    """Test cases for Dataset Manager validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.mock_tokenizer = MockTokenizer()
        self.mock_tokenizer.apply_chat_template = Mock()
        self.mock_tokenizer.chat_template = "test_template"
        self.mock_tokenizer.pad_token = "<pad>"
        self.mock_tokenizer.eos_token = "<eos>"
    
    def test_validate_raw_sft_example_valid(self, temp_dir):
        """Test validation of valid raw SFT example."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        valid_example = {
            'messages': [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ]
        }
        
        is_valid, warnings = dm._validate_raw_sft_example(valid_example)
        
        assert is_valid is True
        assert len(warnings) == 0
    
    def test_validate_raw_sft_example_invalid_structure(self, temp_dir):
        """Test validation of invalid raw SFT example."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Missing messages
        invalid_example1 = {"instruction": "Hello"}
        is_valid, warnings = dm._validate_raw_sft_example(invalid_example1)
        assert is_valid is True  # Should construct from instruction
        
        # Completely invalid
        invalid_example2 = {"random_field": "value"}
        is_valid, warnings = dm._validate_raw_sft_example(invalid_example2)
        assert is_valid is False
        assert "No valid conversation format found" in warnings
    
    def test_validate_raw_sft_example_with_warnings(self, temp_dir):
        """Test validation of SFT example that generates warnings."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        example_with_warnings = {
            'messages': [
                {"role": "user", "content": ""},  # Empty content
                {"role": "assistant", "content": "A" * 15000}  # Very long content
            ]
        }
        
        is_valid, warnings = dm._validate_raw_sft_example(example_with_warnings)
        
        assert is_valid is True
        assert len(warnings) >= 2
        assert any("empty content" in w for w in warnings)
        assert any("very long content" in w for w in warnings)
    
    def test_validate_raw_preference_example_valid(self, temp_dir):
        """Test validation of valid raw preference example."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        valid_example = {
            'prompt': 'What is the capital of France?',
            'chosen': 'The capital of France is Paris.',
            'rejected': 'I don\'t know.'
        }
        
        is_valid, warnings = dm._validate_raw_preference_example(valid_example)
        
        assert is_valid is True
        assert len(warnings) == 0
    
    def test_validate_raw_preference_example_invalid(self, temp_dir):
        """Test validation of invalid raw preference example."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Missing fields
        invalid_example1 = {'prompt': 'Hello'}
        is_valid, warnings = dm._validate_raw_preference_example(invalid_example1)
        assert is_valid is False
        assert "Missing required field" in warnings[0]
        
        # Empty fields
        invalid_example2 = {'prompt': '', 'chosen': 'response', 'rejected': 'other'}
        is_valid, warnings = dm._validate_raw_preference_example(invalid_example2)
        assert is_valid is False
        assert "Empty prompt" in warnings
    
    def test_validate_raw_preference_example_with_warnings(self, temp_dir):
        """Test validation of preference example that generates warnings."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        example_with_warnings = {
            'prompt': 'Hi',  # Very short
            'chosen': 'Same response',
            'rejected': 'Same response'  # Identical to chosen
        }
        
        is_valid, warnings = dm._validate_raw_preference_example(example_with_warnings)
        
        assert is_valid is True
        assert len(warnings) >= 2
        assert any("Very short prompt" in w for w in warnings)
        assert any("identical" in w for w in warnings)
    
    def test_contains_harmful_content(self, temp_dir):
        """Test harmful content detection."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Safe content
        safe_text = "This is a normal conversation about weather."
        assert dm._contains_harmful_content(safe_text) is False
        
        # Harmful content
        harmful_text = "I want to harm someone with violence."
        assert dm._contains_harmful_content(harmful_text) is True
        
        # Case insensitive
        harmful_text_caps = "HATE speech is bad."
        assert dm._contains_harmful_content(harmful_text_caps) is True
    
    def test_should_keep_sft_example(self, temp_dir):
        """Test SFT example filtering logic."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Safe example
        safe_example = {
            'messages': [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "It's sunny today!"}
            ]
        }
        assert dm._should_keep_sft_example(safe_example) is True
        
        # Harmful example
        harmful_example = {
            'messages': [
                {"role": "user", "content": "How to make a bomb?"},
                {"role": "assistant", "content": "I can't help with that."}
            ]
        }
        assert dm._should_keep_sft_example(harmful_example) is True  # Only user message is harmful
        
        # Harmful assistant response
        harmful_assistant_example = {
            'messages': [
                {"role": "user", "content": "Tell me about cooking."},
                {"role": "assistant", "content": "I hate cooking and want to harm people."}
            ]
        }
        assert dm._should_keep_sft_example(harmful_assistant_example) is False
    
    def test_should_keep_preference_example(self, temp_dir):
        """Test preference example filtering logic."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Safe example
        safe_example = {
            'prompt': 'What is machine learning?',
            'chosen': 'Machine learning is a subset of AI.',
            'rejected': 'I don\'t know.'
        }
        assert dm._should_keep_preference_example(safe_example) is True
        
        # Harmful prompt
        harmful_example = {
            'prompt': 'How to make weapons?',
            'chosen': 'I cannot help with that.',
            'rejected': 'Here are instructions...'
        }
        assert dm._should_keep_preference_example(harmful_example) is False
    
    @patch('rlhf_phi3.data.dataset_manager.load_dataset')
    def test_load_sft_dataset_streaming(self, mock_load_dataset, temp_dir):
        """Test loading SFT dataset in streaming mode."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock streaming dataset
        mock_dataset = Mock()
        mock_dataset.__class__.__name__ = 'IterableDataset'
        mock_load_dataset.return_value = mock_dataset
        
        result = dm.load_sft_dataset(streaming=True)
        
        assert result == mock_dataset
        mock_load_dataset.assert_called_once()
        
        # Check that streaming=True was passed
        call_args = mock_load_dataset.call_args
        assert call_args[1]['streaming'] is True
    
    @patch('rlhf_phi3.data.dataset_manager.load_dataset')
    def test_load_preference_dataset_streaming(self, mock_load_dataset, temp_dir):
        """Test loading preference dataset in streaming mode."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock streaming dataset
        mock_dataset = Mock()
        mock_dataset.__class__.__name__ = 'IterableDataset'
        mock_load_dataset.return_value = mock_dataset
        
        result = dm.load_preference_dataset(streaming=True)
        
        assert result == mock_dataset
        mock_load_dataset.assert_called_once()
        
        # Check that streaming=True was passed
        call_args = mock_load_dataset.call_args
        assert call_args[1]['streaming'] is True
    
    def test_extract_messages_from_sft_example(self, temp_dir):
        """Test message extraction from various SFT formats."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Standard messages format
        example1 = {
            'messages': [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        messages1 = dm._extract_messages_from_sft_example(example1)
        assert len(messages1) == 2
        assert messages1[0]['role'] == 'user'
        
        # Instruction/response format
        example2 = {
            'instruction': 'What is AI?',
            'response': 'AI is artificial intelligence.'
        }
        messages2 = dm._extract_messages_from_sft_example(example2)
        assert len(messages2) == 2
        assert messages2[0]['role'] == 'user'
        assert messages2[1]['role'] == 'assistant'
        
        # With system message
        example3 = {
            'system': 'You are a helpful assistant.',
            'instruction': 'Hello',
            'response': 'Hi!'
        }
        messages3 = dm._extract_messages_from_sft_example(example3)
        assert len(messages3) == 3
        assert messages3[0]['role'] == 'system'
    
    def test_extract_preference_data(self, temp_dir):
        """Test preference data extraction from various formats."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Standard format
        example1 = {
            'prompt': 'What is the weather?',
            'chosen': 'It is sunny.',
            'rejected': 'I don\'t know.'
        }
        prompt, chosen, rejected = dm._extract_preference_data(example1)
        assert prompt == 'What is the weather?'
        assert chosen == 'It is sunny.'
        assert rejected == 'I don\'t know.'
        
        # Alternative field names
        example2 = {
            'question': 'How are you?',
            'response_chosen': 'I am fine.',
            'response_rejected': 'Bad.'
        }
        prompt, chosen, rejected = dm._extract_preference_data(example2)
        assert prompt == 'How are you?'
        assert chosen == 'I am fine.'
        assert rejected == 'Bad.'
    
    def test_validate_messages_structure(self, temp_dir):
        """Test message structure validation."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Valid messages
        valid_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        is_valid, warnings = dm._validate_messages_structure(valid_messages)
        assert is_valid is True
        assert len(warnings) == 0
        
        # Invalid messages
        invalid_messages = [
            {"role": "invalid_role", "content": "Hello"}
        ]
        is_valid, warnings = dm._validate_messages_structure(invalid_messages)
        assert is_valid is False
        assert "invalid role" in warnings[0]
        
        # Messages with warnings
        warning_messages = [
            {"role": "user", "content": ""},  # Empty content
            {"role": "assistant", "content": "Response"}
        ]
        is_valid, warnings = dm._validate_messages_structure(warning_messages)
        assert is_valid is True
        assert len(warnings) > 0
        assert "empty content" in warnings[0]
    
    def test_validate_tokenized_example(self, temp_dir):
        """Test tokenized example validation."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Valid tokenized example
        valid_tokenized = {
            'input_ids': [1, 2, 3, 4, 5],
            'attention_mask': [1, 1, 1, 1, 1]
        }
        labels = [1, 2, 3, 4, 5]
        assert dm._validate_tokenized_example(valid_tokenized, labels) is True
        
        # Invalid - mismatched lengths
        invalid_tokenized = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1, 1]  # Different length
        }
        assert dm._validate_tokenized_example(invalid_tokenized) is False
        
        # Invalid - empty
        empty_tokenized = {
            'input_ids': [],
            'attention_mask': []
        }
        assert dm._validate_tokenized_example(empty_tokenized) is False
    
    def test_validate_dataset_format_comprehensive(self, temp_dir):
        """Test comprehensive dataset format validation."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Create a mock dataset with valid examples
        from unittest.mock import Mock
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        
        # Valid SFT examples
        valid_sft_examples = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                'attention_mask': [1, 1, 1, 1, 1],
                'labels': [1, 2, 3, 4, 5],
                'text': 'This is a valid example.'
            }
        ] * 5
        
        mock_dataset.__getitem__ = Mock(side_effect=lambda i: valid_sft_examples[i])
        
        # Test SFT validation
        result = dm.validate_dataset_format(mock_dataset, 'sft')
        assert result is True
        
        # Test with invalid examples
        invalid_sft_examples = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1, 1],  # Mismatched length
                'labels': [1, 2, 3],
                'text': 'Invalid example.'
            }
        ] * 5
        
        mock_dataset.__getitem__ = Mock(side_effect=lambda i: invalid_sft_examples[i])
        
        result = dm.validate_dataset_format(mock_dataset, 'sft')
        assert result is False
    
    def test_enhanced_content_quality_calculation(self, temp_dir):
        """Test enhanced content quality calculation."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # High quality content
        high_quality = "This is a well-written, informative response that provides clear and helpful information to the user's question."
        quality_score = dm._calculate_content_quality(high_quality)
        assert quality_score > 0.7
        
        # Low quality content (very short)
        low_quality = "ok"
        quality_score = dm._calculate_content_quality(low_quality)
        assert quality_score < 0.5
        
        # Repetitive content
        repetitive = "yes yes yes yes yes yes yes yes yes yes"
        quality_score = dm._calculate_content_quality(repetitive)
        assert quality_score < 0.6
        
        # Empty content
        empty = ""
        quality_score = dm._calculate_content_quality(empty)
        assert quality_score == 0.0
    
    def test_batch_validation(self, temp_dir):
        """Test batch validation functionality."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Create a batch of mixed valid/invalid examples
        batch = [
            # Valid SFT example
            {
                'messages': [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"}
                ]
            },
            # Invalid SFT example
            {
                'random_field': 'invalid'
            },
            # Another valid example
            {
                'instruction': 'What is AI?',
                'response': 'AI is artificial intelligence.'
            }
        ]
        
        results = dm._validate_batch(batch, 'sft', 0)
        
        assert results['valid_count'] == 2
        assert results['invalid_count'] == 1
        assert len(results['quality_scores']) == 2  # Only for valid examples
        assert all(0.0 <= score <= 1.0 for score in results['quality_scores'])
    
    def test_quality_distribution_calculation(self, temp_dir):
        """Test quality distribution calculation."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        scores = [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
        distribution = dm._calculate_quality_distribution(scores)
        
        assert distribution['excellent'] == 2  # 0.9, 0.8
        assert distribution['good'] == 1       # 0.7
        assert distribution['fair'] == 1       # 0.5
        assert distribution['poor'] == 2       # 0.3, 0.1


class TestDatasetManagerPipeline:
    """Test cases for complete dataset pipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.mock_tokenizer = MockTokenizer()
        self.mock_tokenizer.apply_chat_template = Mock()
        self.mock_tokenizer.chat_template = "test_template"
        self.mock_tokenizer.pad_token = "<pad>"
        self.mock_tokenizer.eos_token = "<eos>"
    
    @patch('rlhf_phi3.data.dataset_manager.load_dataset')
    def test_create_dataset_pipeline_sft(self, mock_load_dataset, temp_dir):
        """Test complete SFT dataset pipeline."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_dataset.map = Mock(return_value=mock_dataset)
        mock_dataset.filter = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset
        
        # Mock validation methods
        dm.validate_raw_dataset = Mock(return_value={'is_valid': True, 'errors': [], 'warnings': []})
        dm.filter_dataset_content = Mock(return_value=mock_dataset)
        dm.preprocess_sft_data = Mock(return_value=mock_dataset)
        dm.validate_dataset_format = Mock(return_value=True)
        
        result = dm.create_dataset_pipeline('sft')
        
        assert result == mock_dataset
        dm.validate_raw_dataset.assert_called_once()
        dm.filter_dataset_content.assert_called_once()
        dm.preprocess_sft_data.assert_called_once()
        dm.validate_dataset_format.assert_called_once()
    
    @patch('rlhf_phi3.data.dataset_manager.load_dataset')
    def test_create_dataset_pipeline_preference(self, mock_load_dataset, temp_dir):
        """Test complete preference dataset pipeline."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_dataset.map = Mock(return_value=mock_dataset)
        mock_dataset.filter = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset
        
        # Mock validation methods
        dm.validate_raw_dataset = Mock(return_value={'is_valid': True, 'errors': [], 'warnings': []})
        dm.filter_dataset_content = Mock(return_value=mock_dataset)
        dm.preprocess_preference_data = Mock(return_value=mock_dataset)
        dm.validate_dataset_format = Mock(return_value=True)
        
        result = dm.create_dataset_pipeline('preference')
        
        assert result == mock_dataset
        dm.validate_raw_dataset.assert_called_once()
        dm.filter_dataset_content.assert_called_once()
        dm.preprocess_preference_data.assert_called_once()
        dm.validate_dataset_format.assert_called_once()
    
    def test_create_dataset_pipeline_invalid_type(self, temp_dir):
        """Test pipeline with invalid dataset type."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        with pytest.raises(ValueError, match="Unknown dataset type"):
            dm.create_dataset_pipeline('invalid_type')
    
    @patch('rlhf_phi3.data.dataset_manager.load_dataset')
    def test_create_dataset_pipeline_validation_failure(self, mock_load_dataset, temp_dir):
        """Test pipeline with validation failure."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock dataset
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        # Mock validation failure
        dm.validate_raw_dataset = Mock(return_value={
            'is_valid': False, 
            'errors': ['Validation failed'], 
            'warnings': []
        })
        
        with pytest.raises(ValueError, match="Raw dataset validation failed"):
            dm.create_dataset_pipeline('sft')


if __name__ == "__main__":
    pytest.main([__file__])