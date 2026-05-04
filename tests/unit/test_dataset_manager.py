"""
Unit tests for Dataset Manager

Tests the core functionality of the Dataset Manager component including
dataset loading, preprocessing, and chat template formatting.
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


class TestDatasetManager:
    """Test cases for Dataset Manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        
        # Use mock tokenizer from test fixtures
        self.mock_tokenizer = MockTokenizer()
        self.mock_tokenizer.apply_chat_template = Mock()
        self.mock_tokenizer.chat_template = "test_template"
        self.mock_tokenizer.pad_token = "<pad>"
        self.mock_tokenizer.eos_token = "<eos>"
    
    def test_initialization_with_tokenizer(self, temp_dir):
        """Test DatasetManager initialization with provided tokenizer."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        assert dm.config == self.config
        assert dm.tokenizer == self.mock_tokenizer
        assert dm.cache_dir.exists()
        assert dm._dataset_cache == {}
    
    @patch('rlhf_phi3.data.dataset_manager.AutoTokenizer')
    def test_initialization_without_tokenizer(self, mock_auto_tokenizer, temp_dir):
        """Test DatasetManager initialization without tokenizer."""
        self.config.paths.cache_dir = str(temp_dir)
        mock_auto_tokenizer.from_pretrained.return_value = self.mock_tokenizer
        
        dm = DatasetManager(self.config)
        
        assert dm.config == self.config
        assert dm.tokenizer == self.mock_tokenizer
        mock_auto_tokenizer.from_pretrained.assert_called_once()
    
    def test_manual_phi3_format_basic(self):
        """Test manual Phi-3 chat formatting with basic messages."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        
        result = dm._manual_phi3_format(messages)
        
        expected = "<|user|>\nHello, how are you?<|end|>\n<|assistant|>\nI'm doing well, thank you!<|end|>"
        assert result == expected
    
    def test_manual_phi3_format_with_system(self):
        """Test manual Phi-3 chat formatting with system message."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = dm._manual_phi3_format(messages)
        
        expected = ("<|system|>\nYou are a helpful assistant.<|end|>\n"
                   "<|user|>\nHello!<|end|>\n"
                   "<|assistant|>\nHi there!<|end|>")
        assert result == expected
    
    def test_format_chat_template_with_tokenizer_template(self):
        """Test chat template formatting using tokenizer's template."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        messages = [{"role": "user", "content": "Test message"}]
        expected_output = "formatted_by_tokenizer"
        
        self.mock_tokenizer.apply_chat_template.return_value = expected_output
        
        result = dm.format_chat_template(messages)
        
        assert result == expected_output
        self.mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=False
        )
    
    def test_format_chat_template_fallback(self):
        """Test chat template formatting fallback to manual format."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Make tokenizer template fail
        self.mock_tokenizer.apply_chat_template.side_effect = Exception("Template error")
        
        messages = [{"role": "user", "content": "Test message"}]
        
        result = dm.format_chat_template(messages)
        
        expected = "<|user|>\nTest message<|end|>"
        assert result == expected
    
    def test_format_chat_template_validation(self):
        """Test chat template formatting input validation."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Test empty messages
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            dm.format_chat_template([])
        
        # Test invalid message format
        with pytest.raises(ValueError, match="Message 0 must be a dictionary"):
            dm.format_chat_template(["invalid"])
        
        # Test missing keys
        with pytest.raises(ValueError, match="Message 0 must have 'role' and 'content' keys"):
            dm.format_chat_template([{"role": "user"}])
        
        # Test invalid role
        with pytest.raises(ValueError, match="Message 0 role must be"):
            dm.format_chat_template([{"role": "invalid", "content": "test"}])
    
    def test_tokenize_text(self):
        """Test text tokenization with proper strategies."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock tokenizer response
        mock_output = {
            'input_ids': [1, 2, 3, 4],
            'attention_mask': [1, 1, 1, 1]
        }
        self.mock_tokenizer.return_value = mock_output
        
        result = dm._tokenize_text("test text", max_length=512)
        
        assert result == mock_output
        self.mock_tokenizer.assert_called_once_with(
            "test text",
            max_length=512,
            padding=False,
            truncation=True,
            return_tensors=None,
            add_special_tokens=True
        )
    
    def test_create_sft_labels(self):
        """Test SFT label creation."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        formatted_text = "<|user|>\nHello<|end|>\n<|assistant|>\nHi<|end|>"
        input_ids = [1, 2, 3, 4, 5]
        
        labels = dm._create_sft_labels(formatted_text, input_ids)
        
        # For now, labels should be the same as input_ids
        assert labels == input_ids
    
    def test_validate_sft_example_valid(self):
        """Test SFT example validation with valid example."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        valid_example = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1],
            'labels': [1, 2, 3]
        }
        
        assert dm._validate_sft_example(valid_example) is True
    
    def test_validate_sft_example_invalid(self):
        """Test SFT example validation with invalid examples."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Missing field
        invalid_example1 = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1]
            # Missing labels
        }
        assert dm._validate_sft_example(invalid_example1) is False
        
        # Wrong type
        invalid_example2 = {
            'input_ids': "not a list",
            'attention_mask': [1, 1, 1],
            'labels': [1, 2, 3]
        }
        assert dm._validate_sft_example(invalid_example2) is False
        
        # Empty field
        invalid_example3 = {
            'input_ids': [],
            'attention_mask': [1, 1, 1],
            'labels': [1, 2, 3]
        }
        assert dm._validate_sft_example(invalid_example3) is False
        
        # Mismatched lengths
        invalid_example4 = {
            'input_ids': [1, 2],
            'attention_mask': [1, 1, 1],
            'labels': [1, 2, 3]
        }
        assert dm._validate_sft_example(invalid_example4) is False
    
    def test_validate_preference_example_valid(self):
        """Test preference example validation with valid example."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        valid_example = {
            'chosen_input_ids': [1, 2, 3],
            'chosen_attention_mask': [1, 1, 1],
            'rejected_input_ids': [4, 5, 6, 7],
            'rejected_attention_mask': [1, 1, 1, 1]
        }
        
        assert dm._validate_preference_example(valid_example) is True
    
    def test_validate_preference_example_invalid(self):
        """Test preference example validation with invalid examples."""
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Missing field
        invalid_example1 = {
            'chosen_input_ids': [1, 2, 3],
            'chosen_attention_mask': [1, 1, 1],
            'rejected_input_ids': [4, 5, 6]
            # Missing rejected_attention_mask
        }
        assert dm._validate_preference_example(invalid_example1) is False
        
        # Mismatched chosen lengths
        invalid_example2 = {
            'chosen_input_ids': [1, 2],
            'chosen_attention_mask': [1, 1, 1],
            'rejected_input_ids': [4, 5, 6],
            'rejected_attention_mask': [1, 1, 1]
        }
        assert dm._validate_preference_example(invalid_example2) is False
    
    def test_clear_cache(self, temp_dir):
        """Test cache clearing functionality."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Add some items to cache
        dm._dataset_cache['test1'] = Mock()
        dm._dataset_cache['test2'] = Mock()
        
        assert len(dm._dataset_cache) == 2
        
        dm.clear_cache()
        
        assert len(dm._dataset_cache) == 0
    
    @patch('rlhf_phi3.data.dataset_manager.load_dataset')
    def test_load_sft_dataset_success(self, mock_load_dataset, temp_dir):
        """Test successful SFT dataset loading."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset
        
        result = dm.load_sft_dataset()
        
        assert result == mock_dataset
        mock_load_dataset.assert_called_once()
        
        # Check caching
        cache_key = f"sft_{self.config.datasets.sft.name}_{self.config.datasets.sft.split}_{self.config.datasets.sft.max_samples}"
        assert cache_key in dm._dataset_cache
    
    @patch('rlhf_phi3.data.dataset_manager.load_dataset')
    def test_load_preference_dataset_success(self, mock_load_dataset, temp_dir):
        """Test successful preference dataset loading."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=500)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset
        
        result = dm.load_preference_dataset()
        
        assert result == mock_dataset
        mock_load_dataset.assert_called_once()
        
        # Check caching
        cache_key = f"pref_{self.config.datasets.preference.name}_{self.config.datasets.preference.split}_{self.config.datasets.preference.max_samples}"
        assert cache_key in dm._dataset_cache
    
    def test_preprocess_sft_data_with_sample_data(self, sample_sft_data, temp_dir):
        """Test SFT data preprocessing with sample data."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Create mock dataset from sample data
        mock_dataset = Mock()
        mock_dataset.column_names = ['messages']
        
        def mock_map(func, **kwargs):
            # Process each example through the function
            processed = []
            for example in sample_sft_data:
                result = func(example)
                if result is not None:
                    processed.append(result)
            
            # Return mock dataset with processed data
            result_dataset = Mock()
            result_dataset.__len__ = Mock(return_value=len(processed))
            result_dataset.filter = Mock(return_value=result_dataset)
            return result_dataset
        
        mock_dataset.map = mock_map
        
        result = dm.preprocess_sft_data(mock_dataset)
        assert result is not None
    
    def test_preprocess_preference_data_with_sample_data(self, sample_preference_data, temp_dir):
        """Test preference data preprocessing with sample data."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Create mock dataset from sample data
        mock_dataset = Mock()
        mock_dataset.column_names = ['prompt', 'chosen', 'rejected']
        
        def mock_map(func, **kwargs):
            # Process each example through the function
            processed = []
            for example in sample_preference_data:
                result = func(example)
                if result is not None:
                    processed.append(result)
            
            # Return mock dataset with processed data
            result_dataset = Mock()
            result_dataset.__len__ = Mock(return_value=len(processed))
            result_dataset.filter = Mock(return_value=result_dataset)
            return result_dataset
        
        mock_dataset.map = mock_map
        
        result = dm.preprocess_preference_data(mock_dataset)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])