"""
Unit tests for Dataset Manager enhancements added in Task 2.2

Tests the new validation, preprocessing, and streaming enhancements.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rlhf_phi3.data.dataset_manager import DatasetManager
from rlhf_phi3.config.config_manager import Config
from tests.fixtures.test_data import MockTokenizer


class TestDatasetManagerEnhancements:
    """Test cases for Dataset Manager enhancements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.mock_tokenizer = MockTokenizer()
        self.mock_tokenizer.apply_chat_template = Mock()
        self.mock_tokenizer.chat_template = "test_template"
        self.mock_tokenizer.pad_token = "<pad>"
        self.mock_tokenizer.eos_token = "<eos>"
        self.mock_tokenizer.pad_token_id = 0
    
    def test_enhanced_sft_label_creation(self, temp_dir):
        """Test enhanced SFT label creation with token-level alignment."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock tokenizer encode method for enhanced label creation
        def mock_encode(text, add_special_tokens=False):
            if text == '<|assistant|>':
                return [100, 101]  # Mock assistant start tokens
            elif text == '<|end|>':
                return [102]  # Mock end token
            else:
                return [1, 2, 3]  # Default tokens
        
        self.mock_tokenizer.encode = mock_encode
        
        formatted_text = "<|user|>\nHello<|end|>\n<|assistant|>\nHi there!<|end|>"
        input_ids = [1, 2, 3, 100, 101, 4, 5, 6, 102, 7]
        
        labels = dm._create_sft_labels(formatted_text, input_ids)
        
        assert len(labels) == len(input_ids)
        # Check that some tokens are unmasked (not -100)
        unmasked_count = sum(1 for label in labels if label != -100)
        assert unmasked_count > 0
    
    def test_enhanced_sft_label_creation_fallback(self, temp_dir):
        """Test enhanced SFT label creation fallback mechanisms."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock tokenizer encode method for fallback testing
        def mock_encode(text, add_special_tokens=False):
            if 'Response:' in text:
                return [200, 201]  # Mock response indicator
            else:
                return [1, 2, 3]  # Default tokens
        
        self.mock_tokenizer.encode = mock_encode
        
        # Test with response indicator
        formatted_text = "Question: Hello\nResponse: Hi there!"
        input_ids = [1, 2, 3, 200, 201, 4, 5, 6]
        
        labels = dm._create_sft_labels(formatted_text, input_ids)
        
        assert len(labels) == len(input_ids)
        # Should find response indicator and unmask tokens after it
        unmasked_count = sum(1 for label in labels if label != -100)
        assert unmasked_count > 0
    
    @patch('psutil.Process')
    def test_memory_usage_stats(self, mock_process, temp_dir):
        """Test memory usage statistics collection."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock process memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.memory_percent.return_value = 5.0
        
        stats = dm.get_memory_usage_stats()
        
        assert 'process_memory_mb' in stats
        assert 'process_memory_percent' in stats
        assert 'cached_datasets' in stats
        assert 'estimated_cache_size_mb' in stats
        assert stats['process_memory_mb'] == 100.0
        assert stats['process_memory_percent'] == 5.0
    
    @patch('gc.collect')
    @patch('psutil.Process')
    def test_memory_optimization(self, mock_process, mock_gc_collect, temp_dir):
        """Test memory optimization functionality."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Add some items to cache
        dm._dataset_cache['test1'] = Mock()
        dm._dataset_cache['test2'] = Mock()
        
        # Mock memory info
        mock_memory_info_before = Mock()
        mock_memory_info_before.rss = 100 * 1024 * 1024  # 100 MB
        mock_memory_info_after = Mock()
        mock_memory_info_after.rss = 80 * 1024 * 1024   # 80 MB
        
        mock_process.return_value.memory_info.side_effect = [
            mock_memory_info_before, mock_memory_info_after
        ]
        mock_process.return_value.memory_percent.return_value = 5.0
        mock_gc_collect.return_value = 42  # Mock collected objects
        
        results = dm.optimize_memory_usage()
        
        assert results['cleared_cached_datasets'] == 2
        assert results['garbage_collected_objects'] == 42
        assert results['memory_freed_mb'] == 20.0
        assert results['memory_reduction_percent'] == 20.0
        assert len(dm._dataset_cache) == 0
    
    def test_create_streaming_dataloader(self, temp_dir):
        """Test streaming dataloader creation."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Create mock streaming dataset
        mock_dataset = Mock()
        mock_dataset.shuffle = Mock(return_value=mock_dataset)
        
        # Mock torch DataLoader
        with patch('rlhf_phi3.data.dataset_manager.DataLoader') as mock_dataloader:
            mock_dataloader.return_value = Mock()
            
            dataloader = dm.create_streaming_dataloader(
                mock_dataset, 
                batch_size=32, 
                buffer_size=1000
            )
            
            # Verify DataLoader was called with correct parameters
            mock_dataloader.assert_called_once()
            call_args = mock_dataloader.call_args
            assert call_args[1]['batch_size'] == 32
            assert call_args[1]['prefetch_factor'] == 2
    
    def test_enhanced_validation_with_performance_metrics(self, temp_dir):
        """Test enhanced validation with performance metrics."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        # Create sample data
        sample_data = [
            {
                'messages': [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        ] * 100
        
        mock_dataset.__iter__ = Mock(return_value=iter(sample_data))
        
        # Mock the batch validation method
        dm._validate_batch = Mock(return_value={
            'valid_count': 90,
            'invalid_count': 10,
            'format_errors': 5,
            'content_warnings': 2,
            'errors': [],
            'warnings': [],
            'quality_scores': [0.8] * 90
        })
        
        results = dm.validate_raw_dataset(mock_dataset, 'sft')
        
        assert results['is_valid'] is True
        assert 'validation_time' in results['statistics']
        assert 'throughput_examples_per_second' in results['statistics']
        assert results['statistics']['valid_examples'] == 90
        assert results['statistics']['invalid_examples'] == 10
        assert 'quality_std' in results['statistics']
    
    def test_enhanced_preprocessing_with_progress_tracking(self, temp_dir):
        """Test enhanced preprocessing with progress tracking."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Create mock dataset
        sample_data = [
            {
                'messages': [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        ] * 5  # Small dataset for testing
        
        class MockDataset:
            def __init__(self, data):
                self.data = data
                self.column_names = ['messages']
            
            def __len__(self):
                return len(self.data)
            
            def map(self, func, **kwargs):
                processed = []
                for item in self.data:
                    result = func(item)
                    if result is not None:
                        processed.append(result)
                return MockDataset(processed)
            
            def filter(self, func):
                filtered = [item for item in self.data if func(item)]
                return MockDataset(filtered)
        
        mock_dataset = MockDataset(sample_data)
        
        # Test preprocessing
        result = dm.preprocess_sft_data(mock_dataset, validate=True)
        
        assert len(result) > 0
        # Check that validation stats were collected
        assert dm._validation_stats['successful'] > 0
    
    def test_enhanced_pipeline_with_max_samples(self, temp_dir):
        """Test enhanced pipeline with max_samples parameter."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock the load methods
        mock_dataset = Mock()
        mock_dataset.take = Mock(return_value=mock_dataset)
        
        dm.load_sft_dataset = Mock(return_value=mock_dataset)
        dm.validate_raw_dataset = Mock(return_value={
            'is_valid': True, 
            'errors': [], 
            'warnings': [],
            'statistics': {'throughput_examples_per_second': 100.0}
        })
        dm.filter_dataset_content = Mock(return_value=mock_dataset)
        dm.preprocess_sft_data = Mock(return_value=mock_dataset)
        dm.validate_dataset_format = Mock(return_value=True)
        
        # Test pipeline with max_samples
        result = dm.create_dataset_pipeline(
            'sft', 
            streaming=True, 
            max_samples=1000
        )
        
        # Verify take was called with max_samples
        mock_dataset.take.assert_called_once_with(1000)
        assert result == mock_dataset
    
    def test_enhanced_preprocessing_error_recovery(self, temp_dir):
        """Test enhanced preprocessing error recovery mechanisms."""
        self.config.paths.cache_dir = str(temp_dir)
        dm = DatasetManager(self.config, tokenizer=self.mock_tokenizer)
        
        # Mock tokenizer to fail on first call, succeed on second
        call_count = 0
        def mock_tokenize_with_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Tokenization failed")
            return {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1]
            }
        
        dm._tokenize_text = mock_tokenize_with_failure
        
        # Create sample data with one problematic example
        sample_data = [
            {
                'messages': [
                    {"role": "user", "content": "This will fail"},
                    {"role": "assistant", "content": "Response"}
                ]
            },
            {
                'messages': [
                    {"role": "user", "content": "This will succeed"},
                    {"role": "assistant", "content": "Response"}
                ]
            }
        ]
        
        class MockDataset:
            def __init__(self, data):
                self.data = data
                self.column_names = ['messages']
            
            def map(self, func, **kwargs):
                processed = []
                for item in self.data:
                    result = func(item)
                    if result is not None:
                        processed.append(result)
                return MockDataset(processed)
            
            def filter(self, func):
                filtered = [item for item in self.data if func(item)]
                return MockDataset(filtered)
        
        mock_dataset = MockDataset(sample_data)
        
        # Test preprocessing with error recovery
        result = dm.preprocess_sft_data(mock_dataset, validate=True)
        
        # Should have one successful example and one failed
        assert len(result) == 1  # Only the successful one
        assert dm._validation_stats['tokenization_errors'] == 1
        assert dm._validation_stats['successful'] == 1


if __name__ == "__main__":
    pytest.main([__file__])