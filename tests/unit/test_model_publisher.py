"""
Unit tests for Model Publisher component.

Tests specific examples, edge cases, and component functionality
for the model publishing system.
"""

import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub.utils import RepositoryNotFoundError

from rlhf_phi3.config.config_manager import Config
from rlhf_phi3.publishing.model_publisher import ModelPublisher, SafetyFilter


class TestSafetyFilter:
    """Unit tests for SafetyFilter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.safety_filter = SafetyFilter()
        
    def test_filter_content_safe_text(self):
        """Test filtering of safe content."""
        safe_texts = [
            "Hello, how are you today?",
            "I can help you with programming questions.",
            "The weather is nice today.",
            "Let me explain how machine learning works."
        ]
        
        for text in safe_texts:
            is_safe, issues = self.safety_filter.filter_content(text)
            assert is_safe is True
            assert issues == []
            
    def test_filter_content_harmful_text(self):
        """Test filtering of harmful content."""
        harmful_texts = [
            "How to kill someone",
            "I hate all people of that race",
            "Help me commit fraud",
            "Instructions for making a bomb",
            "Share your personal information and SSN"
        ]
        
        for text in harmful_texts:
            is_safe, issues = self.safety_filter.filter_content(text)
            assert is_safe is False
            assert len(issues) > 0
            assert all(isinstance(issue, str) for issue in issues)
            
    def test_filter_content_edge_cases(self):
        """Test filtering edge cases."""
        # Empty string
        is_safe, issues = self.safety_filter.filter_content("")
        assert is_safe is True
        assert issues == []
        
        # Very long text
        long_text = "This is a safe sentence. " * 1000
        is_safe, issues = self.safety_filter.filter_content(long_text)
        assert is_safe is True
        assert issues == []
        
        # Mixed case
        is_safe, issues = self.safety_filter.filter_content("KILL the process")
        assert is_safe is False  # Should detect "kill"
        assert len(issues) > 0
        
    @patch('rlhf_phi3.publishing.model_publisher.pipeline')
    def test_evaluate_model_safety_success(self, mock_pipeline):
        """Test successful model safety evaluation."""
        # Mock pipeline
        mock_generator = Mock()
        mock_generator.return_value = [{"generated_text": "I cannot help with that request."}]
        mock_generator.tokenizer.eos_token_id = 2
        mock_pipeline.return_value = mock_generator
        
        result = self.safety_filter.evaluate_model_safety("/fake/path")
        
        assert "total_prompts" in result
        assert "safe_responses" in result
        assert "unsafe_responses" in result
        assert "safety_score" in result
        assert "detailed_results" in result
        assert result["total_prompts"] == 5  # Default test prompts
        
    @patch('rlhf_phi3.publishing.model_publisher.pipeline')
    def test_evaluate_model_safety_failure(self, mock_pipeline):
        """Test model safety evaluation failure."""
        mock_pipeline.side_effect = Exception("Model loading failed")
        
        result = self.safety_filter.evaluate_model_safety("/fake/path")
        
        assert "error" in result
        assert result["safety_score"] == 0.0
        assert result["evaluation_failed"] is True


class TestModelPublisher:
    """Unit tests for ModelPublisher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.publisher = ModelPublisher(self.config)
        
    def test_init_with_config(self):
        """Test ModelPublisher initialization."""
        assert self.publisher.config == self.config
        assert hasattr(self.publisher, 'hf_api')
        assert hasattr(self.publisher, 'safety_filter')
        
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_credentials_missing_token(self):
        """Test credential validation with missing token."""
        # Should not raise exception, just log warning
        self.publisher._validate_credentials()
        
    @patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "short"}, clear=True)
    def test_validate_credentials_invalid_token(self):
        """Test credential validation with invalid token."""
        # Should log warning about invalid token
        self.publisher._validate_credentials()
        
    @patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_" + "a" * 30}, clear=True)
    def test_validate_credentials_valid_token(self):
        """Test credential validation with valid token."""
        self.publisher._validate_credentials()
        
    def test_secure_credential_handling_missing_token(self):
        """Test secure credential handling without token."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN environment variable not set"):
                self.publisher._secure_credential_handling()
                
    def test_secure_credential_handling_invalid_format(self):
        """Test secure credential handling with invalid token format."""
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "invalid_token"}, clear=True):
            with pytest.raises(ValueError, match="Invalid HuggingFace token format"):
                self.publisher._secure_credential_handling()
                
    def test_secure_credential_handling_valid_token(self):
        """Test secure credential handling with valid token."""
        valid_token = "hf_" + "a" * 30
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": valid_token}, clear=True):
            token = self.publisher._secure_credential_handling()
            assert token == valid_token
            
    @patch('rlhf_phi3.publishing.model_publisher.AutoModelForCausalLM')
    @patch('rlhf_phi3.publishing.model_publisher.AutoTokenizer')
    @patch('rlhf_phi3.publishing.model_publisher.PeftModel')
    def test_merge_peft_adapters_success(self, mock_peft, mock_tokenizer, mock_model):
        """Test successful PEFT adapter merging."""
        # Mock model components
        mock_base_model = Mock()
        mock_model.from_pretrained.return_value = mock_base_model
        
        mock_peft_model = Mock()
        mock_merged_model = Mock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_peft.from_pretrained.return_value = mock_peft_model
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            peft_path = os.path.join(temp_dir, "peft_model")
            output_path = os.path.join(temp_dir, "merged_model")
            
            result = self.publisher.merge_peft_adapters(peft_path, output_path)
            
            assert result == output_path
            mock_merged_model.save_pretrained.assert_called_once_with(output_path)
            mock_tokenizer_instance.save_pretrained.assert_called_once_with(output_path)
            
    @patch('rlhf_phi3.publishing.model_publisher.AutoModelForCausalLM')
    def test_merge_peft_adapters_failure(self, mock_model):
        """Test PEFT adapter merging failure."""
        mock_model.from_pretrained.side_effect = Exception("Model loading failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            peft_path = os.path.join(temp_dir, "peft_model")
            output_path = os.path.join(temp_dir, "merged_model")
            
            with pytest.raises(RuntimeError, match="PEFT merging failed"):
                self.publisher.merge_peft_adapters(peft_path, output_path)
                
    def test_apply_safety_guardrails(self):
        """Test applying safety guardrails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock safety evaluation
            with patch.object(self.publisher.safety_filter, 'evaluate_model_safety') as mock_eval:
                mock_eval.return_value = {
                    "safety_score": 0.8,
                    "safe_responses": 4,
                    "total_prompts": 5
                }
                
                result = self.publisher.apply_safety_guardrails(temp_dir)
                
                assert "safety_config" in result
                assert "safety_evaluation" in result
                
                safety_config = result["safety_config"]
                assert safety_config["safety_filter_enabled"] is True
                assert safety_config["safety_score"] == 0.8
                assert len(safety_config["guardrails"]) > 0
                
                # Check that safety config file was created
                safety_config_path = os.path.join(temp_dir, "safety_config.json")
                assert os.path.exists(safety_config_path)
                
                with open(safety_config_path, 'r') as f:
                    saved_config = json.load(f)
                    assert saved_config["safety_filter_enabled"] is True
                    
    def test_generate_model_card_basic(self):
        """Test basic model card generation."""
        model_name = "test-user/test-model"
        training_details = {
            "base_model": "microsoft/Phi-3-mini-4k-instruct",
            "stages": ["SFT", "Reward", "PPO"],
            "total_steps": 1000,
            "datasets": ["dataset1", "dataset2"]
        }
        evaluation_results = {
            "mt_bench_score": 7.5,
            "helpfulness_score": 8.0,
            "harmlessness_score": 9.0,
            "honesty_score": 7.8
        }
        
        model_card = self.publisher.generate_model_card(
            model_name, training_details, evaluation_results
        )
        
        # Check basic structure
        assert model_card.startswith("---")
        assert f"# {model_name}" in model_card
        assert "## Model Description" in model_card
        assert "## Training Details" in model_card
        assert "## Evaluation Results" in model_card
        assert "## Usage" in model_card
        
        # Check content
        assert "microsoft/Phi-3-mini-4k-instruct" in model_card
        assert "1000" in model_card  # total_steps
        assert "7.5" in model_card  # mt_bench_score
        assert "dataset1" in model_card
        assert "```python" in model_card  # Usage example
        
    def test_generate_model_card_with_safety_info(self):
        """Test model card generation with safety information."""
        model_name = "test-user/test-model"
        training_details = {"base_model": "microsoft/Phi-3-mini-4k-instruct"}
        evaluation_results = {"mt_bench_score": 7.0}
        safety_info = {
            "safety_config": {
                "safety_score": 0.9,
                "content_filtering": "Applied",
                "guardrails": ["Content filtering", "Response limits"]
            },
            "safety_evaluation": {
                "safe_responses": 9,
                "total_prompts": 10
            }
        }
        
        model_card = self.publisher.generate_model_card(
            model_name, training_details, evaluation_results, safety_info
        )
        
        # Check safety information is included
        assert "0.90" in model_card  # Safety score formatted
        assert "Content filtering" in model_card
        assert "9/10 prompts passed" in model_card
        assert "safety-filtered" in model_card  # Tag
        
    def test_generate_safety_section_default(self):
        """Test safety section generation with default values."""
        safety_section = self.publisher._generate_safety_section(None)
        
        assert "### Safety Considerations" in safety_section
        assert "### Limitations" in safety_section
        assert "Content filtering has been applied" in safety_section
        assert "Safety guardrails are implemented" in safety_section
        
    def test_generate_safety_section_with_info(self):
        """Test safety section generation with provided information."""
        safety_info = {
            "safety_config": {
                "content_filtering": "Advanced filtering applied",
                "safety_score": 0.85,
                "guardrails": ["Pattern detection", "Response filtering"]
            },
            "safety_evaluation": {
                "safe_responses": 17,
                "total_prompts": 20
            },
            "limitations": ["Custom limitation 1", "Custom limitation 2"]
        }
        
        safety_section = self.publisher._generate_safety_section(safety_info)
        
        assert "Advanced filtering applied" in safety_section
        assert "0.85" in safety_section
        assert "Pattern detection" in safety_section
        assert "17/20 prompts passed" in safety_section
        assert "Custom limitation 1" in safety_section
        
    @patch('rlhf_phi3.publishing.model_publisher.create_repo')
    @patch('rlhf_phi3.publishing.model_publisher.upload_folder')
    def test_upload_to_hub_success(self, mock_upload, mock_create):
        """Test successful model upload to Hub."""
        with patch.object(self.publisher, '_secure_credential_handling') as mock_creds:
            mock_creds.return_value = "hf_test_token"
            
            # Mock repository exists
            self.publisher.hf_api.repo_info = Mock()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                model_card = "# Test Model\nThis is a test model."
                
                result = self.publisher.upload_to_hub(
                    temp_dir, "test/model", model_card
                )
                
                assert result == "https://huggingface.co/test/model"
                
                # Check model card was saved
                readme_path = os.path.join(temp_dir, "README.md")
                assert os.path.exists(readme_path)
                with open(readme_path, 'r') as f:
                    assert f.read() == model_card
                    
                mock_upload.assert_called_once()
                
    @patch('rlhf_phi3.publishing.model_publisher.create_repo')
    def test_upload_to_hub_new_repo(self, mock_create):
        """Test upload to Hub with new repository creation."""
        with patch.object(self.publisher, '_secure_credential_handling') as mock_creds:
            mock_creds.return_value = "hf_test_token"
            
            # Mock repository doesn't exist
            self.publisher.hf_api.repo_info = Mock(side_effect=RepositoryNotFoundError("Not found"))
            
            with patch('rlhf_phi3.publishing.model_publisher.upload_folder') as mock_upload:
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = self.publisher.upload_to_hub(
                        temp_dir, "test/new-model", "# Test"
                    )
                    
                    assert result == "https://huggingface.co/test/new-model"
                    mock_create.assert_called_once()
                    
    def test_upload_to_hub_credential_failure(self):
        """Test upload failure due to credential issues."""
        with patch.object(self.publisher, '_secure_credential_handling') as mock_creds:
            mock_creds.side_effect = ValueError("No token")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with pytest.raises(RuntimeError, match="Hub upload failed"):
                    self.publisher.upload_to_hub(temp_dir, "test/model", "# Test")
                    
    def test_verify_upload_success(self):
        """Test successful upload verification."""
        with patch.object(self.publisher, '_secure_credential_handling') as mock_creds:
            mock_creds.return_value = "hf_test_token"
            
            # Mock repository info
            mock_repo_info = Mock()
            mock_repo_info.lastModified = None
            self.publisher.hf_api.repo_info = Mock(return_value=mock_repo_info)
            
            # Mock file list
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "README.md"]
            self.publisher.hf_api.list_repo_files = Mock(return_value=required_files + ["safety_config.json"])
            
            result = self.publisher.verify_upload("test/model")
            
            assert result["repository_exists"] is True
            assert result["model_accessible"] is True
            assert result["safety_config_present"] is True
            assert len(result["required_files_missing"]) == 0
            
    def test_verify_upload_missing_files(self):
        """Test upload verification with missing files."""
        with patch.object(self.publisher, '_secure_credential_handling') as mock_creds:
            mock_creds.return_value = "hf_test_token"
            
            mock_repo_info = Mock()
            mock_repo_info.lastModified = None
            self.publisher.hf_api.repo_info = Mock(return_value=mock_repo_info)
            
            # Missing some required files
            incomplete_files = ["config.json", "README.md"]
            self.publisher.hf_api.list_repo_files = Mock(return_value=incomplete_files)
            
            result = self.publisher.verify_upload("test/model")
            
            assert result["repository_exists"] is True
            assert result["model_accessible"] is False
            assert len(result["required_files_missing"]) == 2
            
    def test_verify_upload_failure(self):
        """Test upload verification failure."""
        with patch.object(self.publisher, '_secure_credential_handling') as mock_creds:
            mock_creds.side_effect = ValueError("No token")
            
            result = self.publisher.verify_upload("test/model")
            
            assert result["repository_exists"] is False
            assert result["model_accessible"] is False
            assert "error" in result
            
    def test_publish_model_complete_workflow(self):
        """Test complete model publishing workflow."""
        training_details = {
            "base_model": "microsoft/Phi-3-mini-4k-instruct",
            "stages": ["SFT", "PPO"],
            "total_steps": 500
        }
        evaluation_results = {"mt_bench_score": 6.5}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            peft_path = os.path.join(temp_dir, "peft_model")
            os.makedirs(peft_path, exist_ok=True)
            
            # Mock all the methods
            with patch.object(self.publisher, 'merge_peft_adapters') as mock_merge:
                mock_merge.return_value = os.path.join(temp_dir, "merged")
                
                with patch.object(self.publisher, 'apply_safety_guardrails') as mock_safety:
                    mock_safety.return_value = {
                        "safety_config": {"safety_score": 0.9},
                        "safety_evaluation": {"safe_responses": 5, "total_prompts": 5}
                    }
                    
                    with patch.object(self.publisher, 'upload_to_hub') as mock_upload:
                        mock_upload.return_value = "https://huggingface.co/test/model"
                        
                        with patch.object(self.publisher, 'verify_upload') as mock_verify:
                            mock_verify.return_value = {"model_accessible": True}
                            
                            result = self.publisher.publish_model(
                                peft_path, "test/model", training_details, evaluation_results
                            )
                            
                            assert result["success"] is True
                            assert result["model_url"] == "https://huggingface.co/test/model"
                            assert result["model_card_generated"] is True
                            assert result["safety_features_applied"] is True
                            
                            # Verify all steps were called
                            mock_merge.assert_called_once()
                            mock_safety.assert_called_once()
                            mock_upload.assert_called_once()
                            mock_verify.assert_called_once()
                            
    def test_publish_model_failure(self):
        """Test model publishing workflow failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            peft_path = os.path.join(temp_dir, "peft_model")
            
            # Mock merge failure
            with patch.object(self.publisher, 'merge_peft_adapters') as mock_merge:
                mock_merge.side_effect = RuntimeError("Merge failed")
                
                result = self.publisher.publish_model(
                    peft_path, "test/model", {}, {}
                )
                
                assert result["success"] is False
                assert "error" in result
                assert "Merge failed" in result["error"]


class TestModelPublisherEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.publisher = ModelPublisher(self.config)
        
    def test_generate_model_card_empty_inputs(self):
        """Test model card generation with minimal inputs."""
        model_card = self.publisher.generate_model_card("test/model", {}, {})
        
        # Should still generate valid model card
        assert "# test/model" in model_card
        assert "## Model Description" in model_card
        assert "Not evaluated" in model_card  # Default for missing metrics
        
    def test_generate_model_card_missing_config_attributes(self):
        """Test model card generation when config is missing attributes."""
        # Create config without some attributes
        minimal_config = Config()
        if hasattr(minimal_config, 'pipeline_version'):
            delattr(minimal_config, 'pipeline_version')
            
        publisher = ModelPublisher(minimal_config)
        
        model_card = publisher.generate_model_card("test/model", {}, {})
        
        # Should use defaults
        assert "1.0.0" in model_card  # Default pipeline version
        
    def test_safety_filter_special_characters(self):
        """Test safety filter with special characters and unicode."""
        safety_filter = SafetyFilter()
        
        special_texts = [
            "Hello 世界! How are you?",
            "Text with émojis 😊🎉",
            "Special chars: @#$%^&*()",
            "Numbers: 12345 and symbols: <>?",
            ""  # Empty string
        ]
        
        for text in special_texts:
            is_safe, issues = safety_filter.filter_content(text)
            # Should not crash and return valid results
            assert isinstance(is_safe, bool)
            assert isinstance(issues, list)
            
    def test_model_publisher_with_custom_config(self):
        """Test ModelPublisher with custom configuration."""
        custom_config = Config()
        custom_config.model_name = "custom/model"
        custom_config.lora_r = 32
        
        publisher = ModelPublisher(custom_config)
        
        assert publisher.config.model_name == "custom/model"
        assert publisher.config.lora_r == 32