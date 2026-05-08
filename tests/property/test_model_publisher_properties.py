"""
Property-based tests for Model Publisher component.

Tests universal properties that should hold across all valid executions
of the model publishing system.
"""

import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite

from rlhf_phi3.config.config_manager import Config
from rlhf_phi3.publishing.model_publisher import ModelPublisher, SafetyFilter


# Test data generators
@composite
def model_names(draw):
    """Generate valid model names for HuggingFace Hub."""
    username = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'))))
    model_name = draw(st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd', 'Pd'))))
    return f"{username}/{model_name}"


@composite
def training_details(draw):
    """Generate training details dictionaries."""
    return {
        "base_model": draw(st.sampled_from(["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct"])),
        "stages": draw(st.lists(st.sampled_from(["SFT", "Reward", "PPO"]), min_size=1, max_size=3, unique=True)),
        "total_steps": draw(st.integers(min_value=100, max_value=10000)),
        "datasets": draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=5)),
        "training_timestamp": "2024-01-01T00:00:00",
        "config_hash": draw(st.text(min_size=32, max_size=64, alphabet=st.characters(whitelist_categories=('Ll', 'Nd'))))
    }


@composite
def evaluation_results(draw):
    """Generate evaluation results dictionaries."""
    return {
        "mt_bench_score": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False)),
        "helpfulness_score": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False)),
        "harmlessness_score": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False)),
        "honesty_score": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
    }


@composite
def safety_info(draw):
    """Generate safety information dictionaries."""
    return {
        "safety_config": {
            "safety_filter_enabled": True,
            "content_filtering": "Applied during training and inference",
            "safety_score": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            "guardrails": draw(st.lists(st.text(min_size=5, max_size=50), min_size=1, max_size=5))
        },
        "limitations": draw(st.lists(st.text(min_size=10, max_size=100), min_size=1, max_size=5))
    }


class TestModelPublisherProperties:
    """Property-based tests for ModelPublisher."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        
    @given(training_details(), evaluation_results(), safety_info())
    @settings(max_examples=50, deadline=30000)
    def test_property_26_model_card_completeness(self, train_details, eval_results, safety_inf):
        """
        Property 26: Model Card Completeness
        
        For any training details and evaluation results, the Model_Manager SHALL 
        include model cards with training details, evaluation results, and usage instructions.
        
        **Validates: Requirement 10.3**
        """
        publisher = ModelPublisher(self.config)
        model_name = "test-user/test-model"
        
        # Generate model card
        model_card = publisher.generate_model_card(
            model_name, train_details, eval_results, safety_inf
        )
        
        # Verify model card completeness
        assert isinstance(model_card, str)
        assert len(model_card) > 100  # Non-trivial content
        
        # Check required sections are present
        required_sections = [
            "# " + model_name,  # Title
            "## Model Description",
            "## Training Details", 
            "## Evaluation Results",
            "## Usage",
            "## Safety and Limitations",
            "## Training Provenance",
            "## Citation",
            "## License"
        ]
        
        for section in required_sections:
            assert section in model_card, f"Missing required section: {section}"
            
        # Verify training details are included
        base_model = train_details.get("base_model", self.config.model_name)
        assert base_model in model_card
        
        if train_details.get("total_steps"):
            assert str(train_details["total_steps"]) in model_card
            
        # Verify evaluation results are included
        for metric in ["mt_bench_score", "helpfulness_score", "harmlessness_score", "honesty_score"]:
            if metric in eval_results:
                # Either the score or "Not evaluated" should be present
                assert (str(eval_results[metric]) in model_card or "Not evaluated" in model_card)
                
        # Verify usage instructions are present
        assert "```python" in model_card  # Code example
        assert "AutoModelForCausalLM" in model_card
        assert "AutoTokenizer" in model_card
        
        # Verify safety information is included if provided
        if safety_inf and "safety_config" in safety_inf:
            safety_config = safety_inf["safety_config"]
            if "safety_score" in safety_config:
                # Safety score should be formatted to 2 decimal places
                expected_score = f"{safety_config['safety_score']:.2f}"
                assert expected_score in model_card
                
    @given(st.text(min_size=10, max_size=1000))
    @settings(max_examples=30, deadline=10000)
    def test_property_34_safety_guardrail_activation(self, test_text):
        """
        Property 34: Safety Guardrail Activation
        
        For any input to the final model, the Model_Manager SHALL implement 
        content filtering and safety guardrails.
        
        **Validates: Requirement 14.3**
        """
        safety_filter = SafetyFilter()
        
        # Test content filtering
        is_safe, detected_issues = safety_filter.filter_content(test_text)
        
        # Verify return types
        assert isinstance(is_safe, bool)
        assert isinstance(detected_issues, list)
        
        # If issues are detected, safety should be False
        if detected_issues:
            assert not is_safe
            
        # If no issues, safety should be True
        if not detected_issues:
            assert is_safe
            
        # All detected issues should be strings
        for issue in detected_issues:
            assert isinstance(issue, str)
            assert len(issue) > 0
            
    @settings(max_examples=20, deadline=15000)
    def test_property_35_credential_security(self):
        """
        Property 35: Credential Security
        
        For any API keys and credentials, the RLHF_Pipeline SHALL secure them 
        using environment variables and encryption.
        
        **Validates: Requirement 14.4**
        """
        publisher = ModelPublisher(self.config)
        
        # Test credential validation without token
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise exception during initialization
            # but should log warning
            publisher._validate_credentials()
            
        # Test secure credential handling
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN environment variable not set"):
                publisher._secure_credential_handling()
                
        # Test with invalid token format
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "invalid_token"}, clear=True):
            with pytest.raises(ValueError, match="Invalid HuggingFace token format"):
                publisher._secure_credential_handling()
                
        # Test with valid token format
        valid_token = "hf_" + "a" * 30
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": valid_token}, clear=True):
            token = publisher._secure_credential_handling()
            assert token == valid_token
            
        # Verify no credentials are stored in class attributes
        assert not hasattr(publisher, 'token')
        assert not hasattr(publisher, 'api_key')
        
    @given(training_details(), evaluation_results())
    @settings(max_examples=30, deadline=20000)
    def test_property_39_training_provenance_inclusion(self, train_details, eval_results):
        """
        Property 39: Training Provenance Inclusion
        
        For any training scenario, the Model_Manager SHALL include training 
        provenance information in model metadata.
        
        **Validates: Requirement 15.4**
        """
        publisher = ModelPublisher(self.config)
        model_name = "test-user/test-model"
        
        # Generate model card with training provenance
        model_card = publisher.generate_model_card(
            model_name, train_details, eval_results
        )
        
        # Verify training provenance section exists
        assert "## Training Provenance" in model_card
        
        # Check required provenance information
        provenance_items = [
            "Pipeline Version:",
            "Training Environment:",
            "Training Timestamp:",
            "Configuration Hash:",
            "PyTorch:",
            "Transformers:",
            "PEFT:",
            "TRL:"
        ]
        
        for item in provenance_items:
            assert item in model_card, f"Missing provenance item: {item}"
            
        # Verify specific training details are included
        if "config_hash" in train_details:
            assert train_details["config_hash"] in model_card
            
        if "training_timestamp" in train_details:
            assert train_details["training_timestamp"] in model_card
            
        # Verify reproducibility statement
        assert "reproducibility" in model_card.lower()
        assert "configuration" in model_card.lower()
        
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=20, deadline=10000)
    def test_safety_filter_consistency(self, text_input):
        """
        Test that safety filter behaves consistently.
        
        The same input should always produce the same output.
        """
        safety_filter = SafetyFilter()
        
        # Run filter multiple times on same input
        result1 = safety_filter.filter_content(text_input)
        result2 = safety_filter.filter_content(text_input)
        result3 = safety_filter.filter_content(text_input)
        
        # Results should be identical
        assert result1 == result2 == result3
        
        # Verify structure
        is_safe, issues = result1
        assert isinstance(is_safe, bool)
        assert isinstance(issues, list)
        
    @given(model_names(), training_details(), evaluation_results())
    @settings(max_examples=20, deadline=15000)
    def test_model_card_structure_consistency(self, model_name, train_details, eval_results):
        """
        Test that model cards have consistent structure regardless of input.
        """
        publisher = ModelPublisher(self.config)
        
        model_card = publisher.generate_model_card(
            model_name, train_details, eval_results
        )
        
        # Should start with YAML frontmatter
        assert model_card.startswith("---")
        
        # Should have proper markdown structure
        lines = model_card.split('\n')
        yaml_end = None
        for i, line in enumerate(lines):
            if i > 0 and line.strip() == "---":
                yaml_end = i
                break
                
        assert yaml_end is not None, "YAML frontmatter not properly closed"
        
        # Should have main title after YAML
        title_line = lines[yaml_end + 2]  # Skip empty line after ---
        assert title_line.startswith(f"# {model_name}")
        
        # Should have consistent section ordering
        content = '\n'.join(lines[yaml_end + 1:])
        sections = [
            "## Model Description",
            "## Training Details", 
            "## Evaluation Results",
            "## Usage",
            "## Safety and Limitations",
            "## Training Provenance"
        ]
        
        last_pos = 0
        for section in sections:
            pos = content.find(section)
            assert pos > last_pos, f"Section {section} not in correct order"
            last_pos = pos


class TestSafetyFilterProperties:
    """Property-based tests for SafetyFilter."""
    
    @given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10))
    @settings(max_examples=30, deadline=10000)
    def test_batch_filtering_consistency(self, text_list):
        """
        Test that filtering multiple texts produces consistent results.
        """
        safety_filter = SafetyFilter()
        
        # Filter each text individually
        individual_results = [safety_filter.filter_content(text) for text in text_list]
        
        # Filter again to ensure consistency
        repeat_results = [safety_filter.filter_content(text) for text in text_list]
        
        assert individual_results == repeat_results
        
        # Verify all results have correct structure
        for is_safe, issues in individual_results:
            assert isinstance(is_safe, bool)
            assert isinstance(issues, list)
            for issue in issues:
                assert isinstance(issue, str)
                
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=50, deadline=10000)
    def test_safety_filter_robustness(self, text_input):
        """
        Test that safety filter handles all text inputs without crashing.
        """
        safety_filter = SafetyFilter()
        
        # Should not raise any exceptions
        try:
            is_safe, issues = safety_filter.filter_content(text_input)
            
            # Verify output structure
            assert isinstance(is_safe, bool)
            assert isinstance(issues, list)
            
            # If issues found, should not be safe
            if issues:
                assert not is_safe
                
        except Exception as e:
            pytest.fail(f"Safety filter crashed on input: {repr(text_input)}, error: {e}")


class TestModelPublisherIntegration:
    """Integration property tests for ModelPublisher."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        
    @given(training_details(), evaluation_results())
    @settings(max_examples=10, deadline=20000)
    def test_publish_workflow_completeness(self, train_details, eval_results):
        """
        Test that the complete publishing workflow includes all required components.
        """
        publisher = ModelPublisher(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock PEFT model path
            peft_path = os.path.join(temp_dir, "peft_model")
            os.makedirs(peft_path, exist_ok=True)
            
            # Mock the merge_peft_adapters method to avoid actual model loading
            with patch.object(publisher, 'merge_peft_adapters') as mock_merge:
                mock_merge.return_value = os.path.join(temp_dir, "merged_model")
                
                # Mock the apply_safety_guardrails method
                with patch.object(publisher, 'apply_safety_guardrails') as mock_safety:
                    mock_safety.return_value = {
                        "safety_config": {"safety_score": 0.8},
                        "safety_evaluation": {"safe_responses": 4, "total_prompts": 5}
                    }
                    
                    # Mock the upload methods
                    with patch.object(publisher, 'upload_to_hub') as mock_upload:
                        mock_upload.return_value = "https://huggingface.co/test/model"
                        
                        with patch.object(publisher, 'verify_upload') as mock_verify:
                            mock_verify.return_value = {"model_accessible": True}
                            
                            # Test the complete workflow
                            result = publisher.publish_model(
                                peft_path, "test/model", train_details, eval_results
                            )
                            
                            # Verify workflow completion
                            assert result["success"] is True
                            assert "model_url" in result
                            assert "safety_evaluation" in result
                            assert result["model_card_generated"] is True
                            assert result["safety_features_applied"] is True
                            
                            # Verify all steps were called
                            mock_merge.assert_called_once()
                            mock_safety.assert_called_once()
                            mock_upload.assert_called_once()
                            mock_verify.assert_called_once()