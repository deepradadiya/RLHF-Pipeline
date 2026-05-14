"""
Dataset loading and preprocessing for SFT training.

What SFT means: We take a pretrained base model and teach it to follow instructions 
using high-quality examples. Think of it like teaching a student using a textbook 
before giving them practice exams.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)


def load_sft_dataset(
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    split: str = "train_sft",
    max_samples: Optional[int] = None,
    streaming: bool = False
) -> Dataset:
    """
    Load the SFT training dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to use (train_sft for instruction following)
        max_samples: Maximum number of samples to load (10,000 for Colab to avoid memory issues)
        streaming: Whether to use streaming mode for large datasets
    
    Returns:
        Loaded dataset
    """
    logger.info(f"Loading dataset {dataset_name}, split: {split}")
    
    try:
        # Load the dataset
        dataset = load_dataset(
            dataset_name, 
            split=split,
            streaming=streaming
        )
        
        # For Colab: limit to first 10,000 examples to avoid memory issues
        # This is a practical limitation due to T4 GPU memory constraints
        if max_samples is not None:
            if streaming:
                dataset = dataset.take(max_samples)
            else:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Limited dataset to {max_samples} samples for Colab compatibility")
        
        logger.info(f"Successfully loaded dataset with {len(dataset) if not streaming else 'streaming'} examples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
        raise


def format_chat(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """
    Convert each example into Phi-3's chat template format.
    
    Phi-3 uses a specific chat template format:
    <|user|>\n{user_message}<|end|>\n<|assistant|>\n{assistant_message}<|end|>\n
    
    Args:
        example: Single dataset example with 'messages' field
        tokenizer: Phi-3 tokenizer with chat template
    
    Returns:
        Formatted example with 'text' field containing the chat template
    """
    try:
        # Extract messages from the example
        messages = example.get('messages', [])
        
        if not messages:
            raise ValueError("No messages found in example")
        
        # Apply Phi-3 chat template
        # The tokenizer.apply_chat_template handles the proper formatting
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Return string, not tokens
            add_generation_prompt=False  # Include assistant response
        )
        
        return {
            **example,
            'text': formatted_text,
            'length': len(formatted_text)
        }
        
    except Exception as e:
        logger.warning(f"Failed to format chat example: {str(e)}")
        # Return empty text for invalid examples
        return {
            **example,
            'text': '',
            'length': 0
        }


def validate_dataset(dataset: Dataset, max_length: int = 512) -> Tuple[Dataset, Dict]:
    """
    Validate dataset and filter out problematic examples.
    
    Checks for:
    - Empty strings
    - Too-long sequences (>max_length to avoid OOM)
    - Missing required fields
    
    Args:
        dataset: Dataset to validate
        max_length: Maximum sequence length (512 for Colab T4 GPU)
    
    Returns:
        Tuple of (filtered_dataset, validation_stats)
    """
    logger.info("Starting dataset validation...")
    
    original_size = len(dataset)
    validation_stats = {
        'original_size': original_size,
        'empty_examples': 0,
        'too_long_examples': 0,
        'missing_fields': 0,
        'valid_examples': 0
    }
    
    def is_valid_example(example):
        # Check for empty text
        if not example.get('text') or len(example['text'].strip()) == 0:
            validation_stats['empty_examples'] += 1
            return False
        
        # Check for missing messages field
        if 'messages' not in example:
            validation_stats['missing_fields'] += 1
            return False
        
        # Check sequence length (character count as proxy)
        if example.get('length', 0) > max_length * 4:  # Rough estimate: 4 chars per token
            validation_stats['too_long_examples'] += 1
            return False
        
        validation_stats['valid_examples'] += 1
        return True
    
    # Filter the dataset
    filtered_dataset = dataset.filter(is_valid_example)
    
    logger.info(f"Dataset validation complete:")
    logger.info(f"  Original examples: {validation_stats['original_size']}")
    logger.info(f"  Valid examples: {validation_stats['valid_examples']}")
    logger.info(f"  Filtered out - Empty: {validation_stats['empty_examples']}")
    logger.info(f"  Filtered out - Too long: {validation_stats['too_long_examples']}")
    logger.info(f"  Filtered out - Missing fields: {validation_stats['missing_fields']}")
    
    return filtered_dataset, validation_stats


def print_dataset_statistics(dataset: Dataset, tokenizer: AutoTokenizer) -> None:
    """
    Print comprehensive dataset statistics.
    
    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer for length calculations
    """
    logger.info("Computing dataset statistics...")
    
    # Sample a subset for statistics (to avoid memory issues)
    sample_size = min(1000, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    sample_dataset = dataset.select(sample_indices)
    
    # Calculate text lengths
    text_lengths = []
    token_lengths = []
    
    for example in sample_dataset:
        text = example.get('text', '')
        text_lengths.append(len(text))
        
        # Tokenize to get actual token count
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_lengths.append(len(tokens))
    
    # Calculate statistics
    stats = {
        'num_examples': len(dataset),
        'sample_size': sample_size,
        'avg_text_length': np.mean(text_lengths),
        'max_text_length': np.max(text_lengths),
        'min_text_length': np.min(text_lengths),
        'avg_token_length': np.mean(token_lengths),
        'max_token_length': np.max(token_lengths),
        'min_token_length': np.min(token_lengths),
    }
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total examples: {stats['num_examples']:,}")
    print(f"Sample size for stats: {stats['sample_size']:,}")
    print(f"\nText Length (characters):")
    print(f"  Average: {stats['avg_text_length']:.1f}")
    print(f"  Maximum: {stats['max_text_length']:,}")
    print(f"  Minimum: {stats['min_text_length']:,}")
    print(f"\nToken Length:")
    print(f"  Average: {stats['avg_token_length']:.1f}")
    print(f"  Maximum: {stats['max_token_length']:,}")
    print(f"  Minimum: {stats['min_token_length']:,}")
    print("="*50)


def prepare_sft_dataset(
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    max_samples: int = 10000,  # Colab limitation
    max_length: int = 512,     # Colab T4 GPU limitation
    streaming: bool = False
) -> Tuple[Dataset, AutoTokenizer, Dict]:
    """
    Complete pipeline to prepare SFT dataset.
    
    Args:
        dataset_name: HuggingFace dataset to load
        model_name: Model name for tokenizer
        max_samples: Maximum samples (10,000 for Colab memory constraints)
        max_length: Maximum sequence length (512 to avoid OOM on T4)
        streaming: Whether to use streaming mode
    
    Returns:
        Tuple of (prepared_dataset, tokenizer, statistics)
    """
    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_sft_dataset(
        dataset_name=dataset_name,
        max_samples=max_samples,
        streaming=streaming
    )
    
    # Format examples with chat template
    logger.info("Formatting examples with Phi-3 chat template...")
    formatted_dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        desc="Formatting chat templates"
    )
    
    # Validate and filter dataset
    validated_dataset, validation_stats = validate_dataset(
        formatted_dataset, 
        max_length=max_length
    )
    
    # Print statistics
    print_dataset_statistics(validated_dataset, tokenizer)
    
    return validated_dataset, tokenizer, validation_stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Preparing SFT dataset for Phi-3 training...")
    dataset, tokenizer, stats = prepare_sft_dataset()
    
    print(f"\nDataset ready for training!")
    print(f"Examples: {len(dataset)}")
    print(f"Validation stats: {stats}")