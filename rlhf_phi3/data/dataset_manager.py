"""
Dataset Manager for RLHF Phi-3 Pipeline

This module handles dataset loading, preprocessing, and formatting for all training stages.
It supports both SFT (instruction-following) datasets and preference datasets for reward 
training, with proper Phi-3 chat template formatting and efficient caching mechanisms.

Requirements satisfied:
- 7.1: Load datasets from HuggingFace Hub with automatic caching
- 7.2: Preprocess data according to Phi-3's chat template format
- 7.3: Validate dataset integrity and format compliance before training
- 7.4: Handle tokenization with proper padding and truncation strategies
- 7.5: Support both instruction-following datasets for SFT and preference datasets for reward training
- 5.4: Use streaming datasets to avoid loading entire datasets into memory
"""

import logging
import os
import re
import hashlib
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
import warnings
from collections import defaultdict

import torch
from datasets import Dataset, load_dataset, DatasetDict, IterableDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from ..config.config_manager import Config, DatasetConfig

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Dataset Manager for handling dataset loading, preprocessing, and formatting.
    
    This class provides comprehensive dataset management for the RLHF pipeline,
    supporting both SFT and preference datasets with Phi-3 chat template formatting.
    
    Requirements satisfied:
    - 7.1: Load datasets from HuggingFace Hub with automatic caching
    - 7.2: Preprocess data according to Phi-3's chat template format
    - 7.3: Validate dataset integrity and format compliance before training
    - 7.4: Handle tokenization with proper padding and truncation strategies
    - 7.5: Support both instruction-following datasets for SFT and preference datasets for reward training
    - 5.4: Use streaming datasets to avoid loading entire datasets into memory
    """
    
    def __init__(self, config: Config, tokenizer: Optional[PreTrainedTokenizer] = None):
        """
        Initialize the Dataset Manager.
        
        Args:
            config: Configuration object containing dataset and model settings
            tokenizer: Optional pre-initialized tokenizer. If None, will load from config
        """
        self.config = config
        self.tokenizer = tokenizer
        self.cache_dir = Path(config.paths.cache_dir) / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer if not provided
        if self.tokenizer is None:
            self._initialize_tokenizer()
        
        # Cache for loaded datasets
        self._dataset_cache: Dict[str, Dataset] = {}
        
        # Content filtering patterns for safety
        self._harmful_patterns = self._load_harmful_patterns()
        
        # Dataset validation statistics
        self._validation_stats = defaultdict(int)
        
        logger.info(f"DatasetManager initialized with cache dir: {self.cache_dir}")
    
    def _load_harmful_patterns(self) -> List[str]:
        """
        Load patterns for detecting harmful content.
        
        Returns:
            List of regex patterns for harmful content detection
            
        Requirement 7.3: Validate dataset integrity and format compliance before training
        """
        # Comprehensive harmful content patterns for production use
        patterns = [
            # Violence and harm
            r'\b(?:hate|violence|harm|kill|murder|suicide|death|torture|abuse)\b',
            r'\b(?:attack|assault|fight|beat|punch|kick|stab|shoot)\b',
            r'\b(?:destroy|damage|hurt|injure|wound|pain)\b',
            
            # Discrimination and bias
            r'\b(?:racist|sexist|homophobic|transphobic|xenophobic)\b',
            r'\b(?:nazi|hitler|genocide|holocaust|supremacist)\b',
            r'\b(?:slur|derogatory|offensive|discriminat)\w*\b',
            
            # Weapons and dangerous items
            r'\b(?:bomb|weapon|explosive|terrorist|gun|rifle|pistol)\b',
            r'\b(?:grenade|ammunition|firearm|knife|blade)\b',
            r'\b(?:poison|toxic|chemical|biological)\b',
            
            # Illegal substances and activities
            r'\b(?:drug|cocaine|heroin|meth|marijuana|cannabis)\b',
            r'\b(?:illegal|criminal|fraud|theft|steal|rob)\b',
            r'\b(?:hack|crack|piracy|counterfeit)\b',
            
            # Sexual content (basic detection)
            r'\b(?:sexual|explicit|pornographic|nude|naked)\b',
            r'\b(?:sex|porn|adult|erotic|intimate)\b',
            
            # Self-harm and mental health crisis
            r'\b(?:self.harm|cut.myself|end.my.life|want.to.die)\b',
            r'\b(?:depressed|hopeless|worthless|suicidal)\b',
            
            # Misinformation indicators
            r'\b(?:conspiracy|hoax|fake.news|misinformation)\b',
            r'\b(?:covid.hoax|vaccine.dangerous|flat.earth)\b',
            
            # Privacy violations
            r'\b(?:personal.information|social.security|credit.card)\b',
            r'\b(?:password|private.key|confidential)\b',
            
            # Spam and manipulation
            r'\b(?:click.here|buy.now|limited.time|act.fast)\b',
            r'\b(?:guaranteed|miracle|secret|amazing.results)\b'
        ]
        
        # Compile patterns with case-insensitive matching
        compiled_patterns = []
        for pattern in patterns:
            try:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE | re.DOTALL))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue
        
        logger.info(f"Loaded {len(compiled_patterns)} harmful content detection patterns")
        return compiled_patterns
    
    def _initialize_tokenizer(self) -> None:
        """Initialize the tokenizer from the model configuration."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.name,
                cache_dir=str(self.cache_dir / "tokenizers"),
                trust_remote_code=True
            )
            
            # Ensure tokenizer has required special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            logger.info(f"Tokenizer initialized: {self.config.model.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise
    
    def load_sft_dataset(self, dataset_name: Optional[str] = None, streaming: bool = False) -> Union[Dataset, IterableDataset]:
        """
        Load SFT (Supervised Fine-Tuning) dataset from HuggingFace Hub.
        
        Args:
            dataset_name: Optional dataset name. If None, uses config default
            streaming: Whether to use streaming mode for memory efficiency
            
        Returns:
            Dataset or IterableDataset: Loaded and cached SFT dataset
            
        Requirement 7.1: Load datasets from HuggingFace Hub with automatic caching
        Requirement 5.4: Use streaming datasets to avoid loading entire datasets into memory
        """
        if dataset_name is None:
            dataset_config = self.config.datasets.sft
        else:
            dataset_config = DatasetConfig(
                name=dataset_name,
                split=self.config.datasets.sft.split,
                max_samples=self.config.datasets.sft.max_samples
            )
        
        cache_key = f"sft_{dataset_config.name}_{dataset_config.split}_{dataset_config.max_samples}_{streaming}"
        
        if not streaming and cache_key in self._dataset_cache:
            logger.info(f"Using cached SFT dataset: {cache_key}")
            return self._dataset_cache[cache_key]
        
        try:
            logger.info(f"Loading SFT dataset: {dataset_config.name} (streaming={streaming})")
            
            # Load dataset with caching
            dataset = load_dataset(
                dataset_config.name,
                split=dataset_config.split,
                cache_dir=str(self.cache_dir),
                streaming=streaming
            )
            
            # Handle streaming vs non-streaming datasets differently
            if streaming:
                # For streaming datasets, we can't easily limit samples upfront
                # This will be handled in the preprocessing step
                logger.info(f"SFT streaming dataset loaded: {dataset_config.name}")
                return dataset
            else:
                # Limit samples if specified for non-streaming datasets
                if dataset_config.max_samples > 0 and len(dataset) > dataset_config.max_samples:
                    dataset = dataset.select(range(dataset_config.max_samples))
                    logger.info(f"Limited dataset to {dataset_config.max_samples} samples")
                
                # Cache the dataset
                self._dataset_cache[cache_key] = dataset
                
                logger.info(f"SFT dataset loaded: {len(dataset)} samples")
                return dataset
            
        except Exception as e:
            logger.error(f"Failed to load SFT dataset {dataset_config.name}: {e}")
            raise
    
    def load_preference_dataset(self, dataset_name: Optional[str] = None, streaming: bool = False) -> Union[Dataset, IterableDataset]:
        """
        Load preference dataset for reward model training.
        
        Args:
            dataset_name: Optional dataset name. If None, uses config default
            streaming: Whether to use streaming mode for memory efficiency
            
        Returns:
            Dataset or IterableDataset: Loaded and cached preference dataset
            
        Requirement 7.1: Load datasets from HuggingFace Hub with automatic caching
        Requirement 5.4: Use streaming datasets to avoid loading entire datasets into memory
        """
        if dataset_name is None:
            dataset_config = self.config.datasets.preference
        else:
            dataset_config = DatasetConfig(
                name=dataset_name,
                split=self.config.datasets.preference.split,
                max_samples=self.config.datasets.preference.max_samples
            )
        
        cache_key = f"pref_{dataset_config.name}_{dataset_config.split}_{dataset_config.max_samples}_{streaming}"
        
        if not streaming and cache_key in self._dataset_cache:
            logger.info(f"Using cached preference dataset: {cache_key}")
            return self._dataset_cache[cache_key]
        
        try:
            logger.info(f"Loading preference dataset: {dataset_config.name} (streaming={streaming})")
            
            # Load dataset with caching
            dataset = load_dataset(
                dataset_config.name,
                split=dataset_config.split,
                cache_dir=str(self.cache_dir),
                streaming=streaming
            )
            
            # Handle streaming vs non-streaming datasets differently
            if streaming:
                logger.info(f"Preference streaming dataset loaded: {dataset_config.name}")
                return dataset
            else:
                # Limit samples if specified for non-streaming datasets
                if dataset_config.max_samples > 0 and len(dataset) > dataset_config.max_samples:
                    dataset = dataset.select(range(dataset_config.max_samples))
                    logger.info(f"Limited dataset to {dataset_config.max_samples} samples")
                
                # Cache the dataset
                self._dataset_cache[cache_key] = dataset
                
                logger.info(f"Preference dataset loaded: {len(dataset)} samples")
                return dataset
            
        except Exception as e:
            logger.error(f"Failed to load preference dataset {dataset_config.name}: {e}")
            raise
    
    def format_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages according to Phi-3's chat template format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: Formatted chat string according to Phi-3 template
            
        Requirement 7.2: Preprocess data according to Phi-3's chat template format
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Validate message format
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Message {i} must have 'role' and 'content' keys")
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Message {i} role must be 'system', 'user', or 'assistant'")
        
        try:
            # Use the tokenizer's chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return formatted
            else:
                # Fallback to manual Phi-3 format
                return self._manual_phi3_format(messages)
                
        except Exception as e:
            logger.warning(f"Chat template formatting failed, using manual format: {e}")
            return self._manual_phi3_format(messages)
    
    def _manual_phi3_format(self, messages: List[Dict[str, str]]) -> str:
        """
        Manual implementation of Phi-3 chat format as fallback.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Manually formatted chat string
        """
        formatted_parts = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content'].strip()
            
            if role == 'system':
                formatted_parts.append(f"<|system|>\n{content}<|end|>")
            elif role == 'user':
                formatted_parts.append(f"<|user|>\n{content}<|end|>")
            elif role == 'assistant':
                formatted_parts.append(f"<|assistant|>\n{content}<|end|>")
        
        return "\n".join(formatted_parts)
    
    def preprocess_sft_data(self, dataset: Union[Dataset, IterableDataset], validate: bool = True) -> Union[Dataset, IterableDataset]:
        """
        Preprocess SFT dataset with chat template formatting and tokenization.
        
        Args:
            dataset: Raw SFT dataset
            validate: Whether to validate examples during preprocessing
            
        Returns:
            Dataset: Preprocessed dataset with tokenized inputs
            
        Requirement 7.2: Preprocess data according to Phi-3's chat template format
        Requirement 7.4: Handle tokenization with proper padding and truncation strategies
        Requirement 7.5: Support both instruction-following datasets for SFT and preference datasets for reward training
        """
        logger.info("Preprocessing SFT dataset...")
        
        # Reset validation statistics
        self._validation_stats.clear()
        
        # Track preprocessing performance
        import time
        start_time = time.time()
        processed_count = 0
        
        def process_sft_example(example):
            """Process a single SFT example with enhanced error handling."""
            nonlocal processed_count
            processed_count += 1
            
            # Log progress for large datasets
            if processed_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {processed_count} examples ({rate:.1f} examples/sec)")
            
            try:
                # Extract messages from the example
                # Handle different dataset formats
                messages = self._extract_messages_from_sft_example(example)
                
                if not messages:
                    self._validation_stats['no_messages'] += 1
                    return None
                
                # Validate messages if requested
                if validate:
                    is_valid, warnings = self._validate_messages_structure(messages)
                    if not is_valid:
                        self._validation_stats['invalid_structure'] += 1
                        return None
                    
                    if warnings:
                        self._validation_stats['warnings'] += len(warnings)
                
                # Format using chat template with retry logic
                try:
                    formatted_text = self.format_chat_template(messages)
                except Exception as e:
                    logger.warning(f"Chat template formatting failed, trying fallback: {e}")
                    # Fallback to simple concatenation
                    formatted_text = self._manual_phi3_format(messages)
                
                # Tokenize with proper strategies and error handling
                try:
                    tokenized = self._tokenize_text(
                        formatted_text,
                        max_length=self.config.model.max_length,
                        padding=False,  # We'll pad in the dataloader
                        truncation=True
                    )
                except Exception as e:
                    logger.warning(f"Tokenization failed for example, skipping: {e}")
                    self._validation_stats['tokenization_errors'] += 1
                    return None
                
                # Create labels for causal language modeling
                labels = self._create_sft_labels(formatted_text, tokenized['input_ids'])
                
                # Validate tokenized output
                if validate and not self._validate_tokenized_example(tokenized, labels):
                    self._validation_stats['invalid_tokenization'] += 1
                    return None
                
                self._validation_stats['successful'] += 1
                
                return {
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'labels': labels,
                    'text': formatted_text,
                    'original_length': len(formatted_text),
                    'token_count': len(tokenized['input_ids'])
                }
                
            except Exception as e:
                logger.warning(f"Failed to process SFT example: {e}")
                self._validation_stats['processing_errors'] += 1
                return None
        
        # Apply preprocessing with enhanced streaming support
        if isinstance(dataset, IterableDataset):
            # For streaming datasets
            processed_dataset = dataset.map(process_sft_example)
            # Filter out None examples
            processed_dataset = processed_dataset.filter(lambda x: x is not None)
        else:
            # For regular datasets
            processed_dataset = dataset.map(
                process_sft_example,
                remove_columns=dataset.column_names,
                desc="Processing SFT examples",
                num_proc=1  # Single process to avoid tokenizer issues
            )
            # Filter out None examples
            processed_dataset = processed_dataset.filter(lambda x: x is not None)
            
            logger.info(f"SFT preprocessing complete: {len(processed_dataset)} examples")
        
        # Log validation statistics with performance metrics
        total_time = time.time() - start_time
        self._log_validation_stats("SFT preprocessing")
        logger.info(f"SFT preprocessing performance: {processed_count/total_time:.1f} examples/sec")
        
        return processed_dataset
    
    def preprocess_preference_data(self, dataset: Union[Dataset, IterableDataset], validate: bool = True) -> Union[Dataset, IterableDataset]:
        """
        Preprocess preference dataset for reward model training.
        
        Args:
            dataset: Raw preference dataset
            validate: Whether to validate examples during preprocessing
            
        Returns:
            Dataset: Preprocessed dataset with tokenized chosen/rejected pairs
            
        Requirement 7.2: Preprocess data according to Phi-3's chat template format
        Requirement 7.4: Handle tokenization with proper padding and truncation strategies
        Requirement 7.5: Support both instruction-following datasets for SFT and preference datasets for reward training
        """
        logger.info("Preprocessing preference dataset...")
        
        # Reset validation statistics
        self._validation_stats.clear()
        
        # Track preprocessing performance
        import time
        start_time = time.time()
        processed_count = 0
        
        def process_preference_example(example):
            """Process a single preference example with enhanced error handling."""
            nonlocal processed_count
            processed_count += 1
            
            # Log progress for large datasets
            if processed_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {processed_count} examples ({rate:.1f} examples/sec)")
            
            try:
                # Extract prompt and responses
                prompt, chosen, rejected = self._extract_preference_data(example)
                
                if not all([prompt, chosen, rejected]):
                    self._validation_stats['missing_fields'] += 1
                    return None
                
                # Validate content if requested
                if validate:
                    if chosen == rejected:
                        self._validation_stats['identical_responses'] += 1
                        return None
                    
                    # Check for harmful content
                    for field_name, content in [('prompt', prompt), ('chosen', chosen), ('rejected', rejected)]:
                        if self._contains_harmful_content(content):
                            self._validation_stats['harmful_content'] += 1
                            return None
                
                # Create message format for chosen response
                chosen_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen}
                ]
                
                # Create message format for rejected response
                rejected_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected}
                ]
                
                # Format using chat template with retry logic
                try:
                    chosen_text = self.format_chat_template(chosen_messages)
                    rejected_text = self.format_chat_template(rejected_messages)
                except Exception as e:
                    logger.warning(f"Chat template formatting failed, trying fallback: {e}")
                    # Fallback to manual formatting
                    chosen_text = self._manual_phi3_format(chosen_messages)
                    rejected_text = self._manual_phi3_format(rejected_messages)
                
                # Tokenize both responses with error handling
                try:
                    chosen_tokenized = self._tokenize_text(
                        chosen_text,
                        max_length=self.config.model.max_length,
                        padding=False,
                        truncation=True
                    )
                    
                    rejected_tokenized = self._tokenize_text(
                        rejected_text,
                        max_length=self.config.model.max_length,
                        padding=False,
                        truncation=True
                    )
                except Exception as e:
                    logger.warning(f"Tokenization failed for preference example, skipping: {e}")
                    self._validation_stats['tokenization_errors'] += 1
                    return None
                
                # Validate tokenized outputs
                if validate:
                    if not (self._validate_tokenized_example(chosen_tokenized) and 
                           self._validate_tokenized_example(rejected_tokenized)):
                        self._validation_stats['invalid_tokenization'] += 1
                        return None
                
                self._validation_stats['successful'] += 1
                
                return {
                    'chosen_input_ids': chosen_tokenized['input_ids'],
                    'chosen_attention_mask': chosen_tokenized['attention_mask'],
                    'rejected_input_ids': rejected_tokenized['input_ids'],
                    'rejected_attention_mask': rejected_tokenized['attention_mask'],
                    'chosen_text': chosen_text,
                    'rejected_text': rejected_text,
                    'prompt': prompt,
                    'chosen_length': len(chosen_text),
                    'rejected_length': len(rejected_text),
                    'chosen_token_count': len(chosen_tokenized['input_ids']),
                    'rejected_token_count': len(rejected_tokenized['input_ids'])
                }
                
            except Exception as e:
                logger.warning(f"Failed to process preference example: {e}")
                self._validation_stats['processing_errors'] += 1
                return None
        
        # Apply preprocessing with enhanced streaming support
        if isinstance(dataset, IterableDataset):
            # For streaming datasets
            processed_dataset = dataset.map(process_preference_example)
            # Filter out None examples
            processed_dataset = processed_dataset.filter(lambda x: x is not None)
        else:
            # For regular datasets
            processed_dataset = dataset.map(
                process_preference_example,
                remove_columns=dataset.column_names,
                desc="Processing preference examples",
                num_proc=1  # Single process to avoid tokenizer issues
            )
            # Filter out None examples
            processed_dataset = processed_dataset.filter(lambda x: x is not None)
            
            logger.info(f"Preference preprocessing complete: {len(processed_dataset)} examples")
        
        # Log validation statistics with performance metrics
        total_time = time.time() - start_time
        self._log_validation_stats("Preference preprocessing")
        logger.info(f"Preference preprocessing performance: {processed_count/total_time:.1f} examples/sec")
        
        return processed_dataset
    
    def _extract_messages_from_sft_example(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract messages from various SFT dataset formats."""
        messages = []
        
        if 'messages' in example:
            messages = example['messages']
        elif 'conversations' in example:
            messages = example['conversations']
        elif 'conversation' in example:
            messages = example['conversation']
        else:
            # Try to construct from instruction/response format
            if 'instruction' in example:
                messages.append({"role": "user", "content": example['instruction']})
            elif 'input' in example:
                messages.append({"role": "user", "content": example['input']})
            elif 'question' in example:
                messages.append({"role": "user", "content": example['question']})
            
            # Add system message if present
            if 'system' in example and example['system']:
                messages.insert(0, {"role": "system", "content": example['system']})
            
            # Add assistant response
            if 'response' in example:
                messages.append({"role": "assistant", "content": example['response']})
            elif 'output' in example:
                messages.append({"role": "assistant", "content": example['output']})
            elif 'answer' in example:
                messages.append({"role": "assistant", "content": example['answer']})
        
        return messages
    
    def _extract_preference_data(self, example: Dict[str, Any]) -> Tuple[str, str, str]:
        """Extract prompt, chosen, and rejected responses from preference example."""
        # Handle different field names
        prompt = (example.get('prompt') or 
                 example.get('question') or 
                 example.get('input') or '')
        
        chosen = (example.get('chosen') or 
                 example.get('response_chosen') or 
                 example.get('preferred') or '')
        
        rejected = (example.get('rejected') or 
                   example.get('response_rejected') or 
                   example.get('not_preferred') or '')
        
        return str(prompt).strip(), str(chosen).strip(), str(rejected).strip()
    
    def _validate_messages_structure(self, messages: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
        """Validate the structure of messages."""
        warnings = []
        
        if not messages:
            return False, ["Empty messages list"]
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, [f"Message {i} is not a dictionary"]
            
            if 'role' not in msg or 'content' not in msg:
                return False, [f"Message {i} missing 'role' or 'content'"]
            
            if msg['role'] not in ['system', 'user', 'assistant']:
                return False, [f"Message {i} has invalid role: {msg['role']}"]
            
            content = str(msg['content']).strip()
            if not content:
                warnings.append(f"Message {i} has empty content")
            elif len(content) > 10000:
                warnings.append(f"Message {i} has very long content ({len(content)} chars)")
        
        # Check for at least one assistant response
        if not any(msg['role'] == 'assistant' for msg in messages):
            warnings.append("No assistant response found")
        
        return True, warnings
    
    def _validate_tokenized_example(self, tokenized: Dict[str, List[int]], labels: Optional[List[int]] = None) -> bool:
        """Validate tokenized example."""
        if not tokenized or 'input_ids' not in tokenized or 'attention_mask' not in tokenized:
            return False
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        if not input_ids or not attention_mask:
            return False
        
        if len(input_ids) != len(attention_mask):
            return False
        
        if labels is not None and len(labels) != len(input_ids):
            return False
        
        # Check for reasonable sequence length
        if len(input_ids) < 3 or len(input_ids) > self.config.model.max_length:
            return False
        
        return True
    
    def _log_validation_stats(self, stage: str) -> None:
        """Log validation statistics."""
        if not self._validation_stats:
            return
        
        total_processed = sum(self._validation_stats.values())
        successful = self._validation_stats.get('successful', 0)
        
        logger.info(f"{stage} validation statistics:")
        logger.info(f"  Total processed: {total_processed}")
        logger.info(f"  Successful: {successful}")
        
        for key, value in self._validation_stats.items():
            if key != 'successful' and value > 0:
                logger.info(f"  {key}: {value}")
        
        if total_processed > 0:
            success_rate = (successful / total_processed) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")
    
    def _tokenize_text(
        self,
        text: str,
        max_length: int,
        padding: Union[bool, str] = False,
        truncation: bool = True
    ) -> Dict[str, List[int]]:
        """
        Tokenize text with proper padding and truncation strategies.
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            
        Returns:
            Dict containing input_ids and attention_mask
            
        Requirement 7.4: Handle tokenization with proper padding and truncation strategies
        """
        try:
            tokenized = self.tokenizer(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=None,  # Return lists, not tensors
                add_special_tokens=True
            )
            
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask']
            }
            
        except Exception as e:
            logger.error(f"Tokenization failed for text: {text[:100]}... Error: {e}")
            raise
    
    def _create_sft_labels(self, formatted_text: str, input_ids: List[int]) -> List[int]:
        """
        Create labels for SFT training by masking non-assistant tokens.
        
        This implementation masks user and system messages, keeping only assistant
        responses for loss calculation, which is the standard approach for instruction
        fine-tuning.
        
        Args:
            formatted_text: The formatted chat text
            input_ids: Tokenized input IDs
            
        Returns:
            List[int]: Labels with -100 for masked tokens, original token IDs for assistant responses
        """
        labels = [-100] * len(input_ids)  # Start with all tokens masked
        
        try:
            # Enhanced label creation with more precise token alignment
            assistant_start_token = '<|assistant|>'
            assistant_end_token = '<|end|>'
            
            if assistant_start_token in formatted_text:
                # Use more precise token-level alignment
                assistant_start_ids = self.tokenizer.encode(assistant_start_token, add_special_tokens=False)
                assistant_end_ids = self.tokenizer.encode(assistant_end_token, add_special_tokens=False)
                
                # Find assistant response sections using token-level matching
                i = 0
                while i < len(input_ids) - len(assistant_start_ids):
                    # Check if we found the start of an assistant response
                    if input_ids[i:i+len(assistant_start_ids)] == assistant_start_ids:
                        # Skip the assistant start tokens
                        content_start = i + len(assistant_start_ids)
                        
                        # Find the end of the assistant response
                        content_end = len(input_ids)  # Default to end of sequence
                        for j in range(content_start, len(input_ids) - len(assistant_end_ids) + 1):
                            if input_ids[j:j+len(assistant_end_ids)] == assistant_end_ids:
                                content_end = j
                                break
                        
                        # Unmask the assistant response content (excluding end token)
                        for k in range(content_start, content_end):
                            if k < len(labels):
                                labels[k] = input_ids[k]
                        
                        # Move past this assistant response
                        i = content_end + len(assistant_end_ids)
                    else:
                        i += 1
                
                # Count unmasked tokens for logging
                unmasked_count = sum(1 for label in labels if label != -100)
                logger.debug(f"Created SFT labels: {unmasked_count}/{len(labels)} tokens unmasked")
                
            else:
                # Enhanced fallback: try to detect response patterns
                # Look for common response indicators
                response_indicators = ["Response:", "Answer:", "Output:", "Assistant:"]
                found_indicator = False
                
                for indicator in response_indicators:
                    indicator_ids = self.tokenizer.encode(indicator, add_special_tokens=False)
                    for i in range(len(input_ids) - len(indicator_ids) + 1):
                        if input_ids[i:i+len(indicator_ids)] == indicator_ids:
                            # Unmask from this point onwards
                            for j in range(i + len(indicator_ids), len(input_ids)):
                                labels[j] = input_ids[j]
                            found_indicator = True
                            break
                    if found_indicator:
                        break
                
                if not found_indicator:
                    # Final fallback: unmask the last 50% of tokens
                    split_point = len(input_ids) // 2
                    for i in range(split_point, len(input_ids)):
                        labels[i] = input_ids[i]
                
                logger.warning("No assistant tokens found, using enhanced fallback masking strategy")
        
        except Exception as e:
            logger.warning(f"Error creating SFT labels, using fallback: {e}")
            # Fallback: unmask all tokens
            labels = input_ids.copy()
        
        return labels
    
    def create_dataloaders(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create DataLoader with proper collation for the dataset.
        
        Args:
            dataset: Preprocessed dataset
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader: Configured DataLoader
        """
        def collate_fn(batch):
            """Custom collate function for proper padding."""
            # Determine if this is SFT or preference data
            if 'chosen_input_ids' in batch[0]:
                return self._collate_preference_batch(batch)
            else:
                return self._collate_sft_batch(batch)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    def _collate_sft_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for SFT batches with proper padding.
        
        Args:
            batch: List of SFT examples
            
        Returns:
            Dict containing batched and padded tensors
        """
        # Extract sequences
        input_ids = [example['input_ids'] for example in batch]
        attention_masks = [example['attention_mask'] for example in batch]
        labels = [example['labels'] for example in batch]
        
        # Pad sequences
        max_length = max(len(seq) for seq in input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for i in range(len(batch)):
            seq_len = len(input_ids[i])
            pad_length = max_length - seq_len
            
            # Pad input_ids
            padded_input_ids.append(
                input_ids[i] + [self.tokenizer.pad_token_id] * pad_length
            )
            
            # Pad attention_mask
            padded_attention_masks.append(
                attention_masks[i] + [0] * pad_length
            )
            
            # Pad labels (use -100 for padding tokens)
            padded_labels.append(
                labels[i] + [-100] * pad_length
            )
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }
    
    def _collate_preference_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for preference batches with proper padding.
        
        Args:
            batch: List of preference examples
            
        Returns:
            Dict containing batched and padded tensors for chosen/rejected pairs
        """
        # Extract chosen sequences
        chosen_input_ids = [example['chosen_input_ids'] for example in batch]
        chosen_attention_masks = [example['chosen_attention_mask'] for example in batch]
        
        # Extract rejected sequences
        rejected_input_ids = [example['rejected_input_ids'] for example in batch]
        rejected_attention_masks = [example['rejected_attention_mask'] for example in batch]
        
        # Pad chosen sequences
        chosen_max_length = max(len(seq) for seq in chosen_input_ids)
        padded_chosen_input_ids = []
        padded_chosen_attention_masks = []
        
        for i in range(len(batch)):
            seq_len = len(chosen_input_ids[i])
            pad_length = chosen_max_length - seq_len
            
            padded_chosen_input_ids.append(
                chosen_input_ids[i] + [self.tokenizer.pad_token_id] * pad_length
            )
            padded_chosen_attention_masks.append(
                chosen_attention_masks[i] + [0] * pad_length
            )
        
        # Pad rejected sequences
        rejected_max_length = max(len(seq) for seq in rejected_input_ids)
        padded_rejected_input_ids = []
        padded_rejected_attention_masks = []
        
        for i in range(len(batch)):
            seq_len = len(rejected_input_ids[i])
            pad_length = rejected_max_length - seq_len
            
            padded_rejected_input_ids.append(
                rejected_input_ids[i] + [self.tokenizer.pad_token_id] * pad_length
            )
            padded_rejected_attention_masks.append(
                rejected_attention_masks[i] + [0] * pad_length
            )
        
        return {
            'chosen_input_ids': torch.tensor(padded_chosen_input_ids, dtype=torch.long),
            'chosen_attention_mask': torch.tensor(padded_chosen_attention_masks, dtype=torch.long),
            'rejected_input_ids': torch.tensor(padded_rejected_input_ids, dtype=torch.long),
            'rejected_attention_mask': torch.tensor(padded_rejected_attention_masks, dtype=torch.long)
        }
    
    def validate_raw_dataset(self, dataset: Union[Dataset, IterableDataset], dataset_type: str) -> Dict[str, Any]:
        """
        Validate raw dataset integrity and format compliance before preprocessing.
        
        Args:
            dataset: Raw dataset to validate
            dataset_type: Type of dataset ('sft' or 'preference')
            
        Returns:
            Dict containing validation results and statistics
            
        Requirement 7.3: Validate dataset integrity and format compliance before training
        """
        logger.info(f"Validating raw {dataset_type} dataset...")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {
                'total_examples': 0,
                'valid_examples': 0,
                'invalid_examples': 0,
                'filtered_examples': 0,
                'format_errors': 0,
                'content_warnings': 0,
                'quality_distribution': {},
                'validation_time': 0.0,
                'throughput_examples_per_second': 0.0
            }
        }
        
        import time
        start_time = time.time()
        
        try:
            # Handle streaming vs non-streaming datasets
            if isinstance(dataset, IterableDataset):
                # For streaming datasets, validate a representative sample
                sample_size = 2000  # Increased sample size for better validation
                try:
                    examples = list(dataset.take(sample_size))
                    validation_results['statistics']['total_examples'] = len(examples)
                    validation_results['statistics']['is_streaming'] = True
                except Exception as e:
                    logger.error(f"Failed to sample from streaming dataset: {e}")
                    validation_results['errors'].append(f"Streaming dataset sampling failed: {e}")
                    validation_results['is_valid'] = False
                    return validation_results
            else:
                # For regular datasets, validate all or a large sample
                max_validation_size = 10000
                if len(dataset) <= max_validation_size:
                    examples = list(dataset)
                else:
                    # Sample evenly across the dataset for better representation
                    step = len(dataset) // max_validation_size
                    indices = list(range(0, len(dataset), step))[:max_validation_size]
                    examples = [dataset[i] for i in indices]
                
                validation_results['statistics']['total_examples'] = len(examples)
                validation_results['statistics']['is_streaming'] = False
                validation_results['statistics']['dataset_size'] = len(dataset)
            
            # Enhanced batch validation for efficiency with parallel processing
            batch_size = 100
            quality_scores = []
            
            # Process batches with progress tracking
            total_batches = (len(examples) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(examples), batch_size):
                batch_end = min(batch_idx + batch_size, len(examples))
                batch = examples[batch_idx:batch_end]
                
                batch_results = self._validate_batch(batch, dataset_type, batch_idx)
                
                # Aggregate results
                validation_results['statistics']['valid_examples'] += batch_results['valid_count']
                validation_results['statistics']['invalid_examples'] += batch_results['invalid_count']
                validation_results['statistics']['format_errors'] += batch_results['format_errors']
                validation_results['statistics']['content_warnings'] += batch_results['content_warnings']
                
                validation_results['errors'].extend(batch_results['errors'])
                validation_results['warnings'].extend(batch_results['warnings'])
                quality_scores.extend(batch_results['quality_scores'])
                
                # Log progress for large datasets
                if total_batches > 10 and (batch_idx // batch_size + 1) % max(1, total_batches // 10) == 0:
                    progress = ((batch_idx // batch_size + 1) / total_batches) * 100
                    logger.info(f"Validation progress: {progress:.1f}% ({batch_idx + len(batch)}/{len(examples)} examples)")
            
            # Calculate final statistics
            total = validation_results['statistics']['total_examples']
            valid = validation_results['statistics']['valid_examples']
            invalid = validation_results['statistics']['invalid_examples']
            
            # Calculate timing statistics
            end_time = time.time()
            validation_time = end_time - start_time
            validation_results['statistics']['validation_time'] = validation_time
            validation_results['statistics']['throughput_examples_per_second'] = total / validation_time if validation_time > 0 else 0
            
            if total > 0:
                validation_results['statistics']['valid_percentage'] = (valid / total) * 100
                validation_results['statistics']['invalid_percentage'] = (invalid / total) * 100
                
                # Calculate enhanced quality distribution
                if quality_scores:
                    validation_results['statistics']['quality_distribution'] = self._calculate_quality_distribution(quality_scores)
                    validation_results['statistics']['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
                    validation_results['statistics']['min_quality_score'] = min(quality_scores)
                    validation_results['statistics']['max_quality_score'] = max(quality_scores)
                    validation_results['statistics']['quality_std'] = self._calculate_std(quality_scores)
            
            # Enhanced validity determination with configurable thresholds
            error_threshold = getattr(self.config.datasets, 'validation_error_threshold', 0.05)  # Allow up to 5% errors
            warning_threshold = getattr(self.config.datasets, 'validation_warning_threshold', 0.20)  # Allow up to 20% warnings
            
            if invalid > total * error_threshold:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"Too many invalid examples: {invalid}/{total} ({invalid/total*100:.1f}%) > {error_threshold*100:.1f}% threshold"
                )
            
            warning_count = len(validation_results['warnings'])
            if warning_count > total * warning_threshold:
                validation_results['warnings'].append(
                    f"High warning rate: {warning_count}/{total} ({warning_count/total*100:.1f}%) > {warning_threshold*100:.1f}% threshold"
                )
            
            # Enhanced logging with performance metrics
            logger.info(f"Dataset validation complete:")
            logger.info(f"  Total examples: {total}")
            logger.info(f"  Valid examples: {valid} ({valid/total*100:.1f}%)")
            logger.info(f"  Invalid examples: {invalid} ({invalid/total*100:.1f}%)")
            logger.info(f"  Warnings: {warning_count}")
            logger.info(f"  Validation time: {validation_time:.2f}s")
            logger.info(f"  Throughput: {validation_results['statistics']['throughput_examples_per_second']:.1f} examples/sec")
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                logger.info(f"  Average quality score: {avg_quality:.3f}")
                logger.info(f"  Quality range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation failed: {e}")
            logger.error(f"Dataset validation failed: {e}")
        
        return validation_results
    
    def _validate_batch(self, batch: List[Dict[str, Any]], dataset_type: str, batch_offset: int) -> Dict[str, Any]:
        """
        Validate a batch of examples efficiently.
        
        Args:
            batch: Batch of examples to validate
            dataset_type: Type of dataset ('sft' or 'preference')
            batch_offset: Offset for example numbering
            
        Returns:
            Dict containing batch validation results
        """
        results = {
            'valid_count': 0,
            'invalid_count': 0,
            'format_errors': 0,
            'content_warnings': 0,
            'errors': [],
            'warnings': [],
            'quality_scores': []
        }
        
        for i, example in enumerate(batch):
            example_idx = batch_offset + i
            
            try:
                if dataset_type == 'sft':
                    is_valid, warnings = self._validate_raw_sft_example(example)
                elif dataset_type == 'preference':
                    is_valid, warnings = self._validate_raw_preference_example(example)
                else:
                    results['errors'].append(f"Example {example_idx}: Unknown dataset type: {dataset_type}")
                    results['invalid_count'] += 1
                    continue
                
                if is_valid:
                    results['valid_count'] += 1
                    
                    # Calculate quality score for valid examples
                    quality_score = self._calculate_example_quality(example, dataset_type)
                    results['quality_scores'].append(quality_score)
                else:
                    results['invalid_count'] += 1
                    results['format_errors'] += 1
                
                if warnings:
                    results['warnings'].extend([f"Example {example_idx}: {w}" for w in warnings])
                    results['content_warnings'] += len(warnings)
            
            except Exception as e:
                results['errors'].append(f"Example {example_idx}: Validation error - {e}")
                results['invalid_count'] += 1
        
        return results
    
    def _calculate_example_quality(self, example: Dict[str, Any], dataset_type: str) -> float:
        """
        Calculate quality score for a raw dataset example.
        
        Args:
            example: Raw dataset example
            dataset_type: Type of dataset ('sft' or 'preference')
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        try:
            if dataset_type == 'sft':
                # Extract text content from SFT example
                messages = self._extract_messages_from_sft_example(example)
                if not messages:
                    return 0.0
                
                # Combine all message content
                combined_text = ' '.join(msg.get('content', '') for msg in messages)
                return self._calculate_content_quality(combined_text)
            
            elif dataset_type == 'preference':
                # Extract text content from preference example
                prompt, chosen, rejected = self._extract_preference_data(example)
                
                if not all([prompt, chosen, rejected]):
                    return 0.0
                
                # Calculate quality for each component and average
                prompt_quality = self._calculate_content_quality(prompt)
                chosen_quality = self._calculate_content_quality(chosen)
                rejected_quality = self._calculate_content_quality(rejected)
                
                # Weight prompt more heavily as it's shared
                return (prompt_quality * 0.4 + chosen_quality * 0.3 + rejected_quality * 0.3)
            
            else:
                return 0.0
        
        except Exception as e:
            logger.debug(f"Error calculating example quality: {e}")
            return 0.0
    
    def _validate_raw_sft_example(self, example: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single raw SFT example.
        
        Args:
            example: Raw SFT example
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Check for required fields based on common SFT formats
        has_messages = 'messages' in example
        has_conversations = 'conversations' in example
        has_instruction_response = 'instruction' in example and ('response' in example or 'output' in example)
        
        if not (has_messages or has_conversations or has_instruction_response):
            return False, ["No valid conversation format found"]
        
        # Extract and validate messages
        messages = []
        if has_messages:
            messages = example['messages']
        elif has_conversations:
            messages = example['conversations']
        elif has_instruction_response:
            messages = [
                {"role": "user", "content": example['instruction']},
                {"role": "assistant", "content": example.get('response', example.get('output', ''))}
            ]
        
        if not messages:
            return False, ["Empty messages list"]
        
        # Validate message structure
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, [f"Message {i} is not a dictionary"]
            
            if 'role' not in msg or 'content' not in msg:
                return False, [f"Message {i} missing 'role' or 'content'"]
            
            if msg['role'] not in ['system', 'user', 'assistant']:
                return False, [f"Message {i} has invalid role: {msg['role']}"]
            
            # Check content quality
            content = str(msg['content']).strip()
            if not content:
                warnings.append(f"Message {i} has empty content")
            elif len(content) < 3:
                warnings.append(f"Message {i} has very short content")
            elif len(content) > 10000:
                warnings.append(f"Message {i} has very long content ({len(content)} chars)")
            
            # Check for harmful content
            if self._contains_harmful_content(content):
                warnings.append(f"Message {i} may contain harmful content")
        
        # Check conversation structure
        if not any(msg['role'] == 'assistant' for msg in messages):
            warnings.append("No assistant response found")
        
        return True, warnings
    
    def _validate_raw_preference_example(self, example: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single raw preference example.
        
        Args:
            example: Raw preference example
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Check for required fields
        required_fields = ['prompt', 'chosen', 'rejected']
        for field in required_fields:
            if field not in example:
                return False, [f"Missing required field: {field}"]
        
        # Validate field contents
        prompt = str(example['prompt']).strip()
        chosen = str(example['chosen']).strip()
        rejected = str(example['rejected']).strip()
        
        if not prompt:
            return False, ["Empty prompt"]
        if not chosen:
            return False, ["Empty chosen response"]
        if not rejected:
            return False, ["Empty rejected response"]
        
        # Check for identical responses
        if chosen == rejected:
            warnings.append("Chosen and rejected responses are identical")
        
        # Check content lengths
        if len(prompt) < 10:
            warnings.append("Very short prompt")
        elif len(prompt) > 5000:
            warnings.append(f"Very long prompt ({len(prompt)} chars)")
        
        if len(chosen) < 10:
            warnings.append("Very short chosen response")
        elif len(chosen) > 10000:
            warnings.append(f"Very long chosen response ({len(chosen)} chars)")
        
        if len(rejected) < 10:
            warnings.append("Very short rejected response")
        elif len(rejected) > 10000:
            warnings.append(f"Very long rejected response ({len(rejected)} chars)")
        
        # Check for harmful content
        for field_name, content in [('prompt', prompt), ('chosen', chosen), ('rejected', rejected)]:
            if self._contains_harmful_content(content):
                warnings.append(f"{field_name} may contain harmful content")
        
        return True, warnings
    
    def _contains_harmful_content(self, text: str) -> bool:
        """
        Check if text contains potentially harmful content.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if harmful content detected
        """
        text_lower = text.lower()
        
        for pattern in self._harmful_patterns:
            if pattern.search(text_lower):
                return True
        
        return False
    
    def filter_dataset_content(self, dataset: Union[Dataset, IterableDataset], dataset_type: str) -> Union[Dataset, IterableDataset]:
        """
        Filter dataset to remove examples with harmful or inappropriate content.
        
        Args:
            dataset: Dataset to filter
            dataset_type: Type of dataset ('sft' or 'preference')
            
        Returns:
            Filtered dataset
            
        Requirement 7.3: Validate dataset integrity and format compliance before training
        """
        logger.info(f"Filtering {dataset_type} dataset for harmful content...")
        
        def should_keep_example(example):
            """Determine if an example should be kept."""
            try:
                if dataset_type == 'sft':
                    return self._should_keep_sft_example(example)
                elif dataset_type == 'preference':
                    return self._should_keep_preference_example(example)
                else:
                    return True  # Keep if unknown type
            except Exception as e:
                logger.warning(f"Error filtering example: {e}")
                return False  # Remove problematic examples
        
        if isinstance(dataset, IterableDataset):
            # For streaming datasets, filter on-the-fly
            filtered_dataset = dataset.filter(should_keep_example)
        else:
            # For regular datasets, filter and track statistics
            original_size = len(dataset)
            filtered_dataset = dataset.filter(should_keep_example)
            filtered_size = len(filtered_dataset)
            
            logger.info(f"Content filtering complete: {filtered_size}/{original_size} examples kept "
                       f"({filtered_size/original_size*100:.1f}%)")
        
        return filtered_dataset
    
    def _should_keep_sft_example(self, example: Dict[str, Any]) -> bool:
        """Determine if an SFT example should be kept after filtering."""
        # Extract messages
        messages = []
        if 'messages' in example:
            messages = example['messages']
        elif 'conversations' in example:
            messages = example['conversations']
        elif 'instruction' in example:
            messages = [
                {"role": "user", "content": example['instruction']},
                {"role": "assistant", "content": example.get('response', example.get('output', ''))}
            ]
        
        # Check each message for harmful content
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                if self._contains_harmful_content(str(msg['content'])):
                    return False
        
        return True
    
    def _should_keep_preference_example(self, example: Dict[str, Any]) -> bool:
        """Determine if a preference example should be kept after filtering."""
        # Check all text fields for harmful content
        for field in ['prompt', 'chosen', 'rejected']:
            if field in example:
                if self._contains_harmful_content(str(example[field])):
                    return False
        
        return True
    
    def _validate_sft_example(self, example: Dict[str, Any]) -> bool:
        """Validate a single SFT example."""
        required_fields = ['input_ids', 'attention_mask', 'labels']
        
        for field in required_fields:
            if field not in example:
                logger.error(f"Missing required field: {field}")
                return False
            
            if not isinstance(example[field], list):
                logger.error(f"Field {field} must be a list")
                return False
            
            if len(example[field]) == 0:
                logger.error(f"Field {field} cannot be empty")
                return False
        
        # Check that all sequences have the same length
        lengths = [len(example[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            logger.error("input_ids, attention_mask, and labels must have the same length")
            return False
        
        return True
    
    def _validate_preference_example(self, example: Dict[str, Any]) -> bool:
        """Validate a single preference example."""
        required_fields = [
            'chosen_input_ids', 'chosen_attention_mask',
            'rejected_input_ids', 'rejected_attention_mask'
        ]
        
        for field in required_fields:
            if field not in example:
                logger.error(f"Missing required field: {field}")
                return False
            
            if not isinstance(example[field], list):
                logger.error(f"Field {field} must be a list")
                return False
            
            if len(example[field]) == 0:
                logger.error(f"Field {field} cannot be empty")
                return False
        
        # Check that chosen sequences have consistent lengths
        if len(example['chosen_input_ids']) != len(example['chosen_attention_mask']):
            logger.error("chosen_input_ids and chosen_attention_mask must have the same length")
            return False
        
        # Check that rejected sequences have consistent lengths
        if len(example['rejected_input_ids']) != len(example['rejected_attention_mask']):
            logger.error("rejected_input_ids and rejected_attention_mask must have the same length")
            return False
        
        return True
    
    def create_dataset_pipeline(
        self,
        dataset_type: str,
        dataset_name: Optional[str] = None,
        streaming: bool = False,
        validate_raw: bool = True,
        filter_content: bool = True,
        validate_processed: bool = True,
        max_samples: Optional[int] = None
    ) -> Union[Dataset, IterableDataset]:
        """
        Create a complete dataset processing pipeline with enhanced streaming support.
        
        Args:
            dataset_type: Type of dataset ('sft' or 'preference')
            dataset_name: Optional dataset name
            streaming: Whether to use streaming mode
            validate_raw: Whether to validate raw dataset
            filter_content: Whether to filter harmful content
            validate_processed: Whether to validate processed dataset
            max_samples: Optional limit on number of samples to process
            
        Returns:
            Fully processed and validated dataset
            
        Requirement 7.3: Validate dataset integrity and format compliance before training
        Requirement 7.5: Support both instruction-following datasets for SFT and preference datasets for reward training
        Requirement 5.4: Use streaming datasets to avoid loading entire datasets into memory
        """
        logger.info(f"Creating {dataset_type} dataset pipeline (streaming={streaming}, max_samples={max_samples})")
        
        # Step 1: Load raw dataset
        if dataset_type == 'sft':
            raw_dataset = self.load_sft_dataset(dataset_name, streaming=streaming)
        elif dataset_type == 'preference':
            raw_dataset = self.load_preference_dataset(dataset_name, streaming=streaming)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Step 2: Apply sample limit for streaming datasets if specified
        if streaming and max_samples is not None:
            raw_dataset = raw_dataset.take(max_samples)
            logger.info(f"Limited streaming dataset to {max_samples} samples")
        
        # Step 3: Validate raw dataset
        if validate_raw:
            validation_results = self.validate_raw_dataset(raw_dataset, dataset_type)
            if not validation_results['is_valid']:
                logger.error(f"Raw dataset validation failed: {validation_results['errors']}")
                raise ValueError(f"Raw dataset validation failed")
            
            if validation_results['warnings']:
                logger.warning(f"Dataset validation warnings: {len(validation_results['warnings'])} warnings")
                
            # Log validation performance metrics
            stats = validation_results['statistics']
            if 'throughput_examples_per_second' in stats:
                logger.info(f"Validation throughput: {stats['throughput_examples_per_second']:.1f} examples/sec")
        
        # Step 4: Filter harmful content
        if filter_content:
            raw_dataset = self.filter_dataset_content(raw_dataset, dataset_type)
        
        # Step 5: Preprocess dataset with enhanced error handling
        try:
            if dataset_type == 'sft':
                processed_dataset = self.preprocess_sft_data(raw_dataset, validate=validate_processed)
            elif dataset_type == 'preference':
                processed_dataset = self.preprocess_preference_data(raw_dataset, validate=validate_processed)
        except Exception as e:
            logger.error(f"Dataset preprocessing failed: {e}")
            raise ValueError(f"Dataset preprocessing failed: {e}")
        
        # Step 6: Final validation of processed dataset
        if validate_processed and not isinstance(processed_dataset, IterableDataset):
            if not self.validate_dataset_format(processed_dataset, dataset_type):
                raise ValueError("Processed dataset validation failed")
        
        # Step 7: Log final statistics
        if isinstance(processed_dataset, IterableDataset):
            logger.info(f"Dataset pipeline complete for {dataset_type} (streaming)")
        else:
            logger.info(f"Dataset pipeline complete for {dataset_type}: {len(processed_dataset)} examples")
            
        return processed_dataset
    
    def validate_dataset_format(self, dataset: Dataset, dataset_type: str) -> bool:
        """
        Validate processed dataset format and integrity with comprehensive checks.
        
        Args:
            dataset: Processed dataset to validate
            dataset_type: Type of dataset ('sft' or 'preference')
            
        Returns:
            bool: True if dataset is valid, False otherwise
            
        Requirement 7.3: Validate dataset integrity and format compliance before training
        """
        try:
            if len(dataset) == 0:
                logger.error("Dataset is empty")
                return False
            
            # Sample a representative set of examples for validation
            sample_size = min(100, len(dataset))
            sample_indices = list(range(0, len(dataset), max(1, len(dataset) // sample_size)))[:sample_size]
            
            validation_errors = []
            validation_warnings = []
            
            for i in sample_indices:
                try:
                    example = dataset[i]
                    
                    if dataset_type == 'sft':
                        errors, warnings = self._validate_sft_example_comprehensive(example)
                    elif dataset_type == 'preference':
                        errors, warnings = self._validate_preference_example_comprehensive(example)
                    else:
                        logger.error(f"Unknown dataset type: {dataset_type}")
                        return False
                    
                    validation_errors.extend([f"Example {i}: {e}" for e in errors])
                    validation_warnings.extend([f"Example {i}: {w}" for w in warnings])
                    
                except Exception as e:
                    validation_errors.append(f"Example {i}: Validation exception - {e}")
            
            # Log validation results
            if validation_errors:
                logger.error(f"Dataset validation failed with {len(validation_errors)} errors:")
                for error in validation_errors[:10]:  # Log first 10 errors
                    logger.error(f"  {error}")
                if len(validation_errors) > 10:
                    logger.error(f"  ... and {len(validation_errors) - 10} more errors")
                return False
            
            if validation_warnings:
                logger.warning(f"Dataset validation completed with {len(validation_warnings)} warnings:")
                for warning in validation_warnings[:5]:  # Log first 5 warnings
                    logger.warning(f"  {warning}")
                if len(validation_warnings) > 5:
                    logger.warning(f"  ... and {len(validation_warnings) - 5} more warnings")
            
            logger.info(f"Dataset validation passed for {dataset_type} dataset ({sample_size} examples checked)")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False
    
    def _validate_sft_example_comprehensive(self, example: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Comprehensive validation of a single SFT example.
        
        Args:
            example: SFT example to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['input_ids', 'attention_mask', 'labels']
        for field in required_fields:
            if field not in example:
                errors.append(f"Missing required field: {field}")
                continue
            
            if not isinstance(example[field], list):
                errors.append(f"Field {field} must be a list, got {type(example[field])}")
                continue
            
            if len(example[field]) == 0:
                errors.append(f"Field {field} cannot be empty")
        
        if errors:
            return errors, warnings
        
        # Check sequence length consistency
        lengths = [len(example[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            errors.append(f"Inconsistent sequence lengths: {dict(zip(required_fields, lengths))}")
        
        # Check sequence length bounds
        seq_length = len(example['input_ids'])
        if seq_length < 3:
            errors.append(f"Sequence too short: {seq_length} tokens")
        elif seq_length > self.config.model.max_length:
            errors.append(f"Sequence too long: {seq_length} > {self.config.model.max_length}")
        
        # Check token ID validity
        input_ids = example['input_ids']
        if any(token_id < 0 for token_id in input_ids):
            errors.append("Negative token IDs found")
        
        # Check attention mask validity
        attention_mask = example['attention_mask']
        if not all(mask in [0, 1] for mask in attention_mask):
            errors.append("Invalid attention mask values (must be 0 or 1)")
        
        # Check labels validity
        labels = example['labels']
        valid_label_values = set(input_ids + [-100])
        if not all(label in valid_label_values for label in labels):
            warnings.append("Some label values don't match input_ids or -100")
        
        # Check for reasonable label distribution
        non_masked_labels = [l for l in labels if l != -100]
        if len(non_masked_labels) == 0:
            warnings.append("All labels are masked (-100)")
        elif len(non_masked_labels) == len(labels):
            warnings.append("No labels are masked (might not be instruction tuning format)")
        
        # Check text content if available
        if 'text' in example:
            text = example['text']
            if not isinstance(text, str):
                warnings.append(f"Text field should be string, got {type(text)}")
            elif len(text.strip()) == 0:
                warnings.append("Empty text content")
            elif self._contains_harmful_content(text):
                warnings.append("Text may contain harmful content")
        
        return errors, warnings
    
    def _validate_preference_example_comprehensive(self, example: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Comprehensive validation of a single preference example.
        
        Args:
            example: Preference example to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = [
            'chosen_input_ids', 'chosen_attention_mask',
            'rejected_input_ids', 'rejected_attention_mask'
        ]
        
        for field in required_fields:
            if field not in example:
                errors.append(f"Missing required field: {field}")
                continue
            
            if not isinstance(example[field], list):
                errors.append(f"Field {field} must be a list, got {type(example[field])}")
                continue
            
            if len(example[field]) == 0:
                errors.append(f"Field {field} cannot be empty")
        
        if errors:
            return errors, warnings
        
        # Check chosen sequence consistency
        chosen_length = len(example['chosen_input_ids'])
        chosen_mask_length = len(example['chosen_attention_mask'])
        if chosen_length != chosen_mask_length:
            errors.append(f"Chosen sequence length mismatch: {chosen_length} vs {chosen_mask_length}")
        
        # Check rejected sequence consistency
        rejected_length = len(example['rejected_input_ids'])
        rejected_mask_length = len(example['rejected_attention_mask'])
        if rejected_length != rejected_mask_length:
            errors.append(f"Rejected sequence length mismatch: {rejected_length} vs {rejected_mask_length}")
        
        # Check sequence length bounds
        for seq_type, seq_length in [('chosen', chosen_length), ('rejected', rejected_length)]:
            if seq_length < 3:
                errors.append(f"{seq_type} sequence too short: {seq_length} tokens")
            elif seq_length > self.config.model.max_length:
                errors.append(f"{seq_type} sequence too long: {seq_length} > {self.config.model.max_length}")
        
        # Check token ID validity
        for field_name, token_ids in [
            ('chosen_input_ids', example['chosen_input_ids']),
            ('rejected_input_ids', example['rejected_input_ids'])
        ]:
            if any(token_id < 0 for token_id in token_ids):
                errors.append(f"Negative token IDs found in {field_name}")
        
        # Check attention mask validity
        for field_name, attention_mask in [
            ('chosen_attention_mask', example['chosen_attention_mask']),
            ('rejected_attention_mask', example['rejected_attention_mask'])
        ]:
            if not all(mask in [0, 1] for mask in attention_mask):
                errors.append(f"Invalid attention mask values in {field_name} (must be 0 or 1)")
        
        # Check text content if available
        text_fields = ['chosen_text', 'rejected_text', 'prompt']
        for field in text_fields:
            if field in example:
                text = example[field]
                if not isinstance(text, str):
                    warnings.append(f"{field} should be string, got {type(text)}")
                elif len(text.strip()) == 0:
                    warnings.append(f"Empty {field} content")
                elif self._contains_harmful_content(text):
                    warnings.append(f"{field} may contain harmful content")
        
        # Check for identical chosen/rejected responses
        if 'chosen_text' in example and 'rejected_text' in example:
            if example['chosen_text'].strip() == example['rejected_text'].strip():
                warnings.append("Chosen and rejected responses are identical")
        
        return errors, warnings
    
    def get_dataset_statistics(self, dataset: Union[Dataset, IterableDataset]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dict containing detailed dataset statistics
        """
        stats = {
            'type': 'streaming' if isinstance(dataset, IterableDataset) else 'regular',
            'validation_stats': dict(self._validation_stats) if self._validation_stats else {}
        }
        
        if isinstance(dataset, IterableDataset):
            # For streaming datasets, analyze a sample
            sample_size = 1000
            try:
                sample = list(dataset.take(sample_size))
                stats['sample_size'] = len(sample)
                stats['estimated_total'] = 'unknown'
                
                if sample:
                    stats.update(self._analyze_dataset_sample(sample))
            except Exception as e:
                logger.warning(f"Failed to analyze streaming dataset sample: {e}")
                stats['error'] = str(e)
        else:
            # For regular datasets, get full statistics
            stats['total_examples'] = len(dataset)
            stats['column_names'] = dataset.column_names
            stats['features'] = str(dataset.features)
            
            # Analyze a sample for detailed statistics
            sample_size = min(1000, len(dataset))
            sample = [dataset[i] for i in range(sample_size)]
            stats.update(self._analyze_dataset_sample(sample))
        
        return stats
    
    def _analyze_dataset_sample(self, sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a sample of dataset examples with enhanced statistics.
        
        Args:
            sample: List of dataset examples to analyze
            
        Returns:
            Dict containing detailed analysis results
        """
        if not sample:
            return {}
        
        analysis = {
            'sample_size': len(sample),
            'avg_input_length': 0,
            'avg_text_length': 0,
            'sequence_length_stats': {},
            'content_stats': {},
            'quality_metrics': {}
        }
        
        # Analyze sequence lengths
        input_lengths = []
        text_lengths = []
        content_quality_scores = []
        
        for example in sample:
            if 'input_ids' in example:
                input_lengths.append(len(example['input_ids']))
            
            # Analyze text content
            texts_to_analyze = []
            if 'text' in example:
                texts_to_analyze.append(example['text'])
            elif 'chosen_text' in example and 'rejected_text' in example:
                texts_to_analyze.extend([example['chosen_text'], example['rejected_text']])
            
            for text in texts_to_analyze:
                text_lengths.append(len(text))
                
                # Basic content quality metrics
                quality_score = self._calculate_content_quality(text)
                content_quality_scores.append(quality_score)
        
        # Calculate sequence length statistics
        if input_lengths:
            analysis['sequence_length_stats'] = {
                'min': min(input_lengths),
                'max': max(input_lengths),
                'mean': sum(input_lengths) / len(input_lengths),
                'median': sorted(input_lengths)[len(input_lengths) // 2],
                'std': self._calculate_std(input_lengths)
            }
            analysis['avg_input_length'] = analysis['sequence_length_stats']['mean']
        
        # Calculate text length statistics
        if text_lengths:
            analysis['avg_text_length'] = sum(text_lengths) / len(text_lengths)
            analysis['content_stats'] = {
                'min_text_length': min(text_lengths),
                'max_text_length': max(text_lengths),
                'median_text_length': sorted(text_lengths)[len(text_lengths) // 2]
            }
        
        # Calculate quality metrics
        if content_quality_scores:
            analysis['quality_metrics'] = {
                'avg_quality_score': sum(content_quality_scores) / len(content_quality_scores),
                'min_quality_score': min(content_quality_scores),
                'max_quality_score': max(content_quality_scores),
                'quality_distribution': self._calculate_quality_distribution(content_quality_scores)
            }
        
        return analysis
    
    def _calculate_content_quality(self, text: str) -> float:
        """
        Calculate a basic content quality score for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        score = 1.0
        text_clean = text.strip()
        
        # Penalize very short content
        if len(text_clean) < 10:
            score *= 0.3
        elif len(text_clean) < 50:
            score *= 0.7
        
        # Penalize very long content (might be spam or low quality)
        if len(text_clean) > 5000:
            score *= 0.8
        elif len(text_clean) > 10000:
            score *= 0.5
        
        # Check for repetitive content
        words = text_clean.lower().split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.3:  # Very repetitive
                score *= 0.4
            elif repetition_ratio < 0.5:
                score *= 0.7
        
        # Check for proper sentence structure
        sentences = text_clean.split('.')
        if len(sentences) > 1:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 3:  # Very short sentences
                score *= 0.6
            elif avg_sentence_length > 50:  # Very long sentences
                score *= 0.8
        
        # Check for harmful content (additional penalty)
        if self._contains_harmful_content(text):
            score *= 0.1  # Heavy penalty for harmful content
        
        return max(0.0, min(1.0, score))
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _calculate_quality_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate distribution of quality scores."""
        distribution = {
            'excellent': 0,  # 0.8-1.0
            'good': 0,       # 0.6-0.8
            'fair': 0,       # 0.4-0.6
            'poor': 0        # 0.0-0.4
        }
        
        for score in scores:
            if score >= 0.8:
                distribution['excellent'] += 1
            elif score >= 0.6:
                distribution['good'] += 1
            elif score >= 0.4:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        self._dataset_cache.clear()
        logger.info("Dataset cache cleared")
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for the dataset manager.
        
        Returns:
            Dict containing memory usage information
        """
        import psutil
        import sys
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_percent': process.memory_percent(),
            'cached_datasets': len(self._dataset_cache),
            'cache_keys': list(self._dataset_cache.keys()),
            'python_objects_count': len(gc.get_objects()) if 'gc' in sys.modules else 0
        }
        
        # Estimate cache memory usage
        cache_size_estimate = 0
        for key, dataset in self._dataset_cache.items():
            if hasattr(dataset, '__len__'):
                # Rough estimate: assume 1KB per example
                cache_size_estimate += len(dataset) * 1024
        
        stats['estimated_cache_size_mb'] = cache_size_estimate / 1024 / 1024
        
        return stats
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage by clearing unnecessary caches and running garbage collection.
        
        Returns:
            Dict containing optimization results
        """
        import gc
        
        # Get initial memory stats
        initial_stats = self.get_memory_usage_stats()
        
        # Clear dataset cache
        cache_count = len(self._dataset_cache)
        self.clear_cache()
        
        # Clear validation statistics
        self._validation_stats.clear()
        
        # Run garbage collection
        collected = gc.collect()
        
        # Get final memory stats
        final_stats = self.get_memory_usage_stats()
        
        optimization_results = {
            'cleared_cached_datasets': cache_count,
            'garbage_collected_objects': collected,
            'memory_before_mb': initial_stats['process_memory_mb'],
            'memory_after_mb': final_stats['process_memory_mb'],
            'memory_freed_mb': initial_stats['process_memory_mb'] - final_stats['process_memory_mb'],
            'memory_reduction_percent': (
                (initial_stats['process_memory_mb'] - final_stats['process_memory_mb']) / 
                initial_stats['process_memory_mb'] * 100
            ) if initial_stats['process_memory_mb'] > 0 else 0
        }
        
        logger.info(f"Memory optimization complete:")
        logger.info(f"  Cleared {cache_count} cached datasets")
        logger.info(f"  Collected {collected} garbage objects")
        logger.info(f"  Memory freed: {optimization_results['memory_freed_mb']:.1f} MB")
        logger.info(f"  Memory reduction: {optimization_results['memory_reduction_percent']:.1f}%")
        
        return optimization_results
    
    def create_streaming_dataloader(
        self,
        dataset: IterableDataset,
        batch_size: int,
        buffer_size: int = 1000,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create an optimized DataLoader for streaming datasets with buffering.
        
        Args:
            dataset: Streaming dataset
            batch_size: Batch size
            buffer_size: Buffer size for prefetching
            num_workers: Number of worker processes
            
        Returns:
            Optimized DataLoader for streaming
        """
        def collate_fn(batch):
            """Custom collate function for proper padding."""
            # Determine if this is SFT or preference data
            if 'chosen_input_ids' in batch[0]:
                return self._collate_preference_batch(batch)
            else:
                return self._collate_sft_batch(batch)
        
        # Add buffering for streaming datasets
        if hasattr(dataset, 'shuffle'):
            dataset = dataset.shuffle(buffer_size=buffer_size)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else 2,
            persistent_workers=num_workers > 0
        )
    
    def load_dataset_with_fallback(self, 
                                  dataset_name: str,
                                  dataset_type: str = "sft",
                                  streaming: bool = False,
                                  max_retries: int = 3) -> Union[Dataset, IterableDataset, None]:
        """
        Load dataset with automatic fallback to alternative sources.
        
        Args:
            dataset_name: Primary dataset name to load
            dataset_type: Type of dataset ("sft" or "preference")
            streaming: Whether to use streaming mode
            max_retries: Maximum retry attempts for each dataset
            
        Returns:
            Loaded dataset or None if all attempts failed
            
        Requirement 9.2: Dataset loading fallbacks with alternative sources and clear guidance
        """
        # Define alternative datasets
        alternatives = {
            'sft': [
                "microsoft/orca-math-word-problems-200k",
                "Open-Orca/OpenOrca", 
                "teknium/OpenHermes-2.5",
                "argilla/distilabel-intel-orca-dpo-pairs"
            ],
            'preference': [
                "Anthropic/hh-rlhf",
                "argilla/ultrafeedback-binarized-preferences-cleaned", 
                "Intel/orca_dpo_pairs",
                "jondurbin/truthy-dpo-v0.1"
            ]
        }
        
        # Try primary dataset first
        datasets_to_try = [dataset_name] + alternatives.get(dataset_type, [])
        
        for attempt_dataset in datasets_to_try:
            for retry in range(max_retries):
                try:
                    logger.info(f"Attempting to load {dataset_type} dataset: {attempt_dataset} (attempt {retry + 1}/{max_retries})")
                    
                    if dataset_type == "sft":
                        dataset = self.load_sft_dataset(attempt_dataset, streaming=streaming)
                    else:
                        dataset = self.load_preference_dataset(attempt_dataset, streaming=streaming)
                    
                    if dataset is not None:
                        if attempt_dataset != dataset_name:
                            logger.warning(f"Using fallback dataset: {attempt_dataset} (original: {dataset_name})")
                        else:
                            logger.info(f"Successfully loaded primary dataset: {dataset_name}")
                        
                        return dataset
                        
                except Exception as e:
                    logger.warning(f"Failed to load {attempt_dataset} (attempt {retry + 1}): {str(e)}")
                    
                    # Wait before retry
                    if retry < max_retries - 1:
                        import time
                        wait_time = 2 ** retry  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
        
        # All attempts failed
        logger.error(f"Failed to load any {dataset_type} dataset after trying all alternatives")
        self._log_dataset_recovery_instructions(dataset_name, dataset_type)
        return None
    
    def _log_dataset_recovery_instructions(self, dataset_name: str, dataset_type: str):
        """Log detailed recovery instructions for dataset loading failures."""
        instructions = [
            "",
            "=" * 60,
            f"DATASET LOADING FAILED: {dataset_name} ({dataset_type})",
            "=" * 60,
            "",
            "RECOVERY INSTRUCTIONS:",
            "",
            "1. CHECK INTERNET CONNECTION:",
            "   - Verify you have a stable internet connection",
            "   - Try accessing https://huggingface.co/ in your browser",
            "",
            "2. VERIFY DATASET EXISTS:",
            f"   - Check if {dataset_name} exists on HuggingFace Hub",
            "   - Verify the dataset name spelling and format",
            "   - Ensure the dataset is publicly accessible",
            "",
            "3. TRY ALTERNATIVE DATASETS:",
            "   SFT alternatives:",
            "   - microsoft/orca-math-word-problems-200k",
            "   - Open-Orca/OpenOrca",
            "   - teknium/OpenHermes-2.5",
            "",
            "   Preference alternatives:",
            "   - Anthropic/hh-rlhf", 
            "   - argilla/ultrafeedback-binarized-preferences-cleaned",
            "   - Intel/orca_dpo_pairs",
            "",
            "4. CLEAR DATASET CACHE:",
            "   - Delete cached datasets that might be corrupted",
            f"   - Cache location: {self.cache_dir}",
            "",
            "5. USE STREAMING MODE:",
            "   - Try loading with streaming=True for large datasets",
            "   - This reduces memory usage and can bypass some loading issues",
            "",
            "6. CHECK HUGGINGFACE TOKEN:",
            "   - Some datasets require authentication",
            "   - Set HF_TOKEN environment variable if needed",
            "",
            "7. MANUAL DOWNLOAD:",
            "   - Download dataset manually and load from local path",
            "   - Use datasets.load_from_disk() for local datasets",
            "",
            "=" * 60
        ]
        
        for instruction in instructions:
            logger.error(instruction)
    
    def validate_dataset_health(self, dataset: Union[Dataset, IterableDataset]) -> Dict[str, Any]:
        """
        Validate dataset health and provide diagnostic information.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        health_report = {
            "dataset_type": type(dataset).__name__,
            "is_streaming": isinstance(dataset, IterableDataset),
            "sample_count": "unknown" if isinstance(dataset, IterableDataset) else len(dataset),
            "column_names": getattr(dataset, 'column_names', []),
            "features": str(getattr(dataset, 'features', {})),
            "issues": [],
            "recommendations": [],
            "sample_validation": {}
        }
        
        try:
            # Test loading a few samples
            if isinstance(dataset, IterableDataset):
                # For streaming datasets, take first few items
                samples = []
                for i, sample in enumerate(dataset):
                    samples.append(sample)
                    if i >= 2:  # Take 3 samples
                        break
            else:
                # For regular datasets, take first few samples
                sample_count = min(3, len(dataset))
                samples = [dataset[i] for i in range(sample_count)]
            
            if not samples:
                health_report["issues"].append("Dataset appears to be empty")
                health_report["recommendations"].append("Check dataset loading parameters")
                return health_report
            
            # Validate sample structure
            first_sample = samples[0]
            required_fields = {
                'sft': ['messages', 'conversations', 'instruction', 'input', 'question'],
                'preference': ['prompt', 'chosen', 'rejected', 'question', 'response_a', 'response_b']
            }
            
            # Detect dataset type based on fields
            sample_fields = set(first_sample.keys()) if isinstance(first_sample, dict) else set()
            
            dataset_format = "unknown"
            for format_type, fields in required_fields.items():
                if any(field in sample_fields for field in fields):
                    dataset_format = format_type
                    break
            
            health_report["detected_format"] = dataset_format
            health_report["sample_fields"] = list(sample_fields)
            
            # Validate content
            content_issues = []
            for i, sample in enumerate(samples):
                try:
                    if dataset_format == "sft":
                        messages = self._extract_messages_from_sft_example(sample)
                        if not messages:
                            content_issues.append(f"Sample {i}: No valid messages found")
                        elif self._contains_harmful_content(str(sample)):
                            content_issues.append(f"Sample {i}: Contains potentially harmful content")
                    
                    elif dataset_format == "preference":
                        prompt, chosen, rejected = self._extract_preference_data(sample)
                        if not all([prompt, chosen, rejected]):
                            content_issues.append(f"Sample {i}: Missing required preference fields")
                        elif chosen == rejected:
                            content_issues.append(f"Sample {i}: Chosen and rejected responses are identical")
                
                except Exception as e:
                    content_issues.append(f"Sample {i}: Validation error - {str(e)}")
            
            health_report["sample_validation"]["issues"] = content_issues
            health_report["sample_validation"]["samples_checked"] = len(samples)
            
            # Generate recommendations
            if content_issues:
                health_report["recommendations"].append("Review dataset quality and preprocessing")
            
            if dataset_format == "unknown":
                health_report["recommendations"].append("Dataset format not recognized - may need custom preprocessing")
            
            if not isinstance(dataset, IterableDataset) and len(dataset) < 100:
                health_report["recommendations"].append("Dataset is very small - consider using a larger dataset")
            
        except Exception as e:
            health_report["issues"].append(f"Health validation failed: {str(e)}")
            health_report["recommendations"].append("Check dataset accessibility and format")
        
        return health_report
    
    def clear_corrupted_cache(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear potentially corrupted dataset cache files.
        
        Args:
            dataset_name: Specific dataset to clear, or None for all
            
        Returns:
            Dictionary with cleanup results
        """
        cleanup_results = {
            "files_removed": 0,
            "space_freed_mb": 0,
            "errors": []
        }
        
        try:
            if dataset_name:
                # Clear specific dataset cache
                cache_pattern = f"*{dataset_name.replace('/', '_')}*"
                cache_files = list(self.cache_dir.glob(cache_pattern))
            else:
                # Clear all cache files
                cache_files = list(self.cache_dir.rglob("*"))
                cache_files = [f for f in cache_files if f.is_file()]
            
            total_size = 0
            for cache_file in cache_files:
                try:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    total_size += file_size
                    cleanup_results["files_removed"] += 1
                except Exception as e:
                    cleanup_results["errors"].append(f"Failed to remove {cache_file}: {str(e)}")
            
            cleanup_results["space_freed_mb"] = total_size / (1024 * 1024)
            
            # Clear in-memory cache as well
            if dataset_name:
                keys_to_remove = [k for k in self._dataset_cache.keys() if dataset_name in k]
                for key in keys_to_remove:
                    del self._dataset_cache[key]
            else:
                self._dataset_cache.clear()
            
            logger.info(f"Cache cleanup complete:")
            logger.info(f"  Files removed: {cleanup_results['files_removed']}")
            logger.info(f"  Space freed: {cleanup_results['space_freed_mb']:.1f} MB")
            
            if cleanup_results["errors"]:
                logger.warning(f"  Errors encountered: {len(cleanup_results['errors'])}")
        
        except Exception as e:
            cleanup_results["errors"].append(f"Cache cleanup failed: {str(e)}")
            logger.error(f"Failed to clear cache: {str(e)}")
        
        return cleanup_results