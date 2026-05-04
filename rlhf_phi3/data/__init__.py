"""
Dataset Management Module

Handles dataset loading, preprocessing, and formatting for all training stages
including SFT instruction datasets and preference datasets for reward modeling.
"""

from .dataset_manager import DatasetManager

__all__ = ["DatasetManager"]