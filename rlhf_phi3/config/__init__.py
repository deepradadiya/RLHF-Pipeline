"""
Configuration Management Module

Provides centralized configuration management for all hyperparameters, 
paths, and environment settings across the RLHF pipeline.
"""

from .config_manager import Config

__all__ = ["Config"]