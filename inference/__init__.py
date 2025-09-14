"""
Inference Module

This module contains all inference-related functionality for the Bizcompass benchmark.
It includes API-based, local model, and debug inference capabilities.
"""

from .api_inference import APIInference
from .local_inference import LocalInference
from .debug_inference import DebugInference

__all__ = ['APIInference', 'LocalInference', 'DebugInference']
