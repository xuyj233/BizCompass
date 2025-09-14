#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Management for Bizcompass Benchmark

This module provides a centralized configuration system for all configurable
parameters in the Bizcompass benchmark. It supports loading from environment
variables, configuration files, and command-line arguments.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    API = "api"
    LOCAL = "local"


class QuestionType(Enum):
    """Supported question types."""
    SINGLE_CHOICE = "Single Choice"
    MULTIPLE_CHOICE = "Multiple Choice"
    GENERAL_QA = "General QA"
    TABLE_QA = "Table QA"


class Domain(Enum):
    """Supported domains."""
    ECON = "Econ"
    FIN = "Fin"
    OM = "OM"
    STAT = "Stat"


@dataclass
class PathConfig:
    """Path configuration settings."""
    dataset_path: str = "Dataset"
    output_path: str = "results"  # Changed from "result" to "results"
    model_path: str = "/path/to/local/model"
    log_path: str = "results/logs"  # Logs now under results
    config_path: str = "config"
    
    def __post_init__(self):
        """Convert string paths to Path objects and create directories."""
        self.dataset_path = Path(self.dataset_path)
        self.output_path = Path(self.output_path)
        self.model_path = Path(self.model_path)
        self.log_path = Path(self.log_path)
        self.config_path = Path(self.config_path)
        
        # Create directories if they don't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.config_path.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    """API configuration settings."""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    max_workers: int = 20
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    
    def __post_init__(self):
        """Load from environment variables if not set."""
        if not self.api_key:
            self.api_key = os.getenv("API_KEY", "")
        if not self.base_url:
            self.base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")


@dataclass
class ModelConfig:
    """Model configuration settings."""
    model_type: ModelType = ModelType.API
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens_choice: int = 512
    max_tokens_general: int = 8192
    max_new_tokens_default: int = 8192
    max_new_tokens_choice: int = 256
    
    # Local model specific settings
    dtype: str = "float16"  # float16, float32, bfloat16
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_cache: bool = True
    
    # Generation parameters
    do_sample: bool = False
    num_beams: int = 1
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.1
    
    # Anti-thinking settings
    suppress_thinking: bool = True
    thinking_keywords: List[str] = field(default_factory=lambda: [
        "thinking", "Thinking", "Let me think", "let me think"
    ])


@dataclass
class BatchConfig:
    """Batch processing configuration settings."""
    default_batch_size: int = 64
    max_retry_attempts: int = 3
    continue_on_error: bool = True
    save_intermediate_results: bool = True
    resume_from_checkpoint: bool = True
    
    # OOM handling
    enable_oom_handling: bool = True
    oom_retry_attempts: int = 3
    oom_batch_size_reduction_factor: float = 0.5
    min_batch_size: int = 1


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    evaluator_model: str = "gpt-4o"
    evaluation_timeout: int = 120
    max_evaluation_retries: int = 3
    enable_auto_evaluation: bool = True
    
    # Answer extraction settings
    enable_json_repair: bool = True
    strict_answer_extraction: bool = False
    upstream_saturation_keyword: str = "Saturation Detected"


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_to_console: bool = True
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class ProcessingConfig:
    """Processing configuration settings."""
    domains: List[str] = field(default_factory=lambda: ["Econ", "Fin", "OM", "Stat"])
    question_types: List[str] = field(default_factory=lambda: [
        "Single Choice", "Multiple Choice", "General QA", "Table QA"
    ])
    
    # File processing settings
    encoding: str = "utf-8"
    backup_encodings: List[str] = field(default_factory=lambda: ["utf-8-sig", "latin1"])
    validate_json: bool = True
    auto_repair_json: bool = True
    
    # Threading settings
    max_workers: int = 20
    thread_safe_writing: bool = True


@dataclass
class BizcompassConfig:
    """Main configuration class for Bizcompass benchmark."""
    
    # Sub-configurations
    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Global settings
    debug_mode: bool = False
    dry_run: bool = False
    verbose: bool = False
    
    # Debug settings
    debug_inference: bool = False
    debug_evaluation: bool = False
    debug_mock_responses: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Load from environment variables
        self._load_from_env()
        
        # Note: Validation is now handled separately to allow debug flags to be set first
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Path settings
        if os.getenv("BIZCOMPASS_DATASET_PATH"):
            self.paths.dataset_path = Path(os.getenv("BIZCOMPASS_DATASET_PATH"))
        if os.getenv("BIZCOMPASS_OUTPUT_PATH"):
            self.paths.output_path = Path(os.getenv("BIZCOMPASS_OUTPUT_PATH"))
        if os.getenv("BIZCOMPASS_MODEL_PATH"):
            self.paths.model_path = Path(os.getenv("BIZCOMPASS_MODEL_PATH"))
        
        # API settings
        if os.getenv("API_KEY"):
            self.api.api_key = os.getenv("API_KEY")
        if os.getenv("BASE_URL"):
            self.api.base_url = os.getenv("BASE_URL")
        if os.getenv("BIZCOMPASS_MAX_WORKERS"):
            self.api.max_workers = int(os.getenv("BIZCOMPASS_MAX_WORKERS"))
        
        # Model settings
        if os.getenv("BIZCOMPASS_MODEL_TYPE"):
            self.model.model_type = ModelType(os.getenv("BIZCOMPASS_MODEL_TYPE"))
        if os.getenv("BIZCOMPASS_MODEL_NAME"):
            self.model.model_name = os.getenv("BIZCOMPASS_MODEL_NAME")
        if os.getenv("BIZCOMPASS_TEMPERATURE"):
            self.model.temperature = float(os.getenv("BIZCOMPASS_TEMPERATURE"))
        if os.getenv("BIZCOMPASS_TOP_P"):
            self.model.top_p = float(os.getenv("BIZCOMPASS_TOP_P"))
        
        # Batch settings
        if os.getenv("LLMBENCH_BATCH_SIZE"):
            self.batch.default_batch_size = int(os.getenv("LLMBENCH_BATCH_SIZE"))
        if os.getenv("LLMBENCH_MAX_RETRIES"):
            self.batch.max_retry_attempts = int(os.getenv("LLMBENCH_MAX_RETRIES"))
        if os.getenv("LLMBENCH_MAX_NEW_TOKENS"):
            self.model.max_new_tokens_default = int(os.getenv("LLMBENCH_MAX_NEW_TOKENS"))
        if os.getenv("LLMBENCH_MAX_NEW_TOKENS_CHOICE"):
            self.model.max_new_tokens_choice = int(os.getenv("LLMBENCH_MAX_NEW_TOKENS_CHOICE"))
        if os.getenv("LLMBENCH_SUPPRESS_THINKING"):
            self.model.suppress_thinking = os.getenv("LLMBENCH_SUPPRESS_THINKING").lower() == "true"
        
        # Processing settings
        if os.getenv("BIZCOMPASS_DOMAINS"):
            self.processing.domains = [d.strip() for d in os.getenv("BIZCOMPASS_DOMAINS").split(",")]
        if os.getenv("BIZCOMPASS_QUESTION_TYPES"):
            self.processing.question_types = [q.strip() for q in os.getenv("BIZCOMPASS_QUESTION_TYPES").split(",")]
        
        # Global settings
        if os.getenv("BIZCOMPASS_DEBUG"):
            self.debug_mode = os.getenv("BIZCOMPASS_DEBUG").lower() == "true"
        if os.getenv("BIZCOMPASS_VERBOSE"):
            self.verbose = os.getenv("BIZCOMPASS_VERBOSE").lower() == "true"
    
    def _validate(self):
        """Validate configuration settings."""
        # Validate paths
        if not self.paths.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.paths.dataset_path}")
        
        # Validate API settings (skip validation in debug mode)
        if not (self.debug_mode or self.debug_inference):
            if (self.model.model_type == ModelType.API and not self.api.api_key):
                raise ValueError("API key is required for API model type")
        
        # Validate model settings
        if self.model.temperature < 0 or self.model.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.model.top_p < 0 or self.model.top_p > 1:
            raise ValueError("Top-p must be between 0 and 1")
        
        # Validate batch settings
        if self.batch.default_batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if self.batch.max_retry_attempts < 0:
            raise ValueError("Max retry attempts must be non-negative")
        
        # Validate processing settings
        if not self.processing.domains:
            raise ValueError("At least one domain must be specified")
        if not self.processing.question_types:
            raise ValueError("At least one question type must be specified")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'BizcompassConfig':
        """Load configuration from a file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BizcompassConfig':
        """Create configuration from a dictionary."""
        # Handle nested configurations
        config = cls()
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                if key in ['paths', 'api', 'model', 'batch', 'evaluation', 'logging', 'processing']:
                    # Update sub-configuration
                    sub_config = getattr(config, key)
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if hasattr(sub_config, sub_key):
                                setattr(sub_config, sub_key, sub_value)
                else:
                    # Update main configuration
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save_to_file(self, config_path: Union[str, Path], format: str = "yaml"):
        """Save configuration to a file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format.lower() in ['yaml', 'yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def update_from_args(self, args: Dict[str, Any]):
        """Update configuration from command-line arguments."""
        # Path updates
        if 'dataset_path' in args and args['dataset_path']:
            self.paths.dataset_path = Path(args['dataset_path'])
        if 'output_path' in args and args['output_path']:
            self.paths.output_path = Path(args['output_path'])
        if 'model_path' in args and args['model_path']:
            self.paths.model_path = Path(args['model_path'])
        
        # API updates
        if 'api_key' in args and args['api_key']:
            self.api.api_key = args['api_key']
        if 'base_url' in args and args['base_url']:
            self.api.base_url = args['base_url']
        
        # Model updates
        if 'model_type' in args and args['model_type']:
            self.model.model_type = ModelType(args['model_type'])
        if 'model_name' in args and args['model_name']:
            self.model.model_name = args['model_name']
        if 'temperature' in args and args['temperature'] is not None:
            self.model.temperature = args['temperature']
        if 'top_p' in args and args['top_p'] is not None:
            self.model.top_p = args['top_p']
        
        # Processing updates
        if 'domains' in args and args['domains']:
            self.processing.domains = args['domains']
        if 'question_types' in args and args['question_types']:
            self.processing.question_types = args['question_types']
        
        # Global updates
        if 'debug_mode' in args and args['debug_mode'] is not None:
            self.debug_mode = args['debug_mode']
        if 'verbose' in args and args['verbose'] is not None:
            self.verbose = args['verbose']
        if 'dry_run' in args and args['dry_run'] is not None:
            self.dry_run = args['dry_run']
    
    def get_model_config_dict(self) -> Dict[str, Any]:
        """Get model configuration as dictionary for easy access."""
        return {
            'model_type': self.model.model_type.value,
            'model_name': self.model.model_name,
            'temperature': self.model.temperature,
            'top_p': self.model.top_p,
            'max_tokens_choice': self.model.max_tokens_choice,
            'max_tokens_general': self.model.max_tokens_general,
            'max_new_tokens_default': self.model.max_new_tokens_default,
            'max_new_tokens_choice': self.model.max_new_tokens_choice,
            'dtype': self.model.dtype,
            'device_map': self.model.device_map,
            'trust_remote_code': self.model.trust_remote_code,
            'use_cache': self.model.use_cache,
            'do_sample': self.model.do_sample,
            'num_beams': self.model.num_beams,
            'early_stopping': self.model.early_stopping,
            'no_repeat_ngram_size': self.model.no_repeat_ngram_size,
            'repetition_penalty': self.model.repetition_penalty,
            'suppress_thinking': self.model.suppress_thinking,
            'thinking_keywords': self.model.thinking_keywords
        }
    
    def get_api_config_dict(self) -> Dict[str, Any]:
        """Get API configuration as dictionary for easy access."""
        return {
            'api_key': self.api.api_key,
            'base_url': self.api.base_url,
            'max_workers': self.api.max_workers,
            'request_timeout': self.api.request_timeout,
            'max_retries': self.api.max_retries,
            'retry_delay': self.api.retry_delay,
            'rate_limit_delay': self.api.rate_limit_delay
        }
    
    def get_batch_config_dict(self) -> Dict[str, Any]:
        """Get batch configuration as dictionary for easy access."""
        return {
            'default_batch_size': self.batch.default_batch_size,
            'max_retry_attempts': self.batch.max_retry_attempts,
            'continue_on_error': self.batch.continue_on_error,
            'save_intermediate_results': self.batch.save_intermediate_results,
            'resume_from_checkpoint': self.batch.resume_from_checkpoint,
            'enable_oom_handling': self.batch.enable_oom_handling,
            'oom_retry_attempts': self.batch.oom_retry_attempts,
            'oom_batch_size_reduction_factor': self.batch.oom_batch_size_reduction_factor,
            'min_batch_size': self.batch.min_batch_size
        }


def create_default_config(debug_mode: bool = False, debug_inference: bool = False, debug_evaluation: bool = False) -> BizcompassConfig:
    """Create a default configuration instance."""
    config = BizcompassConfig()
    # Set debug flags before any validation
    config.debug_mode = debug_mode
    config.debug_inference = debug_inference
    config.debug_evaluation = debug_evaluation
    # Now validate with debug flags set
    config._validate()
    return config


def load_config(config_path: Optional[Union[str, Path]] = None) -> BizcompassConfig:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        return BizcompassConfig.from_file(config_path)
    else:
        return create_default_config()


# Global configuration instance
_config: Optional[BizcompassConfig] = None


def get_config() -> BizcompassConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = create_default_config()
    return _config


def set_config(config: BizcompassConfig):
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config():
    """Reset the global configuration to default."""
    global _config
    _config = None
