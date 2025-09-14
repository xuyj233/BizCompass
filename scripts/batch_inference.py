#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Inference Script

This script provides batch processing capabilities for the Bizcompass benchmark.
It supports both API and local model inference with configurable parameters.
"""

import os
import time
import shlex
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Import the modular components
from inference import APIInference, LocalInference
from experiment_recorder import get_experiment_recorder


class BatchInferenceRunner:
    """Batch inference runner for the Bizcompass benchmark."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the batch inference runner.
        
        Args:
            config: Configuration dictionary containing paths, models, and parameters
        """
        self.config = config
        self.dataset_path = Path(config["dataset_path"])
        self.output_path = Path(config["output_path"])
        self.model_type = config["model_type"]
        
    def run_single_model(self, model_name: str, temperature: float, top_p: float) -> Dict[str, Any]:
        """
        Run inference for a single model with given parameters.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing inference results
        """
        print(f"\n🚀 Running inference for model: {model_name}")
        print(f"   Temperature: {temperature}, Top-p: {top_p}")
        
        if self.model_type == "api":
            inference_engine = APIInference(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"]
            )
        elif self.model_type == "local":
            inference_engine = LocalInference(
                model_path=self.config.get("model_path")
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Run inference
        results = inference_engine.process_directory(
            dataset_path=self.dataset_path,
            output_path=self.output_path,
            model_name=model_name,
            question_types=self.config["question_types"],
            domains=self.config["domains"],
            temperature=temperature,
            top_p=top_p,
            continue_on_error=self.config.get("continue_on_error", True)
        )
        
        # Record the batch experiment
        recorder = get_experiment_recorder()
        experiment_id = recorder.record_inference_experiment(
            config=self.config,  # Note: This needs to be a BizcompassConfig object
            model_name=model_name,
            model_type=self.model_type,
            domains=self.config["domains"],
            question_types=self.config["question_types"],
            temperature=temperature,
            top_p=top_p,
            results=results
        )
        
        print(f"📝 Batch experiment recorded with ID: {experiment_id}")
        
        return results
    
    def run_batch(self, models: List[str], temperatures: List[float], 
                  top_ps: List[float]) -> Dict[str, Any]:
        """
        Run batch inference across multiple models and parameters.
        
        Args:
            models: List of model names
            temperatures: List of temperature values
            top_ps: List of top-p values
            
        Returns:
            Dictionary containing batch results
        """
        print(f"\n🎯 Starting batch inference...")
        print(f"   Models: {models}")
        print(f"   Temperatures: {temperatures}")
        print(f"   Top-p values: {top_ps}")
        print(f"   Domains: {self.config['domains']}")
        print(f"   Question types: {self.config['question_types']}")
        
        batch_results = {
            "start_time": time.time(),
            "total_tasks": len(models) * len(temperatures) * len(top_ps),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "results": {}
        }
        
        task_id = 0
        for model in models:
            for temp in temperatures:
                for top_p in top_ps:
                    task_id += 1
                    task_name = f"{model}_t{temp}_p{top_p}"
                    
                    print(f"\n[{task_id}/{batch_results['total_tasks']}] Processing: {task_name}")
                    
                    try:
                        result = self.run_single_model(model, temp, top_p)
                        batch_results["results"][task_name] = result
                        batch_results["completed_tasks"] += 1
                        
                        print(f"✅ Completed: {task_name}")
                        print(f"   Processed: {result['total_processed']}")
                        print(f"   Skipped: {result['total_skipped']}")
                        if result["errors"]:
                            print(f"   Errors: {len(result['errors'])}")
                        
                    except Exception as e:
                        print(f"❌ Failed: {task_name} - {str(e)}")
                        batch_results["results"][task_name] = {"error": str(e)}
                        batch_results["failed_tasks"] += 1
        
        batch_results["end_time"] = time.time()
        batch_results["duration"] = batch_results["end_time"] - batch_results["start_time"]
        
        # Print summary
        self._print_batch_summary(batch_results)
        
        return batch_results
    
    def _print_batch_summary(self, results: Dict[str, Any]):
        """Print batch processing summary."""
        print(f"\n{'='*60}")
        print(f"🏁 Batch processing completed")
        print(f"{'='*60}")
        print(f"⏱️  Total time: {results['duration']/60:.1f} minutes ({results['duration']:.1f} seconds)")
        print(f"📊 Total tasks: {results['total_tasks']}")
        print(f"✅ Completed: {results['completed_tasks']}")
        print(f"❌ Failed: {results['failed_tasks']}")
        print(f"📈 Success rate: {results['completed_tasks']/results['total_tasks']*100:.1f}%")
        
        if results["failed_tasks"] > 0:
            print(f"\n⚠️  Failed tasks:")
            for task_name, result in results["results"].items():
                if "error" in result:
                    print(f"   - {task_name}: {result['error']}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for batch inference."""
    return {
        "dataset_path": "Dataset",
        "output_path": "result",
        "model_type": "api",
        "api_key": os.getenv("API_KEY", "your-api-key-here"),
        "base_url": os.getenv("BASE_URL", "https://api.openai.com/v1"),
        "model_path": os.getenv("MODEL_PATH", "/path/to/local/model"),
        "domains": ["Econ", "Fin", "OM", "Stat"],
        "question_types": ["Single Choice", "Multiple Choice", "General QA", "Table QA"],
        "continue_on_error": True
    }


def main():
    """Main function for batch inference script."""
    parser = argparse.ArgumentParser(description="Bizcompass Batch Inference Script")
    
    # Configuration arguments
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset directory")
    parser.add_argument("--output_path", type=str, help="Path to output directory")
    parser.add_argument("--model_type", choices=["api", "local"], default="api", help="Model type")
    
    # API configuration
    parser.add_argument("--api_key", type=str, help="API key for API models")
    parser.add_argument("--base_url", type=str, help="Base URL for API")
    parser.add_argument("--model_path", type=str, help="Path to local model")
    
    # Model and parameter selection
    parser.add_argument("--models", type=str, help="Comma-separated list of models")
    parser.add_argument("--domains", type=str, help="Comma-separated list of domains")
    parser.add_argument("--question_types", type=str, help="Comma-separated list of question types")
    parser.add_argument("--temperatures", type=str, help="Comma-separated list of temperatures")
    parser.add_argument("--top_ps", type=str, help="Comma-separated list of top-p values")
    
    # Execution options
    parser.add_argument("--continue_on_error", action="store_true", help="Continue on error")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be executed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_default_config()
    
    # Override with command line arguments
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path
    if args.output_path:
        config["output_path"] = args.output_path
    if args.model_type:
        config["model_type"] = args.model_type
    if args.api_key:
        config["api_key"] = args.api_key
    if args.base_url:
        config["base_url"] = args.base_url
    if args.model_path:
        config["model_path"] = args.model_path
    if args.continue_on_error:
        config["continue_on_error"] = True
    
    # Parse list arguments
    if args.models:
        config["models"] = [m.strip() for m in args.models.split(",")]
    else:
        config["models"] = ["gpt-4o", "claude-3-sonnet"]
    
    if args.domains:
        config["domains"] = [d.strip() for d in args.domains.split(",")]
    
    if args.question_types:
        config["question_types"] = [q.strip() for q in args.question_types.split(",")]
    
    if args.temperatures:
        temperatures = [float(t.strip()) for t in args.temperatures.split(",")]
    else:
        temperatures = [0.0, 0.8]
    
    if args.top_ps:
        top_ps = [float(p.strip()) for p in args.top_ps.split(",")]
    else:
        top_ps = [0.95]
    
    # Validate configuration
    if config["model_type"] == "api" and not config["api_key"]:
        print("❌ Error: API key is required for API model type")
        return 1
    
    if config["model_type"] == "local" and not config["model_path"]:
        print("❌ Error: Model path is required for local model type")
        return 1
    
    # Show configuration
    print("🔧 Configuration:")
    print(f"   Dataset path: {config['dataset_path']}")
    print(f"   Output path: {config['output_path']}")
    print(f"   Model type: {config['model_type']}")
    print(f"   Models: {config['models']}")
    print(f"   Domains: {config['domains']}")
    print(f"   Question types: {config['question_types']}")
    print(f"   Temperatures: {temperatures}")
    print(f"   Top-p values: {top_ps}")
    
    if args.dry_run:
        print("\n🔍 Dry run mode - no actual execution")
        return 0
    
    # Run batch inference
    try:
        runner = BatchInferenceRunner(config)
        results = runner.run_batch(config["models"], temperatures, top_ps)
        
        print(f"\n🎉 Batch inference completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Batch inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
