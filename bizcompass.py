#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bizcompass Benchmark

A comprehensive benchmark for evaluating Large Language Models (LLMs) in business contexts.
This is the main entry point for the modularized Bizcompass system.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import modular components
from inference import APIInference, LocalInference, DebugInference
from evaluation import Evaluator, MetricsCalculator
from dataloader.dataloader import load_questions
from config import BizcompassConfig, create_default_config, load_config
from experiment_recorder import get_experiment_recorder


class BizcompassBenchmark:
    """Main benchmark class for the Bizcompass system."""
    
    def __init__(self, config: BizcompassConfig):
        """
        Initialize the Bizcompass benchmark.
        
        Args:
            config: BizcompassConfig instance
        """
        self.config = config
        self.dataset_path = config.paths.dataset_path
        self.output_path = config.paths.output_path
        
    def run_inference(self, model_name: str = None, model_type: str = None, 
                     domains: List[str] = None, question_types: List[str] = None,
                     temperature: float = None, top_p: float = None) -> Dict[str, Any]:
        """
        Run inference for a specific model.
        
        Args:
            model_name: Name of the model to use (uses config default if None)
            model_type: Type of model ('api' or 'local') (uses config default if None)
            domains: List of domains to process (uses config default if None)
            question_types: List of question types to process (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            top_p: Top-p sampling parameter (uses config default if None)
            
        Returns:
            Dictionary containing inference results
        """
        # Use config defaults if not provided
        model_name = model_name or self.config.model.model_name
        model_type = model_type or self.config.model.model_type.value
        domains = domains or self.config.processing.domains
        question_types = question_types or self.config.processing.question_types
        temperature = temperature if temperature is not None else self.config.model.temperature
        top_p = top_p if top_p is not None else self.config.model.top_p
        
        print(f"🚀 Running inference for {model_name} ({model_type})")
        print(f"   Domains: {domains}")
        print(f"   Question types: {question_types}")
        print(f"   Temperature: {temperature}, Top-p: {top_p}")
        
        # Initialize inference engine
        if self.config.debug_mode or self.config.debug_inference:
            inference_engine = DebugInference(self.config.to_dict())
        elif model_type == "api":
            inference_engine = APIInference(
                api_key=self.config.api.api_key,
                base_url=self.config.api.base_url
            )
        elif model_type == "local":
            inference_engine = LocalInference(
                model_path=self.config.paths.model_path
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Run inference
        results = inference_engine.process_directory(
            dataset_path=self.dataset_path,
            output_path=self.output_path,
            model_name=model_name,
            question_types=question_types,
            domains=domains,
            temperature=temperature,
            top_p=top_p,
            continue_on_error=self.config.batch.continue_on_error
        )
        
        # Record the experiment (skip in debug mode to avoid cluttering)
        if not (self.config.debug_mode or self.config.debug_inference):
            recorder = get_experiment_recorder()
            experiment_id = recorder.record_inference_experiment(
                config=self.config,
                model_name=model_name,
                model_type=model_type,
                domains=domains,
                question_types=question_types,
                temperature=temperature,
                top_p=top_p,
                results=results
            )
            print(f"📝 Experiment recorded with ID: {experiment_id}")
        
        return results
    
    def run_evaluation(self, model_name: str, evaluator_model: str = "gpt-4o",
                      domains: List[str] = None, question_types: List[str] = None) -> Dict[str, Any]:
        """
        Run evaluation for a specific model's results.
        
        Args:
            model_name: Name of the model to evaluate
            evaluator_model: Model to use for evaluation
            domains: List of domains to evaluate
            question_types: List of question types to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        if domains is None:
            domains = ["Econ", "Fin", "OM", "Stat"]
        if question_types is None:
            question_types = ["Single Choice", "Multiple Choice", "General QA", "Table QA"]
        
        print(f"📊 Running evaluation for {model_name}")
        print(f"   Evaluator model: {evaluator_model}")
        print(f"   Domains: {domains}")
        print(f"   Question types: {question_types}")
        
        # Initialize evaluator
        evaluator = Evaluator(
            api_key=self.config.api.api_key,
            base_url=self.config.api.base_url,
            model_name=evaluator_model
        )
        
        # Find evaluation files
        evaluation_files = []
        for domain in domains:
            for question_type in question_types:
                eval_dir = self.output_path / domain / question_type.replace(" ", "").lower() / model_name
                if eval_dir.exists():
                    for eval_file in eval_dir.rglob("questions_eval.jsonl"):
                        evaluation_files.append(eval_file)
        
        if not evaluation_files:
            print("⚠️  No evaluation files found")
            return {"error": "No evaluation files found"}
        
        print(f"   Found {len(evaluation_files)} evaluation files")
        
        # Run evaluation
        all_results = []
        for eval_file in evaluation_files:
            print(f"   Evaluating: {eval_file.relative_to(self.output_path)}")
            results = evaluator.evaluate_from_jsonl(eval_file)
            if "detailed_results" in results:
                all_results.extend(results["detailed_results"])
        
        # Calculate metrics
        metrics_calculator = MetricsCalculator()
        report = metrics_calculator.generate_report(all_results)
        
        # Record the evaluation experiment
        recorder = get_experiment_recorder()
        experiment_id = recorder.record_evaluation_experiment(
            config=self.config,
            model_name=model_name,
            evaluator_model=evaluator_model,
            domains=domains,
            question_types=question_types,
            results=report
        )
        print(f"📝 Evaluation experiment recorded with ID: {experiment_id}")
        
        return report
    
    def run_full_pipeline(self, model_name: str, model_type: str = "api",
                         domains: List[str] = None, question_types: List[str] = None,
                         temperature: float = 0.0, top_p: float = 0.95,
                         run_evaluation: bool = True) -> Dict[str, Any]:
        """
        Run the full pipeline: inference + evaluation.
        
        Args:
            model_name: Name of the model to use
            model_type: Type of model ('api' or 'local')
            domains: List of domains to process
            question_types: List of question types to process
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            run_evaluation: Whether to run evaluation after inference
            
        Returns:
            Dictionary containing full pipeline results
        """
        print(f"🎯 Running full pipeline for {model_name}")
        
        # Run inference
        inference_results = self.run_inference(
            model_name=model_name,
            model_type=model_type,
            domains=domains,
            question_types=question_types,
            temperature=temperature,
            top_p=top_p
        )
        
        pipeline_results = {
            "inference": inference_results,
            "evaluation": None
        }
        
        # Run evaluation if requested
        if run_evaluation:
            evaluation_results = self.run_evaluation(
                model_name=model_name,
                domains=domains,
                question_types=question_types
            )
            pipeline_results["evaluation"] = evaluation_results
        
        return pipeline_results


def create_default_config(debug_mode: bool = False, debug_inference: bool = False, debug_evaluation: bool = False) -> BizcompassConfig:
    """Create default configuration."""
    from config import create_default_config as create_config
    config = create_config(debug_mode, debug_inference, debug_evaluation)
    return config


def main():
    """Main function for the Bizcompass benchmark."""
    parser = argparse.ArgumentParser(description="Bizcompass Benchmark - LLM Evaluation System")
    
    # Main command
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference only")
    inference_parser.add_argument("--model_name", required=True, help="Name of the model to use")
    inference_parser.add_argument("--model_type", choices=["api", "local"], default="api", help="Model type")
    inference_parser.add_argument("--domains", type=str, help="Comma-separated list of domains")
    inference_parser.add_argument("--question_types", type=str, help="Comma-separated list of question types")
    inference_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    inference_parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    
    # Evaluation command
    evaluation_parser = subparsers.add_parser("evaluation", help="Run evaluation only")
    evaluation_parser.add_argument("--model_name", required=True, help="Name of the model to evaluate")
    evaluation_parser.add_argument("--evaluator_model", default="gpt-4o", help="Model to use for evaluation")
    evaluation_parser.add_argument("--domains", type=str, help="Comma-separated list of domains")
    evaluation_parser.add_argument("--question_types", type=str, help="Comma-separated list of question types")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline (inference + evaluation)")
    pipeline_parser.add_argument("--model_name", required=True, help="Name of the model to use")
    pipeline_parser.add_argument("--model_type", choices=["api", "local"], default="api", help="Model type")
    pipeline_parser.add_argument("--domains", type=str, help="Comma-separated list of domains")
    pipeline_parser.add_argument("--question_types", type=str, help="Comma-separated list of question types")
    pipeline_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    pipeline_parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    pipeline_parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation step")
    
    # Common arguments
    for subparser in [inference_parser, evaluation_parser, pipeline_parser]:
        subparser.add_argument("--dataset_path", type=str, help="Path to dataset directory")
        subparser.add_argument("--output_path", type=str, help="Path to output directory")
        subparser.add_argument("--api_key", type=str, help="API key for API models")
        subparser.add_argument("--base_url", type=str, help="Base URL for API")
        subparser.add_argument("--model_path", type=str, help="Path to local model")
        subparser.add_argument("--debug", action="store_true", help="Enable debug mode with mock responses")
        subparser.add_argument("--debug_inference", action="store_true", help="Enable debug mode for inference only")
        subparser.add_argument("--debug_evaluation", action="store_true", help="Enable debug mode for evaluation only")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle debug flags first (before config creation)
    debug_mode = args.debug
    debug_inference = args.debug_inference or args.debug
    debug_evaluation = args.debug_evaluation or args.debug
    
    # Load configuration with debug flags
    config = create_default_config(debug_mode, debug_inference, debug_evaluation)
    
    # Override with command line arguments
    args_dict = vars(args)
    config.update_from_args(args_dict)
    
    # Parse list arguments
    domains = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
    
    question_types = None
    if args.question_types:
        question_types = [q.strip() for q in args.question_types.split(",")]
    
    # Initialize benchmark
    benchmark = BizcompassBenchmark(config)
    
    try:
        if args.command == "inference":
            results = benchmark.run_inference(
                model_name=args.model_name,
                model_type=args.model_type,
                domains=domains,
                question_types=question_types,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(f"\n✅ Inference completed successfully!")
            
        elif args.command == "evaluation":
            results = benchmark.run_evaluation(
                model_name=args.model_name,
                evaluator_model=args.evaluator_model,
                domains=domains,
                question_types=question_types
            )
            print(f"\n✅ Evaluation completed successfully!")
            
        elif args.command == "pipeline":
            results = benchmark.run_full_pipeline(
                model_name=args.model_name,
                model_type=args.model_type,
                domains=domains,
                question_types=question_types,
                temperature=args.temperature,
                top_p=args.top_p,
                run_evaluation=not args.skip_evaluation
            )
            print(f"\n✅ Full pipeline completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Command failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
