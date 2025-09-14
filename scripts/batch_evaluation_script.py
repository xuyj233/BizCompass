#!/usr/bin/env python3
"""
Batch Evaluation Script

Used for batch evaluation of all QA files in specified directories
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Add the parent directory to Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from bizcompass import BizcompassBenchmark
from config import create_default_config
from evaluation.evaluator import Evaluator


class BatchEvaluator:
    """Batch evaluator for processing multiple evaluation files"""
    
    def __init__(self, config=None):
        """Initialize batch evaluator"""
        if config is None:
            # Create a minimal config for batch evaluation
            try:
                self.config = create_default_config(debug_evaluation=False)
            except ValueError:
                # If validation fails, create a minimal config manually
                from config import BizcompassConfig
                self.config = BizcompassConfig()
                self.config.debug_evaluation = False
                self.config.debug_mode = False
        else:
            self.config = config
        self.benchmark = BizcompassBenchmark(self.config)
    
    def find_all_evaluation_files(self, base_path: Path, pattern: str = "questions_eval.json") -> List[Path]:
        """Find all evaluation files in specified directory"""
        evaluation_files = []
        
        if not base_path.exists():
            print(f"❌ Path does not exist: {base_path}")
            return evaluation_files
        
        # Recursively find all matching files
        for file_path in base_path.rglob(pattern):
            evaluation_files.append(file_path)
        
        print(f"📁 Found {len(evaluation_files)} evaluation files in {base_path}")
        return evaluation_files
    
    def extract_model_info_from_path(self, file_path: Path) -> Dict[str, str]:
        """Extract model information from file path"""
        parts = file_path.parts
        
        # Find domain
        domain = None
        for part in parts:
            if part in ["Econ", "Fin", "OM", "Stat"]:
                domain = part
                break
        
        # Find question type
        question_type = None
        for part in parts:
            if part.lower() in ["singlechoice", "multiplechoice", "generalqa", "tableqa"]:
                question_type = part
                break
        
        # Find model name
        model_name = None
        for i, part in enumerate(parts):
            if part.startswith("tem") and i > 0:
                model_name = parts[i-1]
                break
        
        return {
            "domain": domain,
            "question_type": question_type,
            "model_name": model_name,
            "file_path": file_path
        }
    
    def evaluate_single_file(self, file_info: Dict[str, str], evaluator_model: str = "gpt-4o") -> Dict[str, Any]:
        """Evaluate a single file"""
        file_path = file_info["file_path"]
        domain = file_info["domain"]
        question_type = file_info["question_type"]
        model_name = file_info["model_name"]
        
        # Safely display relative path
        try:
            relative_path = file_path.relative_to(Path.cwd())
        except ValueError:
            # If not in current working directory, display absolute path
            relative_path = file_path
        print(f"🔄 Evaluating: {relative_path}")
        
        try:
            # Create evaluator
            evaluator = Evaluator(
                api_key=self.config.api.api_key,
                base_url=self.config.api.base_url,
                model_name=evaluator_model,
                verbose=False,
                debug_mode=self.config.debug_evaluation
            )
            
            # Initialize the evaluator (important!)
            evaluator.initialize()
            
            # Execute evaluation
            results = evaluator.evaluate_from_jsonl(file_path, question_type=question_type)
            
            return {
                "file_path": str(file_path),
                "domain": domain,
                "question_type": question_type,
                "model_name": model_name,
                "status": "success",
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Evaluation failed: {file_path} - {str(e)}")
            return {
                "file_path": str(file_path),
                "domain": domain,
                "question_type": question_type,
                "model_name": model_name,
                "status": "error",
                "error": str(e)
            }
    
    def batch_evaluate_directory(self, base_path: Path, evaluator_model: str = "gpt-4o", 
                                max_workers: int = 2, pattern: str = "questions_eval.json") -> Dict[str, Any]:
        """Batch evaluate all files in directory"""
        
        print(f"🚀 Starting batch evaluation for directory: {base_path}")
        print(f"   Evaluator model: {evaluator_model}")
        print(f"   Max workers: {max_workers}")
        print(f"   File pattern: {pattern}")
        print("=" * 60)
        
        # Find all evaluation files
        evaluation_files = self.find_all_evaluation_files(base_path, pattern)
        
        if not evaluation_files:
            return {"error": "No evaluation files found"}
        
        # Extract file information
        file_infos = []
        for file_path in evaluation_files:
            file_info = self.extract_model_info_from_path(file_path)
            file_infos.append(file_info)
        
        # Batch evaluation
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_info = {
                executor.submit(self.evaluate_single_file, file_info, evaluator_model): file_info
                for file_info in file_infos
            }
            
            # Collect results
            for future in as_completed(future_to_info):
                result = future.result()
                results.append(result)
                
                if result["status"] == "success":
                    print(f"✅ Completed: {Path(result['file_path']).name}")
                else:
                    print(f"❌ Failed: {Path(result['file_path']).name}")
        
        # Statistics
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        
        print("\n" + "=" * 60)
        print(f"📊 Batch evaluation completed!")
        print(f"   Total files: {len(results)}")
        print(f"   Success: {success_count}")
        print(f"   Failed: {error_count}")
        
        return {
            "total_files": len(results),
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        }
    
    def evaluate_by_domain_and_type(self, base_path: Path, domains: List[str] = None, 
                                   question_types: List[str] = None, 
                                   evaluator_model: str = "gpt-4o") -> Dict[str, Any]:
        """Evaluate by domain and question type filtering"""
        
        if domains is None:
            domains = ["Econ", "Fin", "OM", "Stat"]
        if question_types is None:
            question_types = ["Single Choice", "Multiple Choice", "General QA", "Table QA"]
        
        print(f"🎯 Filtered evaluation by criteria:")
        print(f"   Domains: {domains}")
        print(f"   Question types: {question_types}")
        print(f"   Evaluator model: {evaluator_model}")
        print("=" * 60)
        
        # Find all evaluation files
        evaluation_files = self.find_all_evaluation_files(base_path)
        
        # Filter files
        filtered_files = []
        for file_path in evaluation_files:
            file_info = self.extract_model_info_from_path(file_path)
            
            # Check domain
            if file_info["domain"] not in domains:
                continue
            
            # Check question type
            if file_info["question_type"] and file_info["question_type"].lower() not in [qt.lower().replace(" ", "") for qt in question_types]:
                continue
            
            filtered_files.append(file_info)
        
        print(f"📁 Found {len(filtered_files)} files after filtering")
        
        if not filtered_files:
            return {"error": "No files match the criteria"}
        
        # Batch evaluation
        results = []
        for file_info in filtered_files:
            result = self.evaluate_single_file(file_info, evaluator_model)
            results.append(result)
        
        # Statistics
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        
        print(f"\n📊 Filtered evaluation completed!")
        print(f"   Total files: {len(results)}")
        print(f"   Success: {success_count}")
        print(f"   Failed: {error_count}")
        
        return {
            "total_files": len(results),
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch evaluation script")
    parser.add_argument("--base_path", required=True, help="Base path for evaluation")
    parser.add_argument("--evaluator_model", default="gpt-5-nano", help="Evaluator model")
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum number of workers")
    parser.add_argument("--domains", help="Comma-separated list of domains")
    parser.add_argument("--question_types", help="Comma-separated list of question types")
    parser.add_argument("--pattern", default="questions_eval.json", help="File matching pattern")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Create configuration
    try:
        config = create_default_config(debug_evaluation=args.debug)
    except ValueError as e:
        print(f"⚠️  Configuration validation failed: {e}")
        print("   Using minimal configuration for batch evaluation...")
        # Create minimal config with correct debug flag
        from config import BizcompassConfig
        config = BizcompassConfig()
        config.debug_evaluation = args.debug
        config.debug_mode = args.debug
    
    # Create batch evaluator
    batch_evaluator = BatchEvaluator(config)
    
    # Parse arguments
    base_path = Path(args.base_path)
    # If relative path, ensure it's relative to current working directory
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    
    domains = args.domains.split(",") if args.domains else None
    question_types = args.question_types.split(",") if args.question_types else None
    
    try:
        if domains or question_types:
            # Filtered evaluation by criteria
            results = batch_evaluator.evaluate_by_domain_and_type(
                base_path=base_path,
                domains=domains,
                question_types=question_types,
                evaluator_model=args.evaluator_model
            )
        else:
            # Batch evaluate all files
            results = batch_evaluator.batch_evaluate_directory(
                base_path=base_path,
                evaluator_model=args.evaluator_model,
                max_workers=args.max_workers,
                pattern=args.pattern
            )
        
        # Save results
        output_file = base_path / "batch_evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
