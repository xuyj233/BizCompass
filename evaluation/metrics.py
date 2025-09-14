"""
Metrics Calculator

This module provides metrics calculation functionality for the Bizcompass benchmark.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import pandas as pd


class MetricsCalculator:
    """Calculator for various metrics in the Bizcompass benchmark."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_accuracy(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall accuracy from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if not results:
            return 0.0
        
        correct_count = 0
        for result in results:
            category = result.get("category", "")
            if category in ["CORRECT", "CORRECT_BUT_REASONING_MISMATCH"]:
                correct_count += 1
        
        return correct_count / len(results)
    
    def calculate_domain_metrics(self, results: List[Dict[str, Any]], 
                               domain_mapping: Dict[str, str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics by domain.
        
        Args:
            results: List of evaluation results
            domain_mapping: Optional mapping from question ID to domain
            
        Returns:
            Dictionary with metrics by domain
        """
        domain_results = defaultdict(list)
        
        # Group results by domain
        for result in results:
            qid = result.get("qid", "")
            domain = self._extract_domain_from_qid(qid, domain_mapping)
            domain_results[domain].append(result)
        
        # Calculate metrics for each domain
        domain_metrics = {}
        for domain, domain_data in domain_results.items():
            domain_metrics[domain] = {
                "total_questions": len(domain_data),
                "accuracy": self.calculate_accuracy(domain_data),
                "category_distribution": self._get_category_distribution(domain_data),
                "error_rate": self._calculate_error_rate(domain_data)
            }
        
        return domain_metrics
    
    def calculate_question_type_metrics(self, results: List[Dict[str, Any]], 
                                      type_mapping: Dict[str, str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics by question type.
        
        Args:
            results: List of evaluation results
            type_mapping: Optional mapping from question ID to question type
            
        Returns:
            Dictionary with metrics by question type
        """
        type_results = defaultdict(list)
        
        # Group results by question type
        for result in results:
            qid = result.get("qid", "")
            question_type = self._extract_question_type_from_qid(qid, type_mapping)
            type_results[question_type].append(result)
        
        # Calculate metrics for each question type
        type_metrics = {}
        for question_type, type_data in type_results.items():
            type_metrics[question_type] = {
                "total_questions": len(type_data),
                "accuracy": self.calculate_accuracy(type_data),
                "category_distribution": self._get_category_distribution(type_data),
                "error_rate": self._calculate_error_rate(type_data)
            }
        
        return type_metrics
    
    def calculate_model_comparison(self, model_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Calculate comparison metrics across different models.
        
        Args:
            model_results: Dictionary mapping model names to their results
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        for model_name, results in model_results.items():
            comparison[model_name] = {
                "total_questions": len(results),
                "accuracy": self.calculate_accuracy(results),
                "category_distribution": self._get_category_distribution(results),
                "error_rate": self._calculate_error_rate(results)
            }
        
        # Add ranking
        model_accuracies = [(model, data["accuracy"]) for model, data in comparison.items()]
        model_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        comparison["ranking"] = [model for model, _ in model_accuracies]
        comparison["best_model"] = model_accuracies[0][0] if model_accuracies else None
        
        return comparison
    
    def generate_report(self, results: List[Dict[str, Any]], 
                       output_path: Path = None) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: List of evaluation results
            output_path: Path to save the report (optional)
            
        Returns:
            Dictionary containing the full report
        """
        report = {
            "overall_metrics": {
                "total_questions": len(results),
                "accuracy": self.calculate_accuracy(results),
                "category_distribution": self._get_category_distribution(results),
                "error_rate": self._calculate_error_rate(results)
            },
            "domain_metrics": self.calculate_domain_metrics(results),
            "question_type_metrics": self.calculate_question_type_metrics(results),
            "detailed_results": results
        }
        
        # Add summary statistics
        report["summary"] = self._generate_summary(report)
        
        # Save report if output path provided
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _extract_domain_from_qid(self, qid: str, domain_mapping: Dict[str, str] = None) -> str:
        """Extract domain from question ID."""
        if domain_mapping and qid in domain_mapping:
            return domain_mapping[qid]
        
        # Default extraction logic (can be customized)
        return "unknown"
    
    def _extract_question_type_from_qid(self, qid: str, type_mapping: Dict[str, str] = None) -> str:
        """Extract question type from question ID."""
        if type_mapping and qid in type_mapping:
            return type_mapping[qid]
        
        # Default extraction logic (can be customized)
        return "unknown"
    
    def _get_category_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of evaluation categories."""
        categories = [result.get("category", "UNKNOWN") for result in results]
        return dict(Counter(categories))
    
    def _calculate_error_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate error rate from results."""
        if not results:
            return 0.0
        
        error_count = sum(1 for result in results if result.get("error"))
        return error_count / len(results)
    
    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the report."""
        overall = report["overall_metrics"]
        
        summary = {
            "total_questions_evaluated": overall["total_questions"],
            "overall_accuracy": overall["accuracy"],
            "total_errors": int(overall["error_rate"] * overall["total_questions"]),
            "most_common_category": max(overall["category_distribution"].items(), key=lambda x: x[1])[0] if overall["category_distribution"] else "N/A",
            "domains_evaluated": list(report["domain_metrics"].keys()),
            "question_types_evaluated": list(report["question_type_metrics"].keys())
        }
        
        return summary
    
    def _save_report(self, report: Dict[str, Any], output_path: Path):
        """Save the evaluation report to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also save as CSV for easy analysis
        csv_path = output_path.parent / f"{output_path.stem}.csv"
        self._save_results_csv(report["detailed_results"], csv_path)
        
        print(f"Evaluation report saved to:")
        print(f"  JSON: {output_path}")
        print(f"  CSV: {csv_path}")
    
    def _save_results_csv(self, results: List[Dict[str, Any]], csv_path: Path):
        """Save results as CSV for analysis."""
        if not results:
            return
        
        # Convert to DataFrame
        df_data = []
        for result in results:
            df_data.append({
                "qid": result.get("qid", ""),
                "category": result.get("category", ""),
                "explanation": result.get("explanation", ""),
                "has_error": bool(result.get("error")),
                "error_message": result.get("error", "")
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
