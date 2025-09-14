"""
Evaluator Module

This module handles LLM-based evaluation of model responses for the Bizcompass benchmark.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI
from prompts.prompt_manager import PromptManager


class Evaluator:
    """LLM-based evaluator for the Bizcompass benchmark."""
    
    def __init__(self, api_key: str, base_url: str, model_name: str = "gpt-4o"):
        """
        Initialize the evaluator.
        
        Args:
            api_key: API key for the evaluation LLM
            base_url: Base URL for the API
            model_name: Name of the evaluation model
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        self.prompt_manager = PromptManager()
        
    def initialize(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def evaluate_single_response(self, question: str, gold_answer: str, 
                               candidate_answer: str, qid: str) -> Dict[str, Any]:
        """
        Evaluate a single response using LLM-based grading.
        
        Args:
            question: The question text
            gold_answer: The correct answer
            candidate_answer: The model's answer to evaluate
            qid: Question ID
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.client is None:
            self.initialize()
        
        system_prompt = self.prompt_manager.get_evaluation_prompt()
        user_template = self.prompt_manager.get_evaluation_user_template()
        
        user_prompt = user_template.format(
            question=question,
            gold_answer=gold_answer,
            candidate_answer=candidate_answer,
            qid=qid
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=512
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                return {
                    "qid": qid,
                    "category": result.get("category", "UNKNOWN"),
                    "explanation": result.get("explanation", ""),
                    "raw_response": result_text,
                    "error": None
                }
            except json.JSONDecodeError:
                return {
                    "qid": qid,
                    "category": "PARSING_ERROR",
                    "explanation": "Failed to parse evaluation response",
                    "raw_response": result_text,
                    "error": "JSON parsing failed"
                }
                
        except Exception as e:
            return {
                "qid": qid,
                "category": "EVALUATION_ERROR",
                "explanation": f"Evaluation failed: {str(e)}",
                "raw_response": "",
                "error": str(e)
            }
    
    def evaluate_batch(self, evaluation_data: List[Dict[str, Any]], 
                      max_workers: int = 10) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of responses.
        
        Args:
            evaluation_data: List of dictionaries containing question, gold_answer, candidate_answer, qid
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of evaluation results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_data = {
                executor.submit(
                    self.evaluate_single_response,
                    data["question"],
                    data["gold_answer"],
                    data["candidate_answer"],
                    data["qid"]
                ): data for data in evaluation_data
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_data), 
                             total=len(evaluation_data), 
                             desc="Evaluating responses"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    data = future_to_data[future]
                    results.append({
                        "qid": data["qid"],
                        "category": "EVALUATION_ERROR",
                        "explanation": f"Evaluation failed: {str(e)}",
                        "raw_response": "",
                        "error": str(e)
                    })
        
        return results
    
    def evaluate_from_jsonl(self, jsonl_path: Path, output_path: Path = None) -> Dict[str, Any]:
        """
        Evaluate responses from a JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file containing evaluation data
            output_path: Path to save evaluation results (optional)
            
        Returns:
            Dictionary containing evaluation summary
        """
        # Load evaluation data
        evaluation_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                
                # Extract evaluation data
                question = record.get("Question", record.get("question", ""))
                gold_answer = record.get("Answer", record.get("gold_answer", ""))
                
                model_result = record.get("model_evaluation_result", {})
                candidate_answer = model_result.get("model_answer", "")
                qid = record.get("ID", record.get("qid", ""))
                
                if question and gold_answer and candidate_answer and qid:
                    evaluation_data.append({
                        "question": question,
                        "gold_answer": gold_answer,
                        "candidate_answer": candidate_answer,
                        "qid": str(qid)
                    })
        
        if not evaluation_data:
            return {"error": "No valid evaluation data found"}
        
        # Evaluate responses
        results = self.evaluate_batch(evaluation_data)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Save results if output path provided
        if output_path:
            self._save_results(results, summary, output_path)
        
        return summary
    
    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results."""
        total = len(results)
        categories = {}
        errors = 0
        
        for result in results:
            category = result.get("category", "UNKNOWN")
            categories[category] = categories.get(category, 0) + 1
            
            if result.get("error"):
                errors += 1
        
        # Calculate accuracy (CORRECT + CORRECT_BUT_REASONING_MISMATCH)
        correct_count = categories.get("CORRECT", 0) + categories.get("CORRECT_BUT_REASONING_MISMATCH", 0)
        accuracy = correct_count / total if total > 0 else 0
        
        return {
            "total_questions": total,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "category_distribution": categories,
            "error_count": errors,
            "error_rate": errors / total if total > 0 else 0
        }
    
    def _save_results(self, results: List[Dict[str, Any]], 
                     summary: Dict[str, Any], output_path: Path):
        """Save evaluation results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        detailed_path = output_path.parent / f"{output_path.stem}_detailed.jsonl"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Save summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to:")
        print(f"  Detailed: {detailed_path}")
        print(f"  Summary: {summary_path}")
