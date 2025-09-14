"""
Evaluator Module

This module handles LLM-based evaluation of model responses for the Bizcompass benchmark.
"""

import json
import os
from pathlib import Path
import re
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI
from prompts.prompt_manager import PromptManager


class Evaluator:
    """LLM-based evaluator for the Bizcompass benchmark."""
    
    def __init__(self, api_key: str, base_url: str, model_name: str = "gpt-4o", verbose: bool = False, debug_mode: bool = False):
        """
        Initialize the evaluator.
        
        Args:
            api_key: API key for the evaluation LLM
            base_url: Base URL for the API
            model_name: Name of the evaluation model
            verbose: Whether to print detailed input/output information
            debug_mode: Whether to use debug mode (mock responses)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.client = None
        self.prompt_manager = PromptManager()
        
    def initialize(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def evaluate_single_response(self, question: str, gold_answer: str, 
                               candidate_answer: str, qid: str, question_type: str = None, domain: str = None) -> Dict[str, Any]:
        """
        Evaluate a single response using appropriate method based on question type.
        
        Args:
            question: The question text
            gold_answer: The correct answer
            candidate_answer: The model's answer to evaluate
            qid: Question ID
            question_type: Type of question (for choosing evaluation method)
            
        Returns:
            Dictionary containing evaluation results
        """
        # For choice questions, use direct comparison
        # Check various formats of question type
        is_choice_question = (
            question_type and (
                question_type.lower() in ["single choice", "multiple choice"] or
                question_type.lower() in ["singlechoice", "multiplechoice"] or
                question_type.lower() in ["single", "multiple"]
            )
        )
        
        if is_choice_question:
            return self._evaluate_choice_question(gold_answer, candidate_answer, qid)
        
        # For open-ended questions, use LLM-based evaluation
        return self._evaluate_open_ended_question(question, gold_answer, candidate_answer, qid, domain)
    
    def _evaluate_choice_question(self, gold_answer: str, candidate_answer: str, qid: str) -> Dict[str, Any]:
        """Evaluate choice questions by direct comparison."""
        # Normalize answers for comparison
        gold_normalized = gold_answer.strip().upper()
        candidate_normalized = candidate_answer.strip().upper()
        
        # Extract letters from candidate answer (in case it's a full sentence)
        import re
        candidate_letters = re.findall(r'[A-D]', candidate_normalized)
        candidate_letter = candidate_letters[0] if candidate_letters else candidate_normalized
        
        is_correct = gold_normalized == candidate_letter
        
        return {
            "qid": qid,
            "correct": is_correct,
            "category": "CORRECT" if is_correct else "INCORRECT",
            "score": 1.0 if is_correct else 0.0
        }
    
    def _evaluate_open_ended_question(self, question: str, gold_answer: str, 
                                    candidate_answer: str, qid: str, domain: str = None) -> Dict[str, Any]:
        """Evaluate open-ended questions using LLM-based grading."""
        if self.client is None:
            self.initialize()
        
        system_prompt = self.prompt_manager.get_evaluation_prompt(domain=domain)
        user_template = self.prompt_manager.get_evaluation_user_template()
        
        user_prompt = user_template.format(
            question=question,
            gold_answer=gold_answer,
            cand=candidate_answer,
            qid=qid
        )
        
        # Prepare messages for API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Print detailed input information if verbose mode is enabled
        if self.verbose:
            self._print_evaluation_input(qid, question, gold_answer, candidate_answer, 
                                       system_prompt, user_prompt)
            # Print the actual messages being sent to the model
            self._print_messages_to_model(qid, messages)
        
        # Check if we're in debug mode
        if self.debug_mode:
            result_text = self._get_debug_evaluation_response(question, gold_answer, candidate_answer, qid, domain)
        else:
            # Retry mechanism for API calls
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # Use requests for streaming API calls
                    import requests
                    import json
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    print(messages)
                    data = {
                        'model': self.model_name,
                        'messages': messages,
                        'stream': True,
                        'temperature': 0.0,
                        'max_tokens': 512
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers=headers,
                        data=json.dumps(data),
                        stream=True
                    )
                
                    
                    if response.status_code != 200:
                        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                    
                    # Process streaming response
                    full_content = ""
                    for chunk in response.iter_lines():
                        if chunk:
                            decoded_chunk = chunk.decode('utf-8')
            
                            if decoded_chunk.startswith('data:'):
                                try:
                                    # Remove the 'data: ' prefix and parse the JSON object
                                    parsed_chunk = json.loads(decoded_chunk[5:])
                                    
                                    if 'choices' in parsed_chunk and parsed_chunk['choices']:
                                        delta = parsed_chunk['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            full_content += content
                                except json.JSONDecodeError:
                                    pass
                                except Exception as e:
                                    print(f"❌ Error parsing chunk: {e}")
                    
                    result_text = full_content.strip()
          
                    if self.verbose:
                        print(f"📥 Full response: {result_text}")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ API call failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    
                    if attempt < max_retries - 1:
                        print(f"⏳ Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"💥 All retry attempts failed for question {qid}")
                        return {
                            "qid": qid,
                            "category": "API_ERROR",
                            "score": 0.0
                        }
        
        # Print detailed output information if verbose mode is enabled
        if self.verbose:
            self._print_evaluation_output(qid, result_text)
        def extract_json_string(text):
            """
            Extracts a single JSON-style dictionary string from a text.
            
            Args:
                text (str): The input text that may contain a JSON dictionary
                
            Returns:
                str or None: The extracted dictionary string if exactly one valid 
                            JSON dictionary is found, None otherwise
            """
            # Regular expression pattern to match JSON dictionary structures
            # Handles newlines, spaces, and standard JSON key-value formatting
            pattern = r'\{\s*"[^"]+"\s*:\s*[^,]+(?:\s*,\s*"[^"]+"\s*:\s*[^,]+)*\s*\}'
            matches = re.findall(pattern, text)
            
            # Check if exactly one potential dictionary was found
            if len(matches) == 1:
                try:
                    # Validate the JSON format
                    import json
                    json.loads(matches[0])
                    return matches[0]  # Return the original string
                except json.JSONDecodeError:
                    # Return None if JSON validation fails
                    return None
            return None
        print('-------------------------')
        print(result_text)
        result_text = extract_json_string(result_text)
        print(result_text)
        print('-------------------------')
        # Try to parse JSON response
        try:
            result = json.loads(result_text)
            score = result.get("score", 0.0)
            print(score)
            # Debug: Print the parsed result if verbose
            if self.verbose:
                print(f"🔍 Parsed JSON result: {result}")
                print(f"🔍 Extracted score: {score}")
            
            # Score is already in 0-1 range from the prompt
            normalized_score = score
            
            # Determine category based on normalized score
            if normalized_score >= 0.8:
                category = "CORRECT"
            elif normalized_score >= 0.6:
                category = "PARTIALLY_CORRECT"
            else:
                category = "INCORRECT"
            
            # In debug mode, print the score prominently
            if self.verbose:
                print(f"\n🎯 SCORE: {normalized_score:.2f} - Question ID: {qid}")
                print("="*50)
            
            return {
                "qid": qid,
                "category": category,
                "score": normalized_score
            }
        except json.JSONDecodeError:
            if self.verbose:
                print(f"\n❌ JSON PARSING ERROR - Question ID: {qid}")
                print("="*50)
            return {
                "qid": qid,
                "category": "PARSING_ERROR",
                "score": 0.0
            }
        except Exception as e:
            if self.verbose:
                print(f"\n❌ EVALUATION ERROR - Question ID: {qid}")
                print(f"Error: {str(e)}")
                print("="*50)
            return {
                "qid": qid,
                "category": "EVALUATION_ERROR",
                "score": 0.0
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
                    data["qid"],
                    data.get("question_type"),
                    data.get("domain")
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
                        "score": 0.0
                    })
        
        return results
    
    def evaluate_from_jsonl(self, jsonl_path: Path, output_path: Path = None, question_type: str = None) -> Dict[str, Any]:
        """
        Evaluate responses from a JSON or JSONL file.
        
        Args:
            jsonl_path: Path to the JSON/JSONL file containing evaluation data
            output_path: Path to save evaluation results (optional)
            question_type: Type of questions being evaluated
            
        Returns:
            Dictionary containing evaluation summary
        """
        # Load evaluation data
        evaluation_data = []
        records = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Try to parse as JSON array first
            try:
                records = json.loads(content)
                if not isinstance(records, list):
                    records = [records]
            except json.JSONDecodeError:
                # If JSON parsing fails, try JSONL format
                for line in content.split('\n'):
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            records.append(record)
                        except json.JSONDecodeError:
                            continue
        
        # Process records
        for record in records:
            # Extract evaluation data
            question = record.get("Question", record.get("question", ""))
            gold_answer = record.get("Answer", record.get("gold_answer", ""))
            
            model_result = record.get("model_evaluation_result", {})
            candidate_answer = model_result.get("model_answer", "")
            qid = record.get("ID", record.get("qid", ""))
            
            if question and gold_answer and candidate_answer and qid:
                # Extract domain from file path
                domain = None
                path_parts = jsonl_path.parts
                for part in path_parts:
                    if part.lower() in ["econ", "fin", "om", "stat"]:
                        domain = part
                        break
                
                evaluation_data.append({
                    "question": question,
                    "gold_answer": gold_answer,
                    "candidate_answer": candidate_answer,
                    "qid": str(qid),
                    "question_type": question_type,
                    "domain": domain
                })
        
        if not evaluation_data:
            return {"error": "No valid evaluation data found"}
        
        # Evaluate responses
        results = self.evaluate_batch(evaluation_data)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Update original records with evaluation results
        updated_records = self._update_records_with_evaluation(records, results)
        
        # Save updated records back to original file
        self._save_updated_records(updated_records, jsonl_path)
        
        # Save results if output path provided
        if output_path:
            self._save_results(results, summary, output_path)
        
        return {
            "summary": summary,
            "detailed_results": results,
            "updated_records": updated_records
        }
    
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
    
    def _update_records_with_evaluation(self, records: List[Dict[str, Any]], 
                                      evaluation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update original records with evaluation results."""
        # Create a mapping from qid to evaluation result
        eval_map = {result["qid"]: result for result in evaluation_results}
        
        updated_records = []
        for record in records:
            qid = str(record.get("ID", record.get("qid", "")))
            if qid in eval_map:
                # Add evaluation results to the record (only essential fields)
                eval_result = eval_map[qid]
                record["evaluation_result"] = {
                    "category": eval_result.get("category"),
                    "score": eval_result.get("score")
                }
            updated_records.append(record)
        
        return updated_records
    
    def _save_updated_records(self, records: List[Dict[str, Any]], file_path: Path):
        """Save updated records back to the original file."""
        # Create backup
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        if file_path.exists():
            import shutil
            shutil.copy2(file_path, backup_path)
        
        # Save updated records
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Updated {len(records)} records in {file_path}")
        print(f"📁 Backup saved to {backup_path}")
    
    def _print_evaluation_input(self, qid: str, question: str, gold_answer: str, 
                              candidate_answer: str, system_prompt: str, user_prompt: str):
        """Print detailed input information for evaluation."""
        print("\n" + "="*80)
        print(f"🔍 EVALUATION INPUT - Question ID: {qid}")
        print("="*80)
        
        print(f"\n📝 QUESTION:")
        print("-" * 40)
        print(question[:500] + "..." if len(question) > 500 else question)
        
        print(f"\n✅ GOLD ANSWER:")
        print("-" * 40)
        print(gold_answer[:300] + "..." if len(gold_answer) > 300 else gold_answer)
        
        print(f"\n🤖 CANDIDATE ANSWER:")
        print("-" * 40)
        print(candidate_answer[:300] + "..." if len(candidate_answer) > 300 else candidate_answer)
        
        print(f"\n🎯 SYSTEM PROMPT:")
        print("-" * 40)
        print(system_prompt[:400] + "..." if len(system_prompt) > 400 else system_prompt)
        
        print(f"\n💬 USER PROMPT:")
        print("-" * 40)
        print(user_prompt[:400] + "..." if len(user_prompt) > 400 else user_prompt)
        
        print(f"\n🚀 SENDING TO MODEL: {self.model_name}")
        print("="*80)
    
    def _print_evaluation_output(self, qid: str, result_text: str):
        """Print detailed output information from evaluation."""
        print(f"\n📤 EVALUATION OUTPUT - Question ID: {qid}")
        print("-" * 60)
        print(result_text[:500] + "..." if len(result_text) > 500 else result_text)
        print("-" * 60)
    
    def _print_messages_to_model(self, qid: str, messages: List[Dict[str, str]]):
        """Print the actual messages being sent to the model."""
        print(f"\n📨 MESSAGES TO MODEL - Question ID: {qid}")
        print("=" * 60)
        for i, message in enumerate(messages):
            role = message["role"].upper()
            content = message["content"]
            print(f"\n[{i+1}] {role} MESSAGE:")
            print("-" * 40)
            # Truncate very long content for readability
            if len(content) > 800:
                print(content[:800] + "\n... [TRUNCATED] ...")
            else:
                print(content)
        print("=" * 60)
    
    def _get_debug_evaluation_response(self, question: str, gold_answer: str, 
                                     candidate_answer: str, qid: str, domain: str = None) -> str:
        """
        Generate a mock evaluation response for debug mode.
        
        Args:
            question: The question text
            gold_answer: The correct answer
            candidate_answer: The model's answer to evaluate
            qid: Question ID
            domain: Domain name
            
        Returns:
            Mock JSON response string
        """
        # Simple logic to generate mock scores based on answer similarity
        gold_lower = gold_answer.lower().strip()
        candidate_lower = candidate_answer.lower().strip()
        
        # Calculate a mock score based on simple string similarity
        if gold_lower == candidate_lower:
            score = 10
        elif gold_lower in candidate_lower or candidate_lower in gold_lower:
            score = 8
        elif any(word in candidate_lower for word in gold_lower.split() if len(word) > 3):
            score = 6
        elif len(set(gold_lower.split()) & set(candidate_lower.split())) > 0:
            score = 4
        else:
            score = 2
        
        # Add some randomness to make it more realistic
        import random
        score += random.randint(-1, 1)
        score = max(0, min(10, score))  # Ensure score is between 0-10
        
        # Generate mock JSON response
        mock_response = {
            "score": score
        }
        
        # Print debug information
        print(f"\n🔧 DEBUG MODE - Evaluation Response")
        print("=" * 50)
        print(f"Question ID: {qid}")
        print(f"Domain: {domain or 'generic'}")
        print(f"Gold Answer: {gold_answer[:100]}{'...' if len(gold_answer) > 100 else ''}")
        print(f"Candidate Answer: {candidate_answer[:100]}{'...' if len(candidate_answer) > 100 else ''}")
        print(f"Generated Score: {score}/10")
        print("=" * 50)
        
        return json.dumps(mock_response)
