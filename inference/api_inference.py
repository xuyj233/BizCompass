"""
API Inference Module

This module handles API-based LLM inference for the Bizcompass benchmark.
It supports various LLM providers through OpenAI-compatible APIs.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import re

from openai import OpenAI
from tqdm import tqdm

from dataloader.dataloader import load_questions
from prompts.prompt_manager import PromptManager
from utils.utils import *   # noqa: F401, needed for *split_options*, *UpstreamSaturationError*, etc.
from json_repair import repair_json

# Configuration will be loaded from BizcompassConfig

# --------------------------- Local exceptions --------------------------- #
class IncompleteAnswerError(Exception):
    """LLM response is incomplete/placeholder/invalid and should be treated as unprocessed."""
    pass

class UpstreamTransientError(Exception):
    """Upstream/capacity/rate-limit/network temporary errors that require retry."""
    pass

# --------------------------- Global variables --------------------------- #
client = None
write_lock = threading.Lock()

def initialize_api_client(api_key: str, base_url: str) -> None:
    """Initialize the OpenAI client for API calls."""
    global client
    client = OpenAI(api_key=api_key, base_url=base_url)

def _get_text_from_choice(choice):
    """
    Extract text from different response formats as fallback:
    - choices[0].message.content
    - choices[0].text (some proxies/old SDKs)
    - choices[0].message.refusal (individual implementations)
    If tool_calls exist but no text, raise IncompleteAnswerError for upper layer to handle as unprocessed.
    Returns: (raw_text: str, diag: dict)
    """
    diag = {}
    
    # Check for tool_calls first
    if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
        diag['tool_calls'] = len(choice.message.tool_calls)
        if not (hasattr(choice.message, 'content') and choice.message.content):
            raise IncompleteAnswerError("Tool calls present but no text content")
    
    # Try different text extraction methods
    raw_text = None
    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
        raw_text = choice.message.content
        diag['source'] = 'message.content'
    elif hasattr(choice, 'text'):
        raw_text = choice.text
        diag['source'] = 'text'
    elif hasattr(choice, 'message') and hasattr(choice.message, 'refusal'):
        raw_text = choice.message.refusal
        diag['source'] = 'message.refusal'
    
    if raw_text is None:
        raise IncompleteAnswerError("No text content found in response")
    
    return raw_text, diag

def call_llm(
    qobj: Dict[str, Any],
    llm_model: str,
    question_type: str,
    temperature: float,
    top_p: float,
    domain: str = "",
) -> Dict[str, Any]:
    """
    Call API LLM
    - Handle [1] multi-format value extraction: content/text/refusal, and identify tool_calls
    - Handle [3] whitespace text: first check if original is whitespace-only, then decide to throw Incomplete, rather than being swallowed by .strip()
    - Other logic remains unchanged: JSON priority parsing; strict extraction; upstream/placeholder → don't write to jsonl
    """
    if client is None:
        raise RuntimeError("API client not initialized. Call initialize_api_client() first.")
    
    try:
        # Get prompt based on domain and question type
        prompt_manager = PromptManager()
        
        # Map question type to the format expected by PromptManager
        question_type_mapping = {
            "single choice": "single",
            "multiple choice": "multiple", 
            "general qa": "general",
            "table qa": "table"
        }
        mapped_question_type = question_type_mapping.get(question_type.lower(), question_type.lower())
        
        system_prompt = prompt_manager.get_prompt(domain, mapped_question_type)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qobj.get("question", qobj.get("Question", ""))}
        ]
        
        # Set max tokens based on question type
        # Note: This will be updated to use config when APIInference is refactored
        max_tokens = 512 if question_type.lower() in ["single", "multiple"] else 8192
        
        # Make API call
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # ---------- Processing 1: multi-format value extraction ----------
        try:
            raw_text, diag = _get_text_from_choice(response.choices[0])
        except IncompleteAnswerError as e:
            return {
                "model_raw_response": "",
                "model_answer": None,
                "error": f"Incomplete response: {str(e)}"
            }
        
        # Upstream saturation keywords (vendor-side response text) → classify as upstream temporary error
        upstream_keyword = "Saturation Detected"  # Will be loaded from config
        if upstream_keyword in raw_text:
            raise UpstreamTransientError(f"Upstream saturation detected: {raw_text[:100]}")
        
        # ---------- Processing 3: whitespace-only text, don't let .strip() swallow it directly ----------
        if not raw_text or not raw_text.strip():
            raise IncompleteAnswerError("Empty or whitespace-only response")
        
        # Prioritize JSON parsing
        try:
            # Try to parse as JSON first
            if raw_text.strip().startswith('{') or raw_text.strip().startswith('['):
                parsed_json = json.loads(raw_text)
                if isinstance(parsed_json, dict) and "answer" in parsed_json:
                    return {
                        "model_raw_response": raw_text,
                        "model_answer": parsed_json["answer"],
                        "error": None
                    }
        except (JSONDecodeError, KeyError):
            pass
        
        # Try JSON repair
        try:
            repaired_json = repair_json(raw_text)
            if isinstance(repaired_json, dict) and "answer" in repaired_json:
                return {
                    "model_raw_response": raw_text,
                    "model_answer": repaired_json["answer"],
                    "error": None
                }
        except:
            pass
        
        # Pure text extraction
        extracted_answer = raw_text.strip()
        
        # For choice questions, try to extract answer
        if question_type.lower() in ["single", "multiple"]:
            # Try to extract answer from text
            answer_match = re.search(r'[A-D]', extracted_answer)
            if answer_match:
                extracted_answer = answer_match.group()
            else:
                raise IncompleteAnswerError("Could not extract answer from response")
        
        return {
            "model_raw_response": raw_text,
            "model_answer": extracted_answer,
            "error": None
        }
        
    except UpstreamTransientError as e:
        # Unify and escalate to upstream temporary error
        raise UpstreamSaturationError(f"Upstream error: {str(e)}")
    except Exception as e:
        # Error: API call exception: {e}
        raise Exception(f"API call exception Q: '{str(qobj.get('question'))[:50]}…' (Model: {llm_model}): {e}")

class APIInference:
    """API-based inference engine for the Bizcompass benchmark."""
    
    def __init__(self, api_key: str, base_url: str):
        """Initialize the API inference engine."""
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.prompt_manager = PromptManager()
        
    def initialize(self):
        """Initialize the API client."""
        initialize_api_client(self.api_key, self.base_url)
        self.client = client
    
    def process_question(self, question: Dict[str, Any], model_name: str, 
                        question_type: str, temperature: float, top_p: float, 
                        domain: str = "") -> Dict[str, Any]:
        """Process a single question using API inference."""
        return call_llm(question, model_name, question_type, temperature, top_p, domain)
    
    def process_directory(self, dataset_path: Path, output_path: Path, 
                         model_name: str, question_types: List[str], 
                         domains: List[str], temperature: float, top_p: float,
                         continue_on_error: bool = True) -> Dict[str, Any]:
        """Process all questions in a directory structure."""
        if self.client is None:
            self.initialize()
        
        results = {
            "total_processed": 0,
            "total_skipped": 0,
            "errors": [],
            "domains": {}
        }
        
        for domain in domains:
            domain_results = {"processed": 0, "skipped": 0, "errors": []}
            
            for question_type in question_types:
                # Find questions file
                questions_file = dataset_path / domain / question_type / "questions.json"
                if not questions_file.exists():
                    continue
                
                # Create output directory
                output_dir = output_path / domain / question_type.lower().replace(" ", "") / model_name / f"tem{temperature}" / f"top_p{top_p}" / "evaluation"
                output_dir.mkdir(parents=True, exist_ok=True)
                eval_file = output_dir / "questions_eval.json"
                
                try:
                    # Load questions with resume functionality
                    all_records, questions_to_process = load_questions(
                        questions_file, eval_file, question_type.lower()
                    )
                    
                    if questions_to_process is None:
                        continue
                    
                    if not questions_to_process:
                        domain_results["skipped"] += len(all_records)
                        continue
                    
                    # Process questions with threading
                    max_workers = 20  # Will be loaded from config
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_question = {
                            executor.submit(
                                self.process_question, q, model_name, question_type.lower(),
                                temperature, top_p, domain
                            ): q for q in questions_to_process
                        }
                        
                        for future in tqdm(concurrent.futures.as_completed(future_to_question), 
                                         total=len(questions_to_process), 
                                         desc=f"Processing {domain} {question_type}"):
                            question = future_to_question[future]
                            try:
                                result = future.result()
                                
                                # Create complete record
                                record = {
                                    "ID": question.get("ID"),
                                    "Question": question.get("Question"),
                                    "Answer": question.get("Answer"),
                                    "gold_answer": question.get("Answer"),
                                    "qid": question.get("qid"),
                                    "question": question.get("question"),
                                    "model_evaluation_result": result
                                }
                                
                                # Thread-safe writing
                                with write_lock:
                                    all_records.append(record)
                                
                                domain_results["processed"] += 1
                                
                            except Exception as e:
                                error_msg = f"Error processing question {question.get('ID', 'unknown')}: {str(e)}"
                                domain_results["errors"].append(error_msg)
                                if not continue_on_error:
                                    raise
                    
                    # Save all records as JSON array
                    with open(eval_file, 'w', encoding='utf-8') as f:
                        json.dump(all_records, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    error_msg = f"Error processing {domain} {question_type}: {str(e)}"
                    domain_results["errors"].append(error_msg)
                    if not continue_on_error:
                        raise
            
            results["domains"][domain] = domain_results
            results["total_processed"] += domain_results["processed"]
            results["total_skipped"] += domain_results["skipped"]
            results["errors"].extend(domain_results["errors"])
        
        return results
