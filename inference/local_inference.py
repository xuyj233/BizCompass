"""
Local Inference Module

This module handles local LLM inference for the Bizcompass benchmark.
It supports local model deployment with multi-GPU support and OOM handling.
"""

import os
import time
import torch
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

# Set environment variables for better performance
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_DISABLE_ADDR2LINE", "1")

from dataloader.dataloader import load_questions
from prompts.prompt_manager import PromptManager

print("[local_inference] rev-2025-09-09 | raw_response is original output & choice max_new_tokens=256")

# Configuration will be loaded from BizcompassConfig
# These are fallback defaults for backward compatibility
MODEL_DIR = "/home/bld/data/data2/Yichun_Lu/fino1/llama3-8b-instruct"
DTYPE = torch.float16
MAX_NEW_TOKENS_DEFAULT = int(os.environ.get("LLMBENCH_MAX_NEW_TOKENS", "8192"))
MAX_NEW_TOKENS_CHOICE = int(os.environ.get("LLMBENCH_MAX_NEW_TOKENS_CHOICE", "256"))
DEFAULT_BATCH_SIZE = int(os.environ.get("LLMBENCH_BATCH_SIZE", "64"))
MAX_RETRY_ATTEMPTS = int(os.environ.get("LLMBENCH_MAX_RETRIES", "3"))
SUPPRESS_THINKING = os.environ.get("LLMBENCH_SUPPRESS_THINKING", "true").lower() == "true"

# Global model and tokenizer
model = None
tokenizer = None
device = None

def initialize_local_model(model_path: str = None, dtype: torch.dtype = None):
    """Initialize the local model and tokenizer."""
    global model, tokenizer, device
    
    if model_path is None:
        model_path = MODEL_DIR
    if dtype is None:
        dtype = DTYPE
    
    print(f"[INFO] Loading model from: {model_path}")
    print(f"[INFO] Using dtype: {dtype}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        device = next(model.parameters()).device
        print(f"[INFO] Model loaded successfully on device: {device}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

def _get_bad_words_ids():
    """Get bad words IDs to suppress thinking style responses."""
    if not SUPPRESS_THINKING or tokenizer is None:
        return None
    
    thinking_words = ["thinking", "Thinking", "Let me think", "let me think"]
    bad_words_ids = []
    
    for word in thinking_words:
        try:
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            if word_ids:
                bad_words_ids.append(word_ids)
        except:
            continue
    
    return bad_words_ids if bad_words_ids else None

def _generate_batch_texts(prompts: List[str], question_type: str, temperature: float, top_p: float) -> List[str]:
    """Generate texts for a batch of prompts."""
    if model is None or tokenizer is None:
        raise RuntimeError("Model not initialized. Call initialize_local_model() first.")
    
    # Set max tokens based on question type
    max_new_tokens = MAX_NEW_TOKENS_CHOICE if question_type.lower() in ["single", "multiple"] else MAX_NEW_TOKENS_DEFAULT
    
    # Prepare generation parameters
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    
    if temperature > 0:
        generation_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        # Greedy decoding
        generation_kwargs.update({
            "num_beams": 1,
            "early_stopping": True,
        })
    
    # Anti-repetition settings
    generation_kwargs.update({
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
    })
    
    # Suppress thinking style
    bad_words_ids = _get_bad_words_ids()
    if bad_words_ids:
        generation_kwargs["bad_words_ids"] = bad_words_ids
    
    # Tokenize inputs
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    input_length = inputs["input_ids"].shape[1]
    
    try:
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # Decode only the newly generated parts
        generated_texts = []
        for i, output in enumerate(outputs):
            # Extract only the newly generated tokens
            new_tokens = output[input_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"⚠️ OOM during generation: {e}")
        raise

def _extract_answer_from_response(response: str, question_type: str) -> Any:
    """Extract answer from model response based on question type."""
    if not response or not response.strip():
        return None
    
    response = response.strip()
    
    if question_type.lower() == "single":
        # Extract single choice answer (A, B, C, D)
        match = re.search(r'\b([A-D])\b', response)
        return match.group(1) if match else None
    
    elif question_type.lower() == "multiple":
        # Extract multiple choice answers
        matches = re.findall(r'\b([A-D])\b', response)
        return matches if matches else []
    
    else:
        # For general QA, return the full response
        return response

def call_llm_local_batch(questions: List[Dict[str, Any]], model_name: str, question_type: str,
                        temperature: float, top_p: float, domain: str = "") -> List[Dict[str, Any]]:
    """Process a batch of questions using local model inference."""
    if model is None or tokenizer is None:
        raise RuntimeError("Model not initialized. Call initialize_local_model() first.")
    
    prompt_manager = PromptManager()
    results = []
    
    # Prepare prompts
    prompts = []
    for question in questions:
        system_prompt = prompt_manager.get_prompt(domain, question_type)
        user_content = question.get("question", question.get("Question", ""))
        
        # Format prompt using chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # Fallback if chat template fails
            prompt = f"{system_prompt}\n\nUser: {user_content}\n\nAssistant:"
        
        prompts.append(prompt)
    
    # Generate responses
    batch_size = DEFAULT_BATCH_SIZE
    all_responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        retry_count = 0
        while retry_count < MAX_RETRY_ATTEMPTS:
            try:
                batch_responses = _generate_batch_texts(batch_prompts, question_type, temperature, top_p)
                all_responses.extend(batch_responses)
                break
                
            except torch.cuda.OutOfMemoryError:
                retry_count += 1
                if retry_count < MAX_RETRY_ATTEMPTS:
                    batch_size = max(1, batch_size // 2)
                    print(f"⚠️ OOM, reducing batch size and retrying, new batch_size={batch_size}")
                    batch_prompts = batch_prompts[:batch_size]
                else:
                    raise
    
    # Process responses
    for i, (question, response) in enumerate(zip(questions, all_responses)):
        # Check for empty responses and retry if needed
        if not response or not response.strip():
            print(f"⚠️ Empty response for question {question.get('ID', i)}, retrying...")
            try:
                single_prompt = prompts[i]
                response = _generate_batch_texts([single_prompt], question_type, temperature, top_p)[0]
            except Exception as e:
                print(f"❌ Retry failed for question {question.get('ID', i)}: {e}")
                response = ""
        
        # Extract answer
        extracted_answer = _extract_answer_from_response(response, question_type)
        
        # Create result
        result = {
            "model_raw_response": response,  # ✅ Ensure original output
            "model_answer": extracted_answer,
            "error": None if extracted_answer is not None else "Failed to extract answer"
        }
        
        results.append(result)
    
    return results

class LocalInference:
    """Local model inference engine for the Bizcompass benchmark."""
    
    def __init__(self, model_path: str = None, dtype: torch.dtype = None):
        """Initialize the local inference engine."""
        self.model_path = model_path or MODEL_DIR
        self.dtype = dtype or DTYPE
        self.prompt_manager = PromptManager()
        self.initialized = False
    
    def initialize(self):
        """Initialize the local model."""
        initialize_local_model(self.model_path, self.dtype)
        self.initialized = True
    
    def process_question(self, question: Dict[str, Any], model_name: str, 
                        question_type: str, temperature: float, top_p: float, 
                        domain: str = "") -> Dict[str, Any]:
        """Process a single question using local inference."""
        if not self.initialized:
            self.initialize()
        
        results = call_llm_local_batch([question], model_name, question_type, temperature, top_p, domain)
        return results[0] if results else {"model_raw_response": "", "model_answer": None, "error": "No response"}
    
    def process_directory(self, dataset_path: Path, output_path: Path, 
                         model_name: str, question_types: List[str], 
                         domains: List[str], temperature: float, top_p: float,
                         continue_on_error: bool = True) -> Dict[str, Any]:
        """Process all questions in a directory structure."""
        if not self.initialized:
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
                eval_file = output_dir / "questions_eval.jsonl"
                
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
                    
                    print(f"Processing {len(questions_to_process)} questions for {domain} {question_type}")
                    
                    # Process questions in batches
                    batch_size = DEFAULT_BATCH_SIZE
                    for i in range(0, len(questions_to_process), batch_size):
                        batch_questions = questions_to_process[i:i + batch_size]
                        
                        try:
                            batch_results = call_llm_local_batch(
                                batch_questions, model_name, question_type.lower(),
                                temperature, top_p, domain
                            )
                            
                            # Create complete records
                            for question, result in zip(batch_questions, batch_results):
                                record = {
                                    "ID": question.get("ID"),
                                    "Question": question.get("Question"),
                                    "Answer": question.get("Answer"),
                                    "qid": question.get("qid"),
                                    "question": question.get("question"),
                                    "model_evaluation_result": result
                                }
                                all_records.append(record)
                                domain_results["processed"] += 1
                            
                        except Exception as e:
                            error_msg = f"Error processing batch {i//batch_size + 1}: {str(e)}"
                            domain_results["errors"].append(error_msg)
                            if not continue_on_error:
                                raise
                    
                    # Save all records
                    with open(eval_file, 'w', encoding='utf-8') as f:
                        for record in all_records:
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    
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
