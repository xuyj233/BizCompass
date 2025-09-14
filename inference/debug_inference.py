#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Inference Module

This module provides debug functionality for the Bizcompass benchmark,
replacing API calls with mock responses while maintaining the same logic as api_inference.py.
"""

import json
import random
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

from dataloader.dataloader import load_questions
from prompts.prompt_manager import PromptManager

# Import the same exceptions and utilities as api_inference
try:
    from json_repair import repair_json
except ImportError:
    def repair_json(text):
        return None

# Mock response classes to simulate API response structure
class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)
        self.finish_reason = "stop"

class MockMessage:
    def __init__(self, content: str):
        self.content = content

class MockResponse:
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]

# Custom exceptions (same as api_inference)
class IncompleteAnswerError(Exception):
    pass

class UpstreamTransientError(Exception):
    pass

class UpstreamSaturationError(Exception):
    pass

def _get_text_from_choice(choice):
    """Extract text from choice object (same as api_inference)."""
    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
        return choice.message.content, "content"
    elif hasattr(choice, 'text'):
        return choice.text, "text"
    else:
        raise IncompleteAnswerError("No text content found in response")

def debug_call_llm(qobj: Dict[str, Any], llm_model: str, question_type: str,
                   temperature: float, top_p: float, domain: str = "") -> Dict[str, Any]:
    """
    Debug version of call_llm that provides mock responses instead of API calls.
    This function maintains the EXACT same logic as api_inference.py but with mock responses.
    """
    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.3))
    
    try:
        # Step 1: Get prompt based on domain and question type (same as api_inference)
        print(f"{Fore.CYAN}📋 Step 1: Getting system prompt")
        prompt_manager = PromptManager()
        
        # Map question type to the format expected by PromptManager
        question_type_mapping = {
            "single choice": "single",
            "multiple choice": "multiple", 
            "general qa": "general",
            "table qa": "table"
        }
        mapped_question_type = question_type_mapping.get(question_type.lower(), question_type.lower())
        print(f"{Fore.YELLOW}   Original question type: {question_type}")
        print(f"{Fore.YELLOW}   Mapped question type: {mapped_question_type}")
        
        system_prompt = prompt_manager.get_prompt(domain, mapped_question_type)
        print(f"{Fore.YELLOW}   System prompt: {system_prompt}")
        
        # Step 2: Prepare messages (same as api_inference)
        print(f"{Fore.CYAN}📋 Step 2: Preparing messages")
        user_question = qobj.get("question", qobj.get("Question", ""))
        print(f"{Fore.YELLOW}   User question: {user_question}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        print(f"{Fore.YELLOW}   Messages array created with {len(messages)} messages")
        
        # Step 3: Set max tokens based on question type (same as api_inference)
        print(f"{Fore.CYAN}📋 Step 3: Setting parameters")
        max_tokens = 512 if question_type.lower() in ["single", "multiple"] else 8192
        print(f"{Fore.YELLOW}   Max tokens: {max_tokens}")
        print(f"{Fore.YELLOW}   Model: {llm_model}")
        print(f"{Fore.YELLOW}   Temperature: {temperature}, Top-p: {top_p}")
        print(f"{Fore.YELLOW}   Domain: {domain}, Question type: {question_type}")
        
        # Step 4: Display complete API call information
        print(f"{Fore.CYAN}📋 Step 4: Complete API call that would be made:")
        print(f"{Fore.YELLOW}   client.chat.completions.create(")
        print(f"{Fore.YELLOW}       model='{llm_model}',")
        print(f"{Fore.YELLOW}       messages={messages},")
        print(f"{Fore.YELLOW}       temperature={temperature},")
        print(f"{Fore.YELLOW}       top_p={top_p},")
        print(f"{Fore.YELLOW}       max_tokens={max_tokens}")
        print(f"{Fore.YELLOW}   )")
        
        # Step 5: Show detailed message breakdown
        print(f"{Fore.CYAN}📋 Step 5: Detailed message breakdown:")
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            print(f"{Fore.YELLOW}   Message {i+1}:")
            print(f"{Fore.YELLOW}     Role: {role}")
            print(f"{Fore.YELLOW}     Content: {content}")
            print(f"{Fore.YELLOW}     Content length: {len(content)} characters")
        
        # Simulate API call with mock response (same structure as real API call)
        # In real API call: response = client.chat.completions.create(model=llm_model, messages=messages, ...)
        # Here we generate a mock response based on question type
        if question_type.lower() in ["single", "single choice"]:
            mock_answer = random.choice(["A", "B", "C", "D"])
            mock_raw_response = f"Based on the {domain} analysis, the correct answer is {mock_answer}."
        elif question_type.lower() in ["multiple", "multiple choice"]:
            mock_answer = random.choice(["A, B", "A, C", "B, D", "A, B, C"])
            mock_raw_response = f"Based on the {domain} analysis, the correct answers are {mock_answer}."
        else:
            # General QA or Table QA
            mock_responses = [
                f"Based on the {domain} principles discussed, the answer is...",
                f"The {domain} analysis shows that...",
                f"From a {domain} perspective, the correct approach is...",
                f"According to the {domain} data provided...",
                f"The most appropriate {domain} response would be..."
            ]
            mock_answer = random.choice(mock_responses)
            mock_raw_response = mock_answer
        
        # Step 6: Create mock response object (same structure as real API response)
        print(f"{Fore.CYAN}📋 Step 6: Creating mock response")
        response = MockResponse(mock_raw_response)
        print(f"{Fore.GREEN}   Mock response created: {mock_raw_response}")
        print(f"{Fore.GREEN}   Mock answer: {mock_answer}")
        
        # Step 7: Process response (same as api_inference)
        print(f"{Fore.CYAN}📋 Step 7: Processing response (multi-format value extraction)")
        try:
            raw_text, diag = _get_text_from_choice(response.choices[0])
            print(f"{Fore.GREEN}   Raw text extracted: {raw_text}")
            print(f"{Fore.GREEN}   Extraction method: {diag}")
        except IncompleteAnswerError as e:
            print(f"{Fore.RED}   ❌ Incomplete response error: {str(e)}")
            return {
                "model_raw_response": "",
                "model_answer": None,
                "error": f"Incomplete response: {str(e)}"
            }
        
        # Step 8: Check for upstream saturation keywords
        print(f"{Fore.CYAN}📋 Step 8: Checking for upstream saturation keywords")
        upstream_keyword = "Saturation Detected"  # Will be loaded from config
        if upstream_keyword in raw_text:
            print(f"{Fore.RED}   ❌ Upstream saturation detected: {raw_text[:100]}")
            raise UpstreamTransientError(f"Upstream saturation detected: {raw_text[:100]}")
        else:
            print(f"{Fore.GREEN}   ✅ No upstream saturation keywords found")
        
        # Step 9: Check for whitespace-only text
        print(f"{Fore.CYAN}📋 Step 9: Checking for whitespace-only text")
        if not raw_text or not raw_text.strip():
            print(f"{Fore.RED}   ❌ Empty or whitespace-only response")
            raise IncompleteAnswerError("Empty or whitespace-only response")
        else:
            print(f"{Fore.GREEN}   ✅ Response has content")
        
        # Step 10: Try JSON parsing
        print(f"{Fore.CYAN}📋 Step 10: Trying JSON parsing")
        try:
            # Try to parse as JSON first
            if raw_text.strip().startswith('{') or raw_text.strip().startswith('['):
                print(f"{Fore.YELLOW}   Attempting JSON parse...")
                parsed_json = json.loads(raw_text)
                if isinstance(parsed_json, dict) and "answer" in parsed_json:
                    print(f"{Fore.GREEN}   ✅ JSON parsing successful, answer found: {parsed_json['answer']}")
                    return {
                        "model_raw_response": raw_text,
                        "model_answer": parsed_json["answer"],
                        "error": None
                    }
                else:
                    print(f"{Fore.YELLOW}   JSON parsed but no 'answer' field found")
            else:
                print(f"{Fore.YELLOW}   Text doesn't start with JSON markers")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{Fore.YELLOW}   JSON parsing failed: {str(e)}")
        
        # Step 11: Try JSON repair
        print(f"{Fore.CYAN}📋 Step 11: Trying JSON repair")
        try:
            repaired_json = repair_json(raw_text)
            if isinstance(repaired_json, dict) and "answer" in repaired_json:
                print(f"{Fore.GREEN}   ✅ JSON repair successful, answer found: {repaired_json['answer']}")
                return {
                    "model_raw_response": raw_text,
                    "model_answer": repaired_json["answer"],
                    "error": None
                }
            else:
                print(f"{Fore.YELLOW}   JSON repair failed or no 'answer' field")
        except Exception as e:
            print(f"{Fore.YELLOW}   JSON repair failed: {str(e)}")
        
        # Step 12: Pure text extraction
        print(f"{Fore.CYAN}📋 Step 12: Pure text extraction")
        extracted_answer = raw_text.strip()
        print(f"{Fore.YELLOW}   Extracted text: {extracted_answer}")
        
        # Step 13: For choice questions, try to extract answer
        # Check both original and mapped question types
        is_choice_question = (
            question_type.lower() in ["single choice", "multiple choice"] or
            mapped_question_type in ["single", "multiple"]
        )
        
        if is_choice_question:
            print(f"{Fore.CYAN}📋 Step 13: Extracting answer for choice question")
            print(f"{Fore.YELLOW}   Question type: {question_type} -> {mapped_question_type}")
            # Try to extract answer from text
            answer_match = re.search(r'[A-D]', extracted_answer)
            if answer_match:
                extracted_answer = answer_match.group()
                print(f"{Fore.GREEN}   ✅ Answer extracted: {extracted_answer}")
            else:
                print(f"{Fore.RED}   ❌ Could not extract answer from response")
                raise IncompleteAnswerError("Could not extract answer from response")
        else:
            print(f"{Fore.CYAN}📋 Step 13: Non-choice question, using full text as answer")
        
        # Step 14: Return final result
        print(f"{Fore.CYAN}📋 Step 14: Returning final result")
        result = {
            "model_raw_response": raw_text,
            "model_answer": extracted_answer,
            "error": None
        }
        print(f"{Fore.GREEN}   ✅ Final result: {result}")
        return result
        
    except UpstreamTransientError as e:
        # Unify and escalate to upstream temporary error (same as api_inference)
        raise UpstreamSaturationError(f"Upstream error: {str(e)}")
    except Exception as e:
        # Error: API call exception: {e} (same as api_inference)
        raise Exception(f"API call exception Q: '{str(qobj.get('question'))[:50]}…' (Model: {llm_model}): {e}")


class DebugInference:
    """Debug inference engine that provides mock responses."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize debug inference engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.prompt_manager = PromptManager()
    
    def initialize(self):
        """Initialize debug mode (does nothing but print message)."""
        print(f"{Fore.YELLOW}🐛 Debug Mode: Skipping model initialization")
        print(f"{Fore.YELLOW}   Using mock responses for all inference calls")
    
    def process_question(self, question: Dict[str, Any], model_name: str, 
                        question_type: str, temperature: float, top_p: float, 
                        domain: str = "") -> Dict[str, Any]:
        """
        Process a single question with mock response.
        This method maintains the same interface as APIInference.process_question.
        """
        print(f"\n{Fore.MAGENTA}📝 Processing Question ID: {question.get('ID', 'unknown')}")
        return debug_call_llm(question, model_name, question_type, temperature, top_p, domain)
    
    def process_directory(self, dataset_path: Path, output_path: Path, 
                         model_name: str, question_types: List[str], 
                         domains: List[str], temperature: float, top_p: float,
                         continue_on_error: bool = True) -> Dict[str, Any]:
        """
        Process all questions in a directory structure with mock responses.
        This method maintains the same interface as APIInference.process_directory.
        """
        print_debug_header()
        
        results = {
            "total_processed": 0,
            "total_skipped": 0,
            "errors": [],
            "domains": {}
        }
        
        for domain in domains:
            domain_results = {"processed": 0, "skipped": 0, "errors": []}
            
            for question_type in question_types:
                # Find questions file (handle question type mapping)
                question_type_dir = question_type  # Use the exact question type as directory name
                questions_file = dataset_path / domain / question_type_dir / "questions.json"
                if not questions_file.exists():
                    print(f"{Fore.YELLOW}⚠️  Questions file not found: {questions_file}")
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
                    
                    print(f"{Fore.CYAN}📊 Load Questions Results:")
                    print(f"{Fore.YELLOW}   All records: {len(all_records) if all_records else 0}")
                    print(f"{Fore.YELLOW}   Questions to process: {len(questions_to_process) if questions_to_process else 0}")
                    
                    if questions_to_process is None:
                        print(f"{Fore.RED}   ❌ Failed to load questions")
                        continue
                    
                    if not questions_to_process:
                        print(f"{Fore.YELLOW}   ⏭️  No new questions to process, skipping")
                        domain_results["skipped"] += len(all_records) if all_records else 0
                        continue
                    
                    print(f"\n{Fore.CYAN}🔍 Debug Processing: {domain} - {question_type}")
                    print(f"{Fore.YELLOW}   Model: {model_name}")
                    print(f"{Fore.YELLOW}   Temperature: {temperature}, Top-p: {top_p}")
                    print(f"{Fore.YELLOW}   Questions: {len(questions_to_process)}")
                    
                    # Process questions with mock responses
                    for i, question in enumerate(questions_to_process):
                        try:
                            result = self.process_question(
                                question, model_name, question_type.lower(),
                                temperature, top_p, domain
                            )
                            
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
                            
                            all_records.append(record)
                            domain_results["processed"] += 1
                            
                            # Display progress
                            if (i + 1) % 10 == 0 or i == len(questions_to_process) - 1:
                                print(f"{Fore.GREEN}   ✅ Processed {i+1}/{len(questions_to_process)} questions")
                            
                        except Exception as e:
                            error_msg = f"Error processing question {question.get('ID', 'unknown')}: {str(e)}"
                            domain_results["errors"].append(error_msg)
                            if not continue_on_error:
                                raise
                    
                    # Save all records as JSON array
                    print(f"{Fore.CYAN}💾 Saving {len(all_records)} records to {eval_file}")
                    with open(eval_file, 'w', encoding='utf-8') as f:
                        json.dump(all_records, f, ensure_ascii=False, indent=2)
                    print(f"{Fore.GREEN}   ✅ Successfully saved records")
                    
                except Exception as e:
                    error_msg = f"Error processing {domain} {question_type}: {str(e)}"
                    domain_results["errors"].append(error_msg)
                    if not continue_on_error:
                        raise
            
            results["domains"][domain] = domain_results
            results["total_processed"] += domain_results["processed"]
            results["total_skipped"] += domain_results["skipped"]
            results["errors"].extend(domain_results["errors"])
        
        print_debug_summary(results, "inference")
        return results


def print_debug_header():
    """Print debug mode header."""
    print(f"\n{Fore.RED}{'='*60}")
    print(f"{Fore.RED}{Style.BRIGHT}🐛 DEBUG MODE ENABLED")
    print(f"{Fore.RED}{'='*60}")
    print(f"{Fore.YELLOW}This is a debug session with mock responses.")
    print(f"{Fore.YELLOW}No actual model calls will be made.")
    print(f"{Fore.RED}{'='*60}\n")


def print_debug_summary(results: Dict[str, Any], experiment_type: str):
    """Print debug session summary."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}📋 DEBUG SESSION SUMMARY")
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}Experiment Type: {experiment_type}")
    print(f"{Fore.YELLOW}Total Processed: {results.get('total_processed', 0)}")
    print(f"{Fore.YELLOW}Total Skipped: {results.get('total_skipped', 0)}")
    print(f"{Fore.YELLOW}Total Errors: {len(results.get('errors', []))}")
    
    if "domains" in results:
        print(f"\n{Fore.GREEN}Domain Results:")
        for domain, domain_results in results["domains"].items():
            print(f"  {Fore.WHITE}{domain}: {domain_results.get('processed', 0)} processed")
    
    print(f"{Fore.CYAN}{'='*60}\n")