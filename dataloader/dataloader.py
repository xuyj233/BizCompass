"""dataloader.py

Utilities for loading question and answer data from JSON files.

This module provides two helper functions:
    - load_questions: Parse a question JSON file and prepare records for evaluation.
    - load_answers_path: Recursively collect answer JSON files from a directory tree.

The functions implement resume‑from‑checkpoint logic, robust file‑encoding
fallback, and defensive error handling to make batch processing resilient.
"""

import json
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from json import JSONDecodeError

from utils.utils import collect_questions

# Import record validation functions
def _record_is_valid(record: Dict[str, Any], question_type: str) -> bool:
    """
    Return True if the record contains a valid evaluation result.
    """
    if not record:
        return False
    
    eval_result = record.get("model_evaluation_result")
    if not isinstance(eval_result, dict):
        return False
    
    raw_response = eval_result.get("model_raw_response", "")
    model_answer = eval_result.get("model_answer")
    error = eval_result.get("error")
    
    # If there's an error, consider invalid (but error being None or empty string doesn't count as error)
    if error and str(error).strip():
        return False
    # Upstream/capacity/rate limit text traces (if historically miswritten, also consider invalid, remove at startup)
    if _looks_like_upstream_error(error) or _looks_like_upstream_error(raw_response):
        return False
    
    # Check if it's an incomplete placeholder
    if _is_incomplete_placeholder(raw_response, question_type, model_answer, error):
        return False
    
    return True

def _looks_like_upstream_error(s: str | None) -> bool:
    """
    Unified judgment of whether it's an upstream temporary error (429/403/503/capacity/rate limit/network etc.).
    If hit, it belongs to the "should retry, don't write to jsonl" case.
    """
    if not s:
        return False
    t = s.lower()
    # HTTP/gateway/network clues
    http_clues = [
        "error code: 429", " error 429", " http 429", " 429 ",
        "error code: 403", " http 403", " 403 ",
        "error code: 503", " http 503", " 503 ",
        "502 bad gateway", "504 gateway timeout", "timeout", "timed out",
        "connection refused", "unreachable", "no route to host", "network error",
    ]
    # Capacity/rate limit/channel clues (Chinese and English)
    capacity_clues = [
        "rate limit", "capacity", "overload", "saturated", "temporarily unavailable",
        "upstream", "model_not_found",
    ]
    return any(c in t for c in (http_clues + capacity_clues))

def _is_incomplete_placeholder(raw: str | None,
                               question_type: str,
                               extracted: Any,
                               err: str | None) -> bool:
    """
    Return True when the content is a placeholder/low-quality response that
    should be treated as 'unprocessed' (skip writing results).
    """
    s = (raw or "").strip()
    if not s:
        return True
    # Pure placeholders: "Answer:", "Final answer:", "Solution:", etc.
    import re
    if re.match(r'^(answer|final\s*answer|solution)\s*[:：]?\s*$', s, flags=re.I):
        return True
    # Very short generic output for general QA
    if question_type.lower() not in ("single", "multiple") and len(s) < 12:
        return True
    # For single-choice: failed to extract a letter -> treat as incomplete
    if question_type.lower() == "single" and (err or "").lower().startswith("warning: could not extract"):
        return True
    # For multiple-choice: no letters extracted
    if question_type.lower() == "multiple" and isinstance(extracted, list) and not extracted:
        return True
    return False


def get_question_id(question_obj: Dict[str, Any]) -> str:
    """Extract question ID from a question object, supporting both old and new formats.
    
    Args:
        question_obj: Question dictionary that may contain 'ID' (new format) or 'qid' (old format)
        
    Returns:
        str: Question ID or empty string if not found
    """
    # Check new format first (ID), then old format (qid)
    qid = question_obj.get("ID") or question_obj.get("qid", "")
    return str(qid) if qid is not None else ""


def normalize_question_object(question_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a question object to have consistent field names for processing.
    
    This function ensures that downstream code can rely on consistent field names
    regardless of whether the input uses old or new format.
    
    Args:
        question_obj: Original question dictionary
        
    Returns:
        Dict[str, Any]: Normalized question dictionary with standardized field names
    """
    normalized = question_obj.copy()
    
    # Ensure consistent ID field
    if "ID" in question_obj:
        normalized["qid"] = str(question_obj["ID"])
    elif "qid" in question_obj:
        normalized["qid"] = str(question_obj["qid"])
    
    # Ensure consistent question text field
    if "Question" in question_obj:
        normalized["question"] = question_obj["Question"]
    elif "question" in question_obj:
        normalized["question"] = question_obj["question"]
    
    # Ensure consistent answer field - keep original Answer field, don't create duplicate gold_answer
    if "Answer" in question_obj:
        normalized["Answer"] = question_obj["Answer"]
    elif "gold_answer" in question_obj:
        normalized["Answer"] = question_obj["gold_answer"]
    
    return normalized


def load_questions(
    input_json_file_path: Path,
    eval_jsonl_path: Path,
    question_type: str = "single",
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """Load questions from a single JSON file and apply resume logic.

    The function scans *input_json_file_path* (which may contain either a
    single root object or a list of objects) and extracts every nested
    question item via ``utils.utils.collect_questions``. It then compares the
    extracted question IDs (``ID`` or ``qid`` fields) with the IDs already present in
    *eval_jsonl_path* so that previously processed questions are skipped when
    the script is re‑run.

    Args:
        input_json_file_path (Path): Path to the source question JSON file.
        eval_jsonl_path (Path): Path to the evaluation ``.jsonl`` file. If the
            file exists it is parsed to determine which questions have already
            been handled.

    Returns:
        Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]
            A two‑tuple where
            - all_records_for_this_file: list that begins with any previously
              processed records loaded from *eval_jsonl_path*. Callers are
              expected to append new records to the same list before writing
              it back to disk in a single operation.
            - questions_to_process_now: list of new question objects that have
              not yet been processed. ``None`` is returned for both elements
              if the input file could not be read or contained no valid
              questions.

    Notes:
        * The function prints progress information instead of raising
          exceptions so that a batch runner can continue with the next file.
        * The resume mechanism relies on each question record containing a
          non‑empty ``ID`` or ``qid`` string.

    Example:
        >>> all_prev, pending = load_questions(
        ...     Path('sample_questions.json'),
        ...     Path('eval_records.jsonl'),
        ... )
        >>> for q in pending:
        ...     process_question(q)
    """
    # Attempt to read file content using UTF‑8 with and without BOM.
    try:
        try:
            file_content = input_json_file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            print(
                f"    [Info] Failed to decode {input_json_file_path.name} using "
                "utf-8-sig, trying utf-8."
            )
            file_content = input_json_file_path.read_text(encoding="utf-8")

        data_from_file_raw = json.loads(file_content)

    except FileNotFoundError:
        print(f"    [Error] File not found: {input_json_file_path}. Skipping.")
        return None, None
    except JSONDecodeError as e:
        print(f"    [Error] Invalid JSON in {input_json_file_path.name}: {e}. Skipping.")
        return None, None
    except Exception as e:
        print(f"    [Error] Failed to read {input_json_file_path.name}: {e}. Skipping.")
        return None, None

    # Normalise to list so collect_questions can work uniformly.
    data_to_scan: List[Any] = (
        data_from_file_raw if isinstance(data_from_file_raw, list) else [data_from_file_raw]
    )

    all_question_objects_from_file: List[Dict[str, Any]] = []
    collect_questions(data_to_scan, all_question_objects_from_file)

    if not all_question_objects_from_file:
        print(f"    [Info] No valid question items found in {input_json_file_path.name}. Skipping.")
        return None, None

    print(
        f"    Found {len(all_question_objects_from_file)} valid question items "
        f"in {input_json_file_path.name}."
    )

    # Initialise containers for resume logic.
    previously_processed_records: List[Dict[str, Any]] = []
    processed_qids: set[str] = set()  # QIDs that have already been processed.
    valid_processed_qids: set[str] = set()  # QIDs with valid results

    # ------------------------------------------------------------------
    # Resume from existing evaluation file, if any.
    # ------------------------------------------------------------------
    if eval_jsonl_path.exists():
        print(f"    [Info] Resuming from existing file: {eval_jsonl_path.name}")
        try:
            with eval_jsonl_path.open("r", encoding="utf-8") as f_json:
                data = json.load(f_json)
                
                # Handle both single object and array formats
                if isinstance(data, dict):
                    records = [data]
                elif isinstance(data, list):
                    records = data
                else:
                    print(f"    [Warning] Unexpected JSON format in {eval_jsonl_path.name}")
                    records = []
                
                for record in records:
                    try:
                        qid = get_question_id(record)

                        # Only add well‑formed, non‑empty QIDs to the skip list.
                        if qid and qid.strip():
                            previously_processed_records.append(record)
                            processed_qids.add(qid)
                            # Check if record is valid, only valid records are added to valid_processed_qids
                            if _record_is_valid(record, question_type):
                                valid_processed_qids.add(qid)
                        else:
                            # Keep the record for aggregation, but we cannot use
                            # it to decide skipping logic.
                            previously_processed_records.append(record)

                    except Exception as e:
                        print(
                            f"    [Warning] Skipped malformed record in {eval_jsonl_path.name}: {e}"
                        )

            # Check file completeness: only completely skip if valid record count equals source file question count
            total_source_questions = len(all_question_objects_from_file)
            total_processed_questions = len(processed_qids)
            total_valid_questions = len(valid_processed_qids)
            
            print(
                f"    Loaded {len(previously_processed_records)} records from "
                f"{eval_jsonl_path.name} "
                f"({total_processed_questions} unique QIDs, {total_valid_questions} valid)."
            )
            
            # Completeness check: use valid record count
            if total_valid_questions == total_source_questions:
                print(f"    ✅ File processing completed ({total_valid_questions}/{total_source_questions})")
            else:
                print(f"    ⚠️  File processing incomplete ({total_valid_questions}/{total_source_questions}), will continue processing remaining questions")
        except Exception as e:
            print(
                f"    [Error] Failed to read/parse {eval_jsonl_path.name} for "
                f"resuming: {e}. Treating all items as new."
            )
            previously_processed_records = []
            processed_qids = set()

    # ------------------------------------------------------------------
    # Determine which questions still need processing.
    # ------------------------------------------------------------------
    questions_to_process_now: List[Dict[str, Any]] = []
    
    # Create mapping of processed records for checking error status
    processed_records_by_qid: Dict[str, Dict[str, Any]] = {}
    for record in previously_processed_records:
        qid = get_question_id(record)
        if qid and qid.strip():
            processed_records_by_qid[qid] = record
    
    for qobj in all_question_objects_from_file:
        qid = get_question_id(qobj)

        # Process item if:
        #   1. qid is not a non‑empty string, or
        #   2. qid not found in the set of processed IDs, or
        #   3. qid found but has upstream error (Error_Upstream_Exception)
        should_process = True
        
        if qid and qid.strip() and qid in processed_qids:
            # Only skip valid records, invalid records need reprocessing
            if qid in valid_processed_qids:
                should_process = False
            else:
                print(f"    [Info] Reprocessing QID {qid} (previous record invalid)")
                should_process = True
        
        if should_process:
            # Normalize question object to ensure consistent field names
            normalized_qobj = normalize_question_object(qobj)
            questions_to_process_now.append(normalized_qobj)

    num_skipped = len(all_question_objects_from_file) - len(questions_to_process_now)
    if num_skipped > 0:
        print(
            f"    Skipped {num_skipped} already processed questions "
            f"(matched QIDs found in {eval_jsonl_path.name})."
        )

    # Aggregate previously processed and yet‑to‑process records so that callers
    # can write everything back in one go.
    all_records_for_this_file: List[Dict[str, Any]] = list(previously_processed_records)

    return all_records_for_this_file, questions_to_process_now


def load_answers_path(input_dir: Path) -> List[Path]:
    """Recursively collect all ``.json`` answer files under *input_dir*.

    Expected directory structure::

        <input_dir>/
            <question_type>/
                <model_name>/
                    answers.json

    Every ``*.json`` file encountered at the third level will be returned.

    Args:
        input_dir (Path): Root directory containing sub‑directories arranged
            by question type and model name.

    Returns:
        List[Path]: Absolute paths to every answer JSON file discovered.
    """
    json_files_to_process: List[Path] = []

    # Walk the directory tree up to three levels.
    for q_type_dir in input_dir.iterdir():
        if q_type_dir.is_dir():
            for model_name_dir in q_type_dir.iterdir():
                if model_name_dir.is_dir():
                    # Pick up *.json files directly under the model directory.
                    for json_file in model_name_dir.glob("*.json"):
                        if json_file.is_file():
                            json_files_to_process.append(json_file)

    return json_files_to_process
