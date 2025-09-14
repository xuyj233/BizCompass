"""
Prompt Manager

This module provides a centralized interface for managing and retrieving prompts
for different domains and question types in the Bizcompass benchmark.
"""

from typing import Dict, Any
from .domain_prompts import *


class PromptManager:
    """Centralized prompt management for the Bizcompass benchmark."""
    
    def __init__(self):
        """Initialize the prompt manager."""
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Dict[str, str]]:
        """Load all prompts into a structured dictionary."""
        prompts = {
            "single": {
                "econ": single_prompt_econ,
                "fin": single_prompt_fin,
                "om": single_prompt_om,
                "stat": single_prompt_stat,
            },
            "multiple": {
                "econ": multiple_prompt_econ,
                "fin": multiple_prompt_fin,
                "om": multiple_prompt_om,
                "stat": multiple_prompt_stat,
            },
            "general": {
                "econ": general_prompt_econ,
                "fin": general_prompt_fin,
                "om": general_prompt_om,
                "stat": general_prompt_stat,
            },
            "table": {
                "econ": table_prompt_econ,
                "fin": table_prompt_fin,
                "om": table_prompt_om,
                "stat": table_prompt_stat,
            }
        }
        return prompts
    
    def get_prompt(self, domain: str, question_type: str) -> str:
        """
        Get the appropriate prompt for a given domain and question type.
        
        Args:
            domain: Domain name ('Econ', 'Fin', 'OM', 'Stat')
            question_type: Question type ('single', 'multiple', 'general', 'table')
            
        Returns:
            The corresponding prompt string
            
        Raises:
            ValueError: If domain or question_type is not supported
        """
        domain_lower = domain.lower()
        question_type_lower = question_type.lower()
        
        # Validate domain
        if domain_lower not in ["econ", "fin", "om", "stat"]:
            raise ValueError(f"Unsupported domain: {domain}. Only Econ, Fin, OM, Stat are supported.")
        
        # Validate question_type  
        if question_type_lower not in ["single", "multiple", "general", "table"]:
            raise ValueError(f"Unsupported question_type: {question_type}. Only single, multiple, general, table are supported.")
        
        # Get prompt
        try:
            return self.prompts[question_type_lower][domain_lower]
        except KeyError:
            raise ValueError(f"Prompt not found for domain '{domain}' and question_type '{question_type}'")
    
    def get_evaluation_prompt(self, domain: str = None, question_type: str = None) -> str:
        """
        Get the evaluation prompt for a given domain and question type.
        
        Args:
            domain: Domain name (optional, for domain-specific evaluation)
            question_type: Question type (optional, for type-specific evaluation)
            
        Returns:
            The evaluation prompt string
        """
        if domain and question_type:
            # Domain and question type specific evaluation prompt
            return f"""You are an expert evaluator for {domain} {question_type} questions. Please evaluate the following answer for correctness and quality.

Please provide your evaluation in the following JSON format:
{{
    "correctness": "correct" or "incorrect" or "partially_correct",
    "score": 0.0 to 1.0,
    "explanation": "Brief explanation of your evaluation",
    "suggestions": "Any suggestions for improvement"
}}"""
        else:
            # Generic evaluation prompt
            return """You are an expert evaluator. Please evaluate the following answer for correctness and quality.

Please provide your evaluation in the following JSON format:
{
    "correctness": "correct" or "incorrect" or "partially_correct",
    "score": 0.0 to 1.0,
    "explanation": "Brief explanation of your evaluation",
    "suggestions": "Any suggestions for improvement"
}"""
    
    def get_evaluation_user_template(self) -> str:
        """Get the user template for evaluation."""
        return USER_TMPL
    
    def list_available_prompts(self) -> Dict[str, list]:
        """List all available prompts by category."""
        return {
            "domains": list(self.prompts["single"].keys()),
            "question_types": list(self.prompts.keys())
        }
    
    def validate_prompt(self, domain: str, question_type: str) -> bool:
        """
        Validate if a prompt exists for the given domain and question type.
        
        Args:
            domain: Domain name
            question_type: Question type
            
        Returns:
            True if prompt exists, False otherwise
        """
        try:
            self.get_prompt(domain, question_type)
            return True
        except ValueError:
            return False


def get_prompt_by_domain_and_type(domain: str, question_type: str) -> str:
    """
    Legacy function for backward compatibility.
    
    Args:
        domain: Domain name ('Econ', 'Fin', 'OM', 'Stat')
        question_type: Question type ('single', 'multiple', 'general', 'table')
    
    Returns:
        The corresponding prompt string
    """
    manager = PromptManager()
    return manager.get_prompt(domain, question_type)
