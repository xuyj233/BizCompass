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
        if domain:
            domain_lower = domain.lower()
            if domain_lower == "econ":
                return EVALUATION_PROMPT_ECON
            elif domain_lower == "fin":
                return EVALUATION_PROMPT_FIN
            elif domain_lower == "om":
                return EVALUATION_PROMPT_OM
            elif domain_lower == "stat":
                return EVALUATION_PROMPT_STAT
        
        # Default to generic evaluation prompt
        return EVALUATION_PROMPT_GENERIC
    
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
