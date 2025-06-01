"""
Evolve Framework - A toolkit for evolving and verifying questions with large language models.

This package provides utilities for question evolution, verification, and API interactions.
"""

# Import and expose all relevant functions from verify_utils
from .verify_utils import (
    SYSTEM_MESSAGES,
    call_gemini_api,
    get_evolve_prompt,
    extract_evolved_question,
    judge_evolved_question_quality,
    get_solver_prompt,
    extract_final_answer,
    verify_answers_match
)

# Define what should be available when using "from evolved_framework import *"
__all__ = [
    'SYSTEM_MESSAGES',
    'call_gemini_api',
    'get_evolve_prompt',
    'extract_evolved_question',
    'judge_evolved_question_quality',
    'get_solver_prompt',
    'extract_final_answer',
    'verify_answers_match'
]