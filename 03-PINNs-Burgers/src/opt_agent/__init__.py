"""LLM-based hyperparameter optimizer module.

Public API mirrors the bo module for drop-in comparison experiments.

Example
-------
>>> from opt_agent import LLMOptimizer, LLMConfig
>>> optimizer = LLMOptimizer(search_space, objective, LLMConfig())
>>> result = optimizer.optimize()
"""

from opt_agent.chain import BaseChain, GeminiChain
from opt_agent.config import LLMConfig, LLMIterationMeta, LLMResult
from opt_agent.optimizer import LLMOptimizer
from opt_agent.proposal import LLMProposal
from opt_agent.prompt import (
    MaximizeObjectivePromptBuilder,
    NarrowSearchSpacePromptBuilder,
    PromptBuilder,
)
from opt_agent.report import IterationReportWriter

__all__ = [
    "BaseChain",
    "GeminiChain",
    "IterationReportWriter",
    "LLMConfig",
    "LLMIterationMeta",
    "LLMResult",
    "LLMOptimizer",
    "LLMProposal",
    "MaximizeObjectivePromptBuilder",
    "NarrowSearchSpacePromptBuilder",
    "PromptBuilder",
]
