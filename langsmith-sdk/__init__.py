"""
LangSmith Agent SDK

一个强大而易用的LangSmith SDK，专为Agent的trace追踪和evaluation评估而设计。
"""

from .langsmith_agent_sdk import (
    LangSmithAgentSDK,
    AgentTrace,
    EvaluationResult,
    accuracy_evaluator,
    similarity_evaluator,
    custom_evaluator,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "LangSmithAgentSDK",
    "AgentTrace",
    "EvaluationResult",
    "accuracy_evaluator",
    "similarity_evaluator",
    "custom_evaluator",
]