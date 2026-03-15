"""Pydantic schema for LLM structured output."""

from pydantic import BaseModel, Field


class LLMProposal(BaseModel):
    """Structured output schema for a single LLM proposal.

    Used as the target schema for LangChain's with_structured_output,
    ensuring the LLM response is validated and typed.
    """

    analysis_report: str = Field(
        description="現在の探索状況の分析（自然言語）"
    )
    proposed_params: dict[str, float | int] = Field(
        description="次に探索するパラメータ値（実スケール）"
    )
    reasoning: str = Field(
        description="提案パラメータの選択理由（自然言語）"
    )
