"""LLM chain abstraction for hyperparameter proposal generation."""

from abc import ABC, abstractmethod

from opt_tool.result import TrialResult
from opt_tool.space import SearchSpace
from opt_agent.proposal import LLMProposal
from opt_agent.prompt import PromptBuilder


class BaseChain(ABC):
    """Abstract interface for LLM-based hyperparameter proposal generation.

    Decouples the LLMOptimizer from any concrete LLM implementation,
    enabling test-time substitution with MockChain (Strategy pattern).
    """

    @abstractmethod
    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """Generate the next hyperparameter proposal from trial history.

        Parameters
        ----------
        search_space:
            Search space definition with parameter bounds.
        trials:
            All trials evaluated so far.
        objective_name:
            Name of the objective function.
        iteration_id:
            Current LLM iteration number (0-based).

        Returns
        -------
        LLMProposal
            Structured proposal with analysis, parameters, and reasoning.
        """
        ...


class GeminiChain(BaseChain):
    """LangChain-based Gemini API implementation of BaseChain.

    Builds a LCEL chain using ChatGoogleGenerativeAI and structured output,
    then calls the Gemini API to generate hyperparameter proposals.
    Retries up to 3 times on API failure.
    """

    def __init__(self, model_name: str, api_key: str) -> None:
        """Initialize GeminiChain with a LangChain LCEL chain.

        Parameters
        ----------
        model_name:
            Gemini model name (e.g., "gemini-2.0-flash").
        api_key:
            Google Gemini API key.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{human_prompt}"),
        ])
        self._chain = prompt | llm.with_structured_output(LLMProposal)

    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """Call the Gemini API and return a structured LLMProposal.

        Retries up to 3 times on API errors.

        Parameters
        ----------
        search_space:
            Search space definition.
        trials:
            All trials evaluated so far.
        objective_name:
            Name of the objective function.
        iteration_id:
            Current LLM iteration number (0-based).

        Returns
        -------
        LLMProposal
            Structured proposal from the LLM.

        Raises
        ------
        RuntimeError
            If all 3 retry attempts fail.
        """
        system_prompt = PromptBuilder.build_system_prompt(search_space, objective_name)
        human_prompt = PromptBuilder.build_human_prompt(trials, iteration_id)

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                result = self._chain.invoke({
                    "system_prompt": system_prompt,
                    "human_prompt": human_prompt,
                })
                return result  # type: ignore[return-value]
            except Exception as e:
                last_error = e
                print(f"  [GeminiChain] Attempt {attempt + 1}/3 failed: {e}")

        raise RuntimeError(
            f"GeminiChain.invoke failed after 3 attempts. Last error: {last_error}"
        )
