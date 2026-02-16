import pytest
from unittest.mock import MagicMock, patch
from paper_analysis_deepagents.research_paper_summary_agent import (
    build_research_paper_summary_agent,
)
from paper_analysis_deepagents.config import Settings


@pytest.fixture
def mock_settings():
    return Settings(
        openai_endpoint="https://api.openai.com/v1/openai/v1",
        openai_api_key="sk-test-key",
        openai_model_name="gpt-5-mini",
        openai_api_version="2024-02-15-preview",
        openai_reasoning_effort=None,
        openai_temperature=0.0,
        azure_search_endpoint="https://test-search.search.windows.net",
        azure_search_api_key="test-search-key",
        azure_search_api_version="2023-11-01",
        history_max_turns=5,
        research_paper_index_name="test-index",
        trace_tool_calls=False,
        trace_skills=False,
        skills_sources=("/skills/",),
    )


def test_build_agent_structure(mock_settings):
    """Test if the research paper agent is built with correct tools."""
    with (
        patch(
            "paper_analysis_deepagents.research_paper_summary_agent.build_llm"
        ) as mock_build_llm,
        patch(
            "paper_analysis_deepagents.research_paper_summary_agent.AzureSearchConfig"
        ),
        patch(
            "paper_analysis_deepagents.research_paper_summary_agent.make_semantic_hybrid_search"
        ),
        patch(
            "paper_analysis_deepagents.research_paper_summary_agent.make_contextual_search"
        ),
        patch(
            "paper_analysis_deepagents.research_paper_summary_agent.make_keyword_search"
        ),
        patch("paper_analysis_deepagents.research_paper_summary_agent.make_chunk_read"),
        patch(
            "paper_analysis_deepagents.research_paper_summary_agent.make_cite_sources"
        ),
        patch(
            "paper_analysis_deepagents.research_paper_summary_agent.make_cite_sources"
        ),
        patch(
            "paper_analysis_deepagents.research_paper_summary_agent.create_deep_agent"
        ) as mock_create_agent,
        patch.dict("os.environ", {"EMBEDDING_MODELNAME": "text-embedding-3-small"}),
    ):
        mock_llm = MagicMock()
        mock_build_llm.return_value = mock_llm

        # Build the agent
        build_research_paper_summary_agent(mock_settings)

        # Verify create_deep_agent called
        assert mock_create_agent.called

        # Check tools provided
        call_kwargs = mock_create_agent.call_args.kwargs
        assert "tools" in call_kwargs
        tools = call_kwargs["tools"]
        assert len(tools) > 0

        # We expect at least the search tools and think tool
        # (Exact count depends on implementation, but >3 is safe bet)
        assert len(tools) >= 5
