import os
from typing import Optional

from tavily import TavilyClient


class WebSearcher:
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "advanced",
    ) -> None:
        key = api_key or os.getenv("TAVILY_API_KEY")
        if not key:
            raise EnvironmentError(
                "Tavily API key not found. "
                "Set the TAVILY_API_KEY environment variable or pass api_key."
            )
        self._client = TavilyClient(api_key=key)
        self.max_results = max_results
        self.search_depth = search_depth

    def search(self, query: str) -> str:
        response = self._client.search(
            query=query,
            search_depth=self.search_depth,
            max_results=self.max_results,
        )
        return "\n\n".join(r["content"] for r in response.get("results", []))