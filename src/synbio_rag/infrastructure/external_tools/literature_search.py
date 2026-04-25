from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from ...domain.config import ToolConfig
from ...domain.schemas import QueryAnalysis, ToolReference


@dataclass
class ToolExecution:
    invoked: bool
    tool_name: str | None
    result: str | None
    references: list[ToolReference] = field(default_factory=list)


class ExternalToolManager:
    """
    企业演示版外部工具层。
    当前内置 PubMed 和 Crossref。
    """

    def __init__(self, config: ToolConfig):
        self.config = config

    def run_if_needed(self, question: str, analysis: QueryAnalysis, low_confidence: bool) -> ToolExecution:
        if not low_confidence and not analysis.requires_external_tools:
            return ToolExecution(invoked=False, tool_name=None, result=None)

        references = self._search_external_sources(question)
        if references:
            return ToolExecution(
                invoked=True,
                tool_name="pubmed_crossref_search",
                result="已获取外部补充文献线索。",
                references=references,
            )

        return ToolExecution(
            invoked=True,
            tool_name="external_research_placeholder",
            result="已触发外部工具回退，但当前未获取到外部结果。",
        )

    def _search_external_sources(self, question: str) -> list[ToolReference]:
        refs: list[ToolReference] = []
        try:
            if self.config.enable_pubmed:
                refs.extend(self._search_pubmed(question))
            if len(refs) < self.config.max_results and self.config.enable_crossref:
                refs.extend(self._search_crossref(question, self.config.max_results - len(refs)))
        except Exception:
            return []
        return refs[: self.config.max_results]

    def _search_pubmed(self, question: str) -> list[ToolReference]:
        with httpx.Client(timeout=15) as client:
            search_resp = client.get(
                f"{self.config.pubmed_base}/esearch.fcgi",
                params={"db": "pubmed", "retmode": "json", "retmax": self.config.max_results, "term": question},
            )
            search_resp.raise_for_status()
            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            summary_resp = client.get(
                f"{self.config.pubmed_base}/esummary.fcgi",
                params={"db": "pubmed", "retmode": "json", "id": ",".join(ids)},
            )
            summary_resp.raise_for_status()
            result = summary_resp.json().get("result", {})

        refs: list[ToolReference] = []
        for pmid in ids:
            item = result.get(pmid, {})
            title = item.get("title", "")
            if title:
                refs.append(
                    ToolReference(
                        source="PubMed",
                        title=title,
                        link=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        snippet=(item.get("pubdate", "") or "").strip(),
                    )
                )
        return refs

    def _search_crossref(self, question: str, remaining: int) -> list[ToolReference]:
        with httpx.Client(timeout=15) as client:
            response = client.get(
                self.config.crossref_base,
                params={"query": question, "rows": remaining},
                headers={"User-Agent": "synbio-rag-demo/0.1"},
            )
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])

        refs: list[ToolReference] = []
        for item in items:
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""
            doi = item.get("DOI", "")
            if title and doi:
                refs.append(
                    ToolReference(
                        source="Crossref",
                        title=title,
                        link=f"https://doi.org/{doi}",
                        snippet="DOI",
                    )
                )
        return refs
