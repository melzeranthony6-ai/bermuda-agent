"""Local trade research agent: DuckDuckGo search + Claude (LangChain) structured lead extraction."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from ddgs import DDGS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_CITY = "Montreal"


class Lead(BaseModel):
    """One local business lead."""

    name: str = Field(description="Business name as commonly listed")
    phone: str = Field(
        description="Phone number if present in the search text; otherwise empty string"
    )
    website: str = Field(
        description="Canonical website URL if present; otherwise empty string"
    )
    description: str = Field(
        description="Single-line summary of what the business does for this trade/area"
    )


class LeadExtraction(BaseModel):
    """Model response wrapper for structured output."""

    leads: list[Lead] = Field(
        description="Distinct local businesses matching the trade in the given city; aim for at least 5"
    )


def _run_text_searches(queries: list[str], max_results: int) -> list[str]:
    """Run DuckDuckGo text searches and return formatted blocks."""
    blocks: list[str] = []
    with DDGS() as ddgs:
        for q in queries:
            try:
                results = ddgs.text(q, max_results=max_results)
                if not results:
                    time.sleep(3)
                    results = ddgs.text(q, max_results=max_results)
            except Exception:
                continue
            if not results:
                continue
            for r in results:
                title = r.get("title") or ""
                href = r.get("href") or ""
                body = r.get("body") or ""
                blocks.append(f"Title: {title}\nURL: {href}\nSnippet: {body}\n---")
    return blocks


def _build_search_queries(trade: str, city: str) -> list[str]:
    """Queries tuned for local services + Montreal."""
    return [
        f'{trade} {city} business phone',
        f'{trade} contractor {city} Quebec contact',
        f'{trade} services {city}',
        f'{trade} company {city} website',
        f'local {trade} {city}',
        f'{trade} near me {city}',
        f'{city} {trade} directory',
        f'hire {trade} {city}',
    ]


def _gather_context(trade: str, city: str) -> str:
    """Collect search snippets for the model (primary + fallback queries)."""

    def _build_once() -> str:
        queries = _build_search_queries(trade, city)
        blocks = _run_text_searches(queries, max_results=12)
        if len(blocks) < 8:
            extra = [
                f"{trade} Montréal entreprise",
                f"{trade} Montréal QC téléphone",
            ]
            blocks.extend(_run_text_searches(extra, max_results=15))
        return "\n".join(blocks)

    context = _build_once()
    if not context.strip():
        time.sleep(3)
        context = _build_once()
    return context


def research_leads(trade: str, city: str = DEFAULT_CITY) -> list[dict[str, Any]]:
    """
    Search for real local businesses for ``trade`` in ``city`` (default Montreal),
    then use Claude to extract structured leads from search snippets.

    Returns a list of dicts with keys: name, phone, website, description.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is missing. Add it to your .env or environment."
        )

    context = _gather_context(trade, city)
    if not context.strip():
        raise RuntimeError(
            "No DuckDuckGo results were returned. Check your network connection."
        )

    # Cap context size for API limits
    max_chars = 100_000
    if len(context) > max_chars:
        context = context[:max_chars]

    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        api_key=api_key,
        temperature=0.1,
    )
    extractor = llm.with_structured_output(LeadExtraction)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI agent for Bermuda AI, a Montreal-based AI automation agency that sells done-for-you conversion websites and automated lead response systems to skilled trades businesses (HVAC, plumbing, electrical, roofing, contracting) in Montreal. "
                "The founder's offer is: We help trades businesses in Montreal stop losing leads after hours through an automated response system that follows up with every inquiry within 5 minutes. The first company to respond gets the job and right now that's almost never you after 6pm. "
                "Key sales principles: sell outcomes not tools, one buyer one pain one outcome, diagnose before pitching, quantify pain in dollars and jobs lost, always close to a specific next step. "
                "The discovery call sequence is: open with control, diagnose real pain, quantify cost, qualify across fit/priority/ability/timeline, frame value, secure locked next step. "
                "Cold call structure: opener, gatekeeper bridge, owner diagnosis, offer delivery, objection handling, close to next step. "
                "Pricing: $1,500 CAD setup, $250/month retainer. "
                "Prioritize small to medium owner-operated trades businesses that likely have no automated lead response and have basic or outdated websites. "
                "You only output lead records that are justified by the supplied search result snippets for the requested city and trade. "
                "Prefer businesses that clearly serve or are located in that metro area. "
                "Extract phone numbers and websites only when they appear in the snippets (or the URL field); otherwise use an empty string-do not invent contact details. "
                "Each description must be a single concise line. "
                "Return at least five distinct businesses when the snippets support that many; if fewer are clearly grounded in the text, return every grounded business you can.",
            ),
            (
                "human",
                "Trade focus: {trade}\nCity: {city}\n\n"
                "Search results:\n{context}\n\n"
                "Produce the structured lead list.",
            ),
        ]
    )

    chain = prompt | extractor
    parsed: LeadExtraction = chain.invoke(  # type: ignore[assignment]
        {"trade": trade, "city": city, "context": context}
    )

    leads_dicts = [lead.model_dump() for lead in parsed.leads]

    # Second pass with broader queries if we still need more grounded rows
    if len(leads_dicts) < 5:
        supplemental_queries = [
            f"{trade} {city} reviews",
            f"{trade} {city} site:.ca",
            f"{trade} service {city} phone",
        ]
        more_blocks = _run_text_searches(supplemental_queries, max_results=15)
        if more_blocks:
            extra_ctx = "\n".join(more_blocks)
            if len(extra_ctx) > max_chars:
                extra_ctx = extra_ctx[:max_chars]
            combined = context + "\n" + extra_ctx
            if len(combined) > max_chars:
                combined = combined[:max_chars]
            parsed2: LeadExtraction = chain.invoke(  # type: ignore[assignment]
                {"trade": trade, "city": city, "context": combined}
            )
            leads_dicts = [lead.model_dump() for lead in parsed2.leads]

    return leads_dicts


if __name__ == "__main__":
    trade_query = "plumbing"
    city_query = DEFAULT_CITY

    print("=" * 60)
    print(f"Bermuda research agent - trade: {trade_query!r}, city: {city_query!r}")
    print("=" * 60)

    results = research_leads(trade=trade_query, city=city_query)

    print(f"\nFound {len(results)} lead(s).\n")

    for idx, lead in enumerate(results, start=1):
        print(f"--- Lead {idx} ---")
        print(f"  name       : {lead.get('name', '')}")
        print(f"  phone      : {lead.get('phone', '')}")
        print(f"  website    : {lead.get('website', '')}")
        print(f"  description: {lead.get('description', '')}")
        print()

    print("JSON (pretty-printed):")
    print(json.dumps(results, indent=2, ensure_ascii=False))
