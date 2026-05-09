"""Discovery prep agent: web search via ddgs + Claude Haiku discovery call brief."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _gather_search_notes(company_name: str, max_results: int = 12) -> str:
    """Collect raw search snippets for the company."""
    queries = [
        company_name,
        f"{company_name} official website",
        f"{company_name} reviews",
        f"{company_name} about",
        f'"{company_name}"',
    ]
    blocks: list[str] = []
    with DDGS() as ddgs:
        for q in queries:
            try:
                results = ddgs.text(q, max_results=max_results)
                if not results:
                    time.sleep(2)
                    results = ddgs.text(q, max_results=max_results)
            except Exception:
                results = []
            if not results:
                blocks.append(f"Query: {q}\n(no results)\n---")
                continue
            for r in results:
                title = r.get("title") or ""
                href = r.get("href") or ""
                body = r.get("body") or ""
                blocks.append(f"Query: {q}\nTitle: {title}\nURL: {href}\nSnippet: {body}\n---")
    return "\n".join(blocks)


def prepare_discovery_brief(company_name: str) -> dict[str, Any]:
    """
    Search the web for ``company_name`` and produce a one-page discovery call brief.

    Returns a dict with keys: ``company``, ``brief``, ``suggested_questions`` (list of 3 str).
    """
    name = company_name.strip()
    if not name:
        raise ValueError("company_name must be non-empty")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing; add it to .env")

    research_notes = _gather_search_notes(name)

    client = Anthropic(api_key=api_key)
    user_prompt = f"""You are preparing a salesperson for a discovery call with this company.

Company to research: {name}

Below is raw text from web search snippets (titles, URLs, short descriptions). The information may be incomplete, outdated, or refer to the wrong business if names collide. Use only what the snippets plausibly support; clearly say when something is unknown or uncertain.

--- SEARCH SNIPPETS ---
{research_notes}
--- END SNIPPETS ---

Write a single narrative "brief" of about one printed page (roughly 400-700 words) that covers, in prose with clear sections or paragraphs:
1) What the company does (services/products, market, geography if known)
2) How long they have been in business, if any source suggests founding year, "since", or tenure; otherwise state that it was not found
3) Whether they appear to have a website: mention the URL if visible in the snippets, and describe what the site seems to offer or how it comes across (professional, dated, thin content, etc.) based only on titles/snippets—do not invent page details not hinted at
4) Their online presence and reviews: directories, social hints, review sites, star ratings or themes if mentioned
5) Pain points or friction that might be inferred from their public web presence (e.g. outdated site, few reviews, inconsistent listings, seasonal demand, etc.); label inference as inference

Do not include the three discovery questions inside the brief text.

Then provide exactly three discovery questions tailored to this specific company and what you found (or explicitly tailored to gaps in public information).

Reply with ONLY a JSON object (no markdown fences), with keys exactly:
{{"brief":"<the full brief as a single string with newline characters where needed>","suggested_questions":["<question 1>","<question 2>","<question 3>"]}}"""

    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=4096,
        system=(
            "You write accurate, sales-ready discovery prep from imperfect web snippets. "
            "Never fabricate specific facts; qualify uncertainty. Output only valid JSON."
        ),
        messages=[{"role": "user", "content": user_prompt}],
    )

    text_parts: list[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    raw = "".join(text_parts).strip()
    parsed = _parse_json_object(raw)
    brief = str(parsed.get("brief", "")).strip()
    questions = parsed.get("suggested_questions")
    if not isinstance(questions, list):
        questions = []
    suggested = [str(q).strip() for q in questions if str(q).strip()]
    if len(suggested) > 3:
        suggested = suggested[:3]
    _fallback = [
        "What does success look like for your business over the next 12 months, and what metrics matter most?",
        "Where do you feel the biggest friction between incoming leads and completed work—and what have you tried so far?",
        "If you could improve one operational or customer-facing process tomorrow, which would it be and why?",
    ]
    for fb in _fallback:
        if len(suggested) >= 3:
            break
        suggested.append(fb)

    return {
        "company": name,
        "brief": brief,
        "suggested_questions": suggested[:3],
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out = prepare_discovery_brief("Celsius Refrigeration Montreal")
    print(json.dumps(out, indent=2, ensure_ascii=False))
