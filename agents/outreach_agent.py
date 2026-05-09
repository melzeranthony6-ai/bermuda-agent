"""Outreach agent: personalized cold email for one research lead via Claude Haiku."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

EXACT_OFFER = (
    "We help trades businesses in Montreal stop losing leads after hours. "
    "We do this through an automated response system that follows up with "
    "every inquiry within 5 minutes. It works because the first company "
    "to respond gets the job."
)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def write_outreach_email(lead: dict[str, Any]) -> dict[str, str]:
    """
    Build a personalized cold email for one lead.

    `lead` matches research_agent output: keys name, phone, website, description.
    Returns: {"to_business", "subject", "body"}.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing; add it to .env")

    name = str(lead.get("name", "")).strip()
    phone = str(lead.get("phone", "")).strip()
    website = str(lead.get("website", "")).strip()
    description = str(lead.get("description", "")).strip()

    client = Anthropic(api_key=api_key)

    user_prompt = f"""You are drafting one cold outreach email.

Lead (Montreal-area trade business):
- Business name: {name}
- Phone: {phone}
- Website: {website}
- Description from research: {description}

The pitch must use this EXACT three-sentence offer, copied word-for-word in the email body as one block (same punctuation; use only ASCII hyphen-minus `-`):
{EXACT_OFFER}

Around that block you may add a short personalized greeting and closing that mentions their name/description in a plausible way.

Rules:
- Personalize using the business name and the description.
- The entire body (including the required offer text) must be strictly under 150 words.
- Plain, professional tone; no fake claims beyond the description above.

Reply with ONLY a JSON object, no markdown fences, keys exactly:
{{"subject":"<short subject line>","body":"<email body>"}}"""

    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system="Follow instructions exactly. Output only JSON.",
        messages=[{"role": "user", "content": user_prompt}],
    )

    text_parts: list[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    raw = "".join(text_parts).strip()

    data = _parse_json_object(raw)
    subject = str(data["subject"]).strip()
    body = str(data["body"]).strip()

    return {
        "to_business": name,
        "subject": subject,
        "body": body,
    }


if __name__ == "__main__":
    fake_lead = {
        "name": "Plomberie Dupont",
        "phone": "514-555-0199",
        "website": "https://www.plomberiedupont.example",
        "description": (
            "Residential and commercial plumber in Montreal: leaks, installs, "
            "drains, and emergency calls."
        ),
    }
    result = write_outreach_email(fake_lead)
    print(result)
