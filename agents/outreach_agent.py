"""Cold call script generator for one lead via Claude Haiku."""

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
    "We help trades businesses in Montreal stop losing leads after hours through an "
    "automated response system that follows up with every inquiry within 5 minutes. "
    "The first company to respond gets the job — and right now that's almost never "
    "you after 6pm."
)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def generate_call_script(lead: dict[str, str]) -> dict[str, Any]:
    """
    Generate a personalized cold call script for one business lead.

    Expected lead keys: name, phone, website, description.
    Returns keys: business, opener, gatekeeper_script, owner_script,
    objection_responses, close.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing; add it to .env")

    name = str(lead.get("name", "")).strip()
    phone = str(lead.get("phone", "")).strip()
    website = str(lead.get("website", "")).strip()
    description = str(lead.get("description", "")).strip()

    if not all([name, phone, website, description]):
        raise ValueError(
            "lead must include non-empty values for name, phone, website, description"
        )

    client = Anthropic(api_key=api_key)

    prompt = f"""You write high-conversion cold call scripts for trades businesses.

Business details:
- Name: {name}
- Phone: {phone}
- Website: {website}
- Description: {description}

Create one script tailored to this business and return ONLY valid JSON.

The output must follow this exact structure and section goals:
1) Opener: short pattern interrupt, state your name and Bermuda AI, ask a qualifying question about their after-hours lead response.
2) Gatekeeper script: if someone other than the owner answers, a short bridge to get to the decision maker.
3) Owner script: if owner answers, diagnose their pain around losing leads after hours, quantify it in dollars or jobs lost, then include this exact offer text verbatim:
"{EXACT_OFFER}"
4) Objection responses: include responses for exactly these objections:
   - not interested
   - too busy
   - already have something
   - send me info
5) Close: lock a specific next step (short call or demo) and never leave without a time.

Personalization requirements:
- Mention the business name naturally.
- Use their description to make the script feel specific to their work.

Return ONLY this JSON shape with these exact keys:
{{
  "business": "<business name>",
  "opener": "<text>",
  "gatekeeper_script": "<text>",
  "owner_script": "<text>",
  "objection_responses": {{
    "not interested": "<text>",
    "too busy": "<text>",
    "already have something": "<text>",
    "send me info": "<text>"
  }},
  "close": "<text>"
}}
"""

    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1400,
        system=(
            "You are an AI agent for Bermuda AI, a Montreal-based AI automation agency that sells done-for-you conversion websites and automated lead response systems to skilled trades businesses (HVAC, plumbing, electrical, roofing, contracting) in Montreal. "
            "The founder's offer is: We help trades businesses in Montreal stop losing leads after hours through an automated response system that follows up with every inquiry within 5 minutes. The first company to respond gets the job and right now that's almost never you after 6pm. "
            "Key sales principles: sell outcomes not tools, one buyer one pain one outcome, diagnose before pitching, quantify pain in dollars and jobs lost, always close to a specific next step. "
            "The discovery call sequence is: open with control, diagnose real pain, quantify cost, qualify across fit/priority/ability/timeline, frame value, secure locked next step. "
            "Cold call structure: opener, gatekeeper bridge, owner diagnosis, offer delivery, objection handling, close to next step. "
            "Pricing: $1,500 CAD setup, $250/month retainer. "
            "Scripts must be sharp, outcome-focused, and follow the cold call pathway framework from opener through close to a locked next step. "
            "Follow instructions exactly. Output only valid JSON."
        ),
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts: list[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    raw = "".join(text_parts).strip()
    data = _parse_json_object(raw)

    required = {
        "business",
        "opener",
        "gatekeeper_script",
        "owner_script",
        "objection_responses",
        "close",
    }
    missing = required.difference(data.keys())
    if missing:
        raise ValueError(f"Model response missing required keys: {sorted(missing)}")

    return {
        "business": str(data["business"]).strip(),
        "opener": str(data["opener"]).strip(),
        "gatekeeper_script": str(data["gatekeeper_script"]).strip(),
        "owner_script": str(data["owner_script"]).strip(),
        "objection_responses": dict(data["objection_responses"]),
        "close": str(data["close"]).strip(),
    }


if __name__ == "__main__":
    fake_lead = {
        "name": "Electrique Tremblay",
        "phone": "514-555-0138",
        "website": "https://www.electrique-tremblay.example",
        "description": (
            "Montreal electrical contractor handling residential service calls, "
            "panel upgrades, lighting installs, and urgent troubleshooting."
        ),
    }
    script = generate_call_script(fake_lead)
    print(json.dumps(script, indent=2, ensure_ascii=False))
