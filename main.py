"""Bermuda Agent — menu-driven CLI for lead research, outreach, and discovery prep."""

from __future__ import annotations

import sys
from typing import Any

from dotenv import load_dotenv

from agents.discovery_agent import prepare_discovery_brief
from agents.outreach_agent import write_outreach_email
from agents.research_agent import research_leads

load_dotenv()


def _configure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def _prompt_nonempty(label: str) -> str:
    while True:
        s = input(f"{label}: ").strip()
        if s:
            return s
        print("Please enter a non-empty value.")


def _prompt_optional(label: str) -> str:
    return input(f"{label} (optional, press Enter to skip): ").strip()


def _run_research_leads() -> None:
    trade = _prompt_nonempty("Trade (e.g. plumbing, HVAC, electrical)")
    leads = research_leads(trade=trade)
    if not leads:
        print("\nNo leads returned.\n")
        return
    print()
    for i, lead in enumerate(leads, start=1):
        name = lead.get("name", "")
        phone = lead.get("phone", "")
        website = lead.get("website", "")
        description = lead.get("description", "")
        print(f"{i}. {name}")
        print(f"   Phone: {phone}")
        print(f"   Website: {website}")
        print(f"   Description: {description}")
        print()


def _run_outreach() -> None:
    name = _prompt_nonempty("Business name")
    phone = _prompt_optional("Phone")
    website = _prompt_optional("Website")
    desc = _prompt_nonempty("Description")
    lead: dict[str, Any] = {
        "name": name,
        "phone": phone,
        "website": website,
        "description": desc,
    }
    result = write_outreach_email(lead)
    subject = result.get("subject", "")
    body = result.get("body", "")
    print()
    print("Subject")
    print("-" * 40)
    print(subject)
    print()
    print("Body")
    print("-" * 40)
    print(body)
    print()


def _run_discovery() -> None:
    company = _prompt_nonempty("Company name")
    out = prepare_discovery_brief(company_name=company)
    brief = out.get("brief", "")
    questions = out.get("suggested_questions") or []
    print()
    print("Discovery brief")
    print("-" * 40)
    print(brief)
    print()
    print("Suggested questions")
    print("-" * 40)
    for i, q in enumerate(questions, start=1):
        print(f"{i}. {q}")
    print()


def _show_menu() -> None:
    print()
    print("Bermuda Agent")
    print("-" * 40)
    print("1) Research Leads")
    print("2) Write Outreach Email")
    print("3) Discovery Call Prep")
    print()


def main() -> None:
    _configure_stdout_utf8()

    while True:
        _show_menu()
        choice = input("Choose a mode (1–3), or q to quit: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("Goodbye.")
            return

        try:
            if choice == "1":
                _run_research_leads()
            elif choice == "2":
                _run_outreach()
            elif choice == "3":
                _run_discovery()
            else:
                print("Invalid choice. Enter 1, 2, 3, or q.\n")
                continue
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return
        except Exception as e:
            print(f"\nError: {e}\n")

        again = input("Run another mode? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("Goodbye.")
            return


if __name__ == "__main__":
    main()
