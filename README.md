# Bermuda Agent

Bermuda Agent is an AI agent system for a **trades business AI agency**. It supports **lead research**, **outreach**, and **discovery prep**: finding and qualifying prospects, preparing outreach, and lining up discovery conversations—all scoped to how trades businesses actually operate.

This repository is a scaffold: implement workflow logic in `agents/`, integrations in `tools/`, and orchestration from `main.py` (e.g. with LangGraph).

## Setup

1. Create and activate a virtual environment (Windows PowerShell):

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env` and set `ANTHROPIC_API_KEY` and `GOOGLE_SHEETS_ID` (and any service-account paths you add for Google Sheets).

4. Run the entry point:

   ```powershell
   python main.py
   ```

## Layout

- `main.py` — application entry point
- `agents/` — research, outreach, and discovery agents
- `tools/` — search and Google Sheets tooling
