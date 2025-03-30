# UNT AI Assistant

This Chainlit-based AI assistant is tailored for the University of North Texas (UNT). It provides structured responses and specific guidance depending on the user's query. The application is powered by multiple specialized agents, each handling a different task.

---

## Overview

- **Multiple Agents**:

  1. EmailComposeAgent
  2. ResearchPaperAgent
  3. AcademicConceptsAgent
  4. RedirectAgent
  5. GeneralAgent

- **Seamless Agent Switching**: The application detects which agent is needed based on keywords in the user's query.
- **Structured, Actionable Responses**: Answers are provided in bullet-point form, including exact URLs when relevant.
- **Logging & Session**: Uses session context to store conversation history and active agent.

---

## Getting Started

1. Clone or download this repository.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Linux/Mac)
   venv\Scripts\activate     # (Windows)
   ```
