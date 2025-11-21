# Agentic HPC Tutor

_A lightweight, offline-friendly, multi-agent tutor for HPC lesson planning, concept explanation, code scaffolding, execution, and review._

This repo implements the system described in the accompanying paper, centered on a **Session agent** that coordinates specialist agents (Explainer, Quizzer, Coder, Reviewer) and an **Executor** that compiles/runs code and returns observations for self-correction. It’s designed to work on laptops or institutional clusters without proprietary APIs.

> **Project status:** early but usable. The code you see here is the minimal core. Additional folders (examples, jobs, scripts, tests, docs) are stubbed below for future population.

---

## Table of contents
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Running on HPC systems](#running-on-hpc-systems)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Security & privacy](#security--privacy)
- [License & citation](#license--citation)
- [Acknowledgments](#acknowledgments)

---

## Quick start

### 1) Environment
```bash
# clone
git clone https://github.com/epautsch/AgenticTutorHPC.git
cd AgenticTutorHPC

# python env (any of venv/conda/uv is fine)
python -m venv .venv && source .venv/bin/activate

# install deps
pip install --upgrade pip
pip install -r requirements.txt

# Need to link Hugging Face account and access token
huggingface-cli login

# run the MCP host
python mcp_host.py

# How to grab a compute node using the MCP host:
To instantly connect to the least-loaded GPU node, just type:

          “grab me the best node available”

# The model will automatically:
  - Determine which GPU node is the least utilized.
  - SSH into that node on your behalf.
  - Drop you directly into an interactive shell so you can start working immediately.
