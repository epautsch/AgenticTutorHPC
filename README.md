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
## How to Grab a Compute Node Using the MCP Host

Once the MCP host is running, you can instantly connect to the least-loaded GPU node by simply typing:

> **“grab me the best node available”**

It takes no Slurm job scripts, no guessing which node is free, and no manual SSH commands.

---

## What Happens Under the Hood

The MCP host will:

-  **Identify the least-utilized GPU node** in real time  
-  **SSH into that node automatically** on your behalf  
-  **Drop you straight into an interactive shell**, ready for work  



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


