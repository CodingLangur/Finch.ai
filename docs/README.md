# Finch.ai Technical Documentation

This directory contains the comprehensive LaTeX technical documentation and architecture reference manual for **Finch.ai**.

The generated documentation is compiled into [`main.pdf`](main.pdf) (or readable directly in PDF viewers).

---

## Document Overview

- **Title**: *Finch.ai: Architecture and Technical Reference Manual*
- **Author**: Trinath Bhattacharya
- **Structure**:
  - `main.tex`: Master LaTeX document configuring packages, styling, typography, title page, and table of contents.
  - `sections/`: Modular chapters detailing each subsystem:
    1. `01_introduction.tex`: Motivation, Core Capabilities, System Specifications.
    2. `02_architecture.tex`: System Topology, Component Breakdown, Request Lifecycle.
    3. `03_context_compression.tex`: Headroom Pipeline, Token Economics, Protection Guarantees.
    4. `04_storage_and_search.tex`: SQLite WAL Schemas, FTS5 Triggers, `sqlite-vec`, RRF Mathematics, Zstandard Column Compression.
    5. `05_agent_mode_and_tools.tex`: Chat vs. Agent Mode, Multi-Turn Loop, Tool Schemas, Two-Tier Guardrail Matrix.
    6. `06_persona_and_memory.tex`: Dynamic Persona Management, XML Tag Protocol, Sliding Window Buffer.
    7. `07_telemetry_and_maintenance.tex`: Real-time Telemetry, SQLite VACUUM, Checkpointing, Cold Archiving.
    8. `08_cli_and_configuration.tex`: Interactive Slash Commands, CLI Startup Flags, Environment Config.
    9. `09_benchmarks_and_verification.tex`: Retrieval Accuracy Benchmarks (MRR), Storage Compression, Test Suite Architecture (127 tests).
    10. `10_developer_guide.tex`: Developer Setup, Adding Tools/Providers, Pre-Push GitHub Checklist, Variable Catalog.

---

## Building the Documentation

### Prerequisites
A working TeX Live installation with standard packages (`latexmk`, `pdflatex`, `tikz`, `tcolorbox`, `booktabs`, `listings`):

```bash
# Debian / Ubuntu
sudo apt update
sudo apt install texlive-latex-extra texlive-fonts-recommended latexmk
```

### Compilation

Using `make`:
```bash
# Compile to PDF
make

# Clean intermediate auxiliary files
make clean

# Clean all artifacts including generated PDF
make cleanall
```

Or using the helper script:
```bash
./build.sh
```

Or directly using `latexmk` / `pdflatex`:
```bash
latexmk -pdf main.tex
```
