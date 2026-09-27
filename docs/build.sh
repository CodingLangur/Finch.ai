#!/usr/bin/env bash
# Script to compile Finch.ai LaTeX Technical Documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Compiling Finch.ai LaTeX Documentation ==="

if command -v latexmk >/dev/null 2>&1; then
    echo "Using latexmk to build main.pdf..."
    latexmk -pdf -interaction=nonstopmode main.tex
elif command -v pdflatex >/dev/null 2>&1; then
    echo "Using pdflatex to build main.pdf..."
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
else
    echo "Error: Neither latexmk nor pdflatex found in PATH." >&2
    echo "Please install TeX Live (e.g. sudo apt install texlive-latex-extra texlive-fonts-recommended latexmk)." >&2
    exit 1
fi

echo "=== Build Complete: docs/main.pdf successfully generated ==="
