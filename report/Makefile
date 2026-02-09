# Makefile for Colored MNIST CV Task Report
# ==========================================
#
# Usage:
#   make          - Build report.pdf
#   make clean    - Remove auxiliary files
#   make distclean - Remove all generated files including PDF

MAIN = main
PDF = report.pdf

# Default target
all: $(PDF)

# Build PDF using pdflatex + bibtex
$(PDF): $(MAIN).tex refs.bib
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	@if exist $(MAIN).pdf move /Y $(MAIN).pdf $(PDF)

# Alternative: use latexmk (if available)
latexmk:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex
	@if exist $(MAIN).pdf move /Y $(MAIN).pdf $(PDF)

# Clean auxiliary files
clean:
	del /Q *.aux *.log *.bbl *.blg *.toc *.out *.lof *.lot 2>nul || echo "Nothing to clean"

# Clean everything including PDF
distclean: clean
	del /Q $(PDF) $(MAIN).pdf 2>nul || echo "Nothing to clean"

.PHONY: all clean distclean latexmk
