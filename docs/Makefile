# Makefile for Finch.ai LaTeX Technical Documentation

TARGET = main
SRC = $(TARGET).tex
SECTIONS = $(wildcard sections/*.tex)

.PHONY: all clean cleanall pdf view

all: pdf

pdf: $(TARGET).pdf

$(TARGET).pdf: $(SRC) $(SECTIONS)
	latexmk -pdf -interaction=nonstopmode $(TARGET).tex

clean:
	latexmk -c
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f sections/*.aux

cleanall: clean
	latexmk -C
	rm -f $(TARGET).pdf

view: pdf
	@which xdg-open > /dev/null && xdg-open $(TARGET).pdf || echo "PDF generated at $(TARGET).pdf"
