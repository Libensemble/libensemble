SOURCE = planning_doc
TEXFILES = *.tex
PICS = ../images
BIBFILES = ../bibs/*.bib

all: $(SOURCE).pdf 

$(SOURCE).pdf: $(TEXFILES) $(BIBFILES) $(PICS)
	pdflatex $(SOURCE).tex
	bibtex $(SOURCE)
	pdflatex $(SOURCE).tex
	pdflatex $(SOURCE).tex

#$(preprint).pdf: $(TEXFILES) 
#  pdflatex $(preprint).tex

#$(response).pdf: $(TEXFILES) 
#  pdflatex $(response).tex

view: $(SOURCE).pdf
	open $(SOURCE).pdf
clean:
	-rm -f  $(SOURCE).log $(SOURCE).aux $(SOURCE).bbl $(SOURCE).blg \
	$(SOURCE).pdf $(SOURCE).out *.backup *.tex~ *.log

