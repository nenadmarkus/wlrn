#
#
#

all: output clean

#
#
#

output:
	pdflatex wlrn
	bibtex wlrn
	pdflatex wlrn
	pdflatex wlrn

#
#
#

clean: output
	rm *.bbl
	rm *.brf *.aux *.log *.blg
