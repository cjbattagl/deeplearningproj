.DEFAULT_GOAL := all

DRAFT_BUILD_DIR = draft

all: draft-build-dir $(DRAFT_BUILD_DIR)/proposal.pdf $(DRAFT_BUILD_DIR)/progress.pdf $(DRAFT_BUILD_DIR)/final.pdf 

.PHONY: draft-build-dir
draft-build-dir:
	test -d $(DRAFT_BUILD_DIR)/ || mkdir -p $(DRAFT_BUILD_DIR)

.PHONY: $(DRAFT_BUILD_DIR)/proposal.pdf
$(DRAFT_BUILD_DIR)/proposal.pdf:
	latexmk -f -pdf -output-directory=$(DRAFT_BUILD_DIR)/ proposal.tex

.PHONY: $(DRAFT_BUILD_DIR)/progress.pdf
$(DRAFT_BUILD_DIR)/progress.pdf:
	latexmk -f -pdf -output-directory=$(DRAFT_BUILD_DIR)/ progress.tex

.PHONY: $(DRAFT_BUILD_DIR)/final.pdf
$(DRAFT_BUILD_DIR)/final.pdf:
	latexmk -f -pdf -output-directory=$(DRAFT_BUILD_DIR)/ final.tex

clean:
	rm -rf $(DRAFT_BUILD_DIR)/ *~ core *.blg *.toc *.bbl *.out *.aux *.log

# eof
