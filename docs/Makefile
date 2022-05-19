# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -W
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
GENDIR        = source/generated
BUILDDIR      = build

ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees source


.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(GENDIR)/*

.PHONY: html
html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)