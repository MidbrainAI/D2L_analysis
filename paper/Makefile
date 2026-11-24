MAIN := doc_to_lora_analysis_query_conditioning_wrong_facts_and_distractor_interference
ICML_ZIP := https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip
ICML_SHA256 := 8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c
ICML_DIR := .styles
ICML_FILES := icml2026.sty icml2026.bst algorithm.sty algorithmic.sty fancyhdr.sty
export TEXINPUTS := $(abspath $(ICML_DIR)):$(TEXINPUTS)
export BSTINPUTS := $(abspath $(ICML_DIR)):$(BSTINPUTS)

.PHONY: all check styles require-styles
all: $(MAIN).pdf

styles:
	@set -eu; tmp=$$(mktemp -d); trap 'rm -f "$$tmp/icml2026.zip"; rmdir "$$tmp"' EXIT; \
		curl -fsSL "$(ICML_ZIP)" -o "$$tmp/icml2026.zip"; \
		printf '%s  %s\n' "$(ICML_SHA256)" "$$tmp/icml2026.zip" | sha256sum -c -; \
		mkdir -p "$(ICML_DIR)"; \
		unzip -o -j "$$tmp/icml2026.zip" $(ICML_FILES) -d "$(ICML_DIR)"

require-styles:
	@for file in $(ICML_FILES); do \
		test -f "$(ICML_DIR)/$$file" || { echo 'Missing ICML styles; run make -C paper styles first.' >&2; exit 1; }; \
	done

check:
	$(MAKE) -B all
	@if grep -E '(^!|Warning:.*undefined|Label\(s\) may have changed|Rerun to get cross-references right)' "$(MAIN).log"; then \
		exit 1; \
	fi

$(MAIN).pdf: $(MAIN).tex $(MAIN).bbl $(wildcard analysis/*.tex figures/*.tex $(ICML_DIR)/*.sty) | require-styles
	pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error $(MAIN).tex
