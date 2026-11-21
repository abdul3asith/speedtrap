# Charts: every script in charts/ reads results/ and writes to charts/out/.
# A script whose data is missing prints the modal command to produce it and is
# skipped. `make publish-charts` copies the output into the blog post and records
# which commit of this repo produced them (charts/SOURCE_COMMIT in the post folder).
# Hand-drawn diagrams live in the blog repo, not here.

BLOG ?= ../../basith/website/src/content/blog/overhead-bound-inference
PY   ?= uv run python

# charts/_*.py are shared helpers, not charts.
CHART_SCRIPTS := $(filter-out charts/_%,$(wildcard charts/*.py))

.PHONY: charts publish-charts

charts:
	@for s in $(CHART_SCRIPTS); do echo "→ $$s"; $(PY) $$s || exit 1; done

publish-charts: charts
	mkdir -p $(BLOG)/charts
	cp charts/out/*.png charts/out/*.svg $(BLOG)/charts/
	@sha=$$(git rev-parse --short HEAD); \
	if [ -n "$$(git status --porcelain -- charts results Makefile)" ]; then \
		sha="$$sha-dirty"; \
		echo "warning: charts/, results/ or Makefile has uncommitted changes; recording $$sha"; \
	fi; \
	echo "$$sha" > $(BLOG)/charts/SOURCE_COMMIT; \
	echo "published charts from $$sha -> $(BLOG)/charts"
