# Charts: every script in charts/ reads results/ and writes to charts/out/.
# `make publish-charts` copies them into the blog post and records which
# commit of this repo produced them (charts/SOURCE_COMMIT in the post folder).

BLOG ?= ../../basith/website/src/content/blog/overhead-bound-inference
PY   ?= uv run python

CHART_SCRIPTS := $(wildcard charts/*.py)

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
