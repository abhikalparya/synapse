# Prefer repo-root .venv when present (see README setup).
PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: test eval eval-fast eval-quality eval-reliability eval-audit eval-analyze curriculum-check proposal-metrics

test:
	cd backend && ../$(PYTHON) -m pytest tests/ -q

curriculum-check:
	cd backend && ../$(PYTHON) -m app.evaluation.runner --curriculum-inventory-check
	cd backend && ../$(PYTHON) -m app.evaluation.runner --domain-coverage-report

eval:
	cd backend && ../$(PYTHON) -m app.evaluation.runner \
		--benchmark quality \
		--dataset ../data/eval/learning_graph_quality_v1.jsonl \
		--systems linear direct synapse \
		--repetitions 3 \
		--ops-latency-samples 5

eval-fast:
	cd backend && ../$(PYTHON) -m app.evaluation.runner \
		--benchmark quality \
		--dataset ../data/eval/learning_graph_quality_v1.jsonl \
		--systems linear direct synapse \
		--repetitions 1 \
		--limit 5 \
		--skip-ops-latency

eval-quality:
	cd backend && ../$(PYTHON) -m app.evaluation.runner \
		--benchmark quality \
		--systems linear direct synapse \
		--skip-ops-latency

eval-reliability:
	cd backend && ../$(PYTHON) -m app.evaluation.runner --benchmark reliability --no-llm

eval-audit:
	cd backend && ../$(PYTHON) -m app.evaluation.runner --benchmark audit --no-llm

eval-analyze:
	cd backend && ../$(PYTHON) -m app.evaluation.runner \
		--analyze ../results/canonical/2026-08-25_104759_quality_gpt-4o-mini_baseline.json \
		--dataset ../data/eval/learning_graph_quality_v1.jsonl

proposal-metrics:
	cd backend && ../$(PYTHON) -m app.evaluation.runner --proposal-metrics-only
