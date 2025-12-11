.PHONY: test test-quick test-cov

test:
	uv run --group test pytest tests/ -v

test-quick:
	uv run --group test pytest tests/ -q

test-cov:
	uv run --group test pytest tests/ --cov=hamux_torch --cov-report=term-missing
