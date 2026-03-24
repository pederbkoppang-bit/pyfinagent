# dev/ — Developer Utilities

Test and diagnostic scripts moved here to keep the project root clean.

## Running

All scripts expect to be run from the **project root** (`pyfinagent/`):

```bash
# Health check
.venv312/Scripts/python.exe dev/health_check.py

# API smoke test
.venv312/Scripts/python.exe dev/t_api_check.py

# Schema validation
.venv312/Scripts/python.exe dev/t_schema_test.py

# Vertex AI SDK test
.venv312/Scripts/python.exe dev/t_vertex.py

# Full mock backtest suite
.venv312/Scripts/python.exe -m dev.t_backtest_mock
```
