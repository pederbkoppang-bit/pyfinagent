"""Immutable governance layer for pyfinagent risk limits.

See `limits_schema.py` for the RiskLimits model + `load()` helper.
Changes to `limits.yaml` require a GPG-signed tag named
`limits-rotation-YYYYMMDD` -- enforced by CI in phase-4.9.1.
"""
